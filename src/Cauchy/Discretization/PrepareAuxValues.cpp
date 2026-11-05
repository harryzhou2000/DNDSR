#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    using FieldView = R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows>;
    namespace
    {
        template <class Values>
        auto AuxiliaryArrays(Values &values)
        {
            using Array = std::conditional_t<std::is_const_v<Values>, const R::Array<real>, R::Array<real>>;
            return std::array<Array *, 7>{values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
                                          values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
                                          values.faceWaveJump ? &*values.faceWaveJump : nullptr,
                                          values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
                                          values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
                                          values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
                                          values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr};
        }
    }
    AuxiliaryValues FiniteVolume::BuildAuxiliaryValues(AuxiliaryRequest request) const
    {
        AuxiliaryValues values;
        const index faces = index(_data->stencilFaces.size());
        if (request.derived.ComponentNumber())
        {
            values._derivedThermodynamics = request.meanThermodynamics;
            if (request.meanThermodynamics)
                for (auto quantity : {DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed, DerivedQuantity::DiffusionBound})
                    request.derived.enabled[int(quantity)] = true;
            values.derivedOwned.emplace(Metrics().Space(), request.derived);
            values.derivedGhost.emplace(Metrics().Space(), request.derived, true);
        }
        if (request.meanThermodynamics && !values._derivedThermodynamics)
            values.meanThermodynamics.emplace(R::RowLayout::Dense(Metrics().NumCellProc(), 3));
        if (request.faceSpectralRadius)
            values.faceSpectralRadius.emplace(R::RowLayout::Dense(faces, 2));
        if (request.faceWaveJump)
            values.faceWaveJump.emplace(R::RowLayout::Dense(faces, 1));
        if (request.facePressureMinimum)
            values.facePressureMinimum.emplace(R::RowLayout::Dense(faces, 1));
        if (request.cellSpectralSum)
            values.cellSpectralSum.emplace(Metrics().BuildField(1));
        if (request.cellWaveJump)
            values.cellWaveJump.emplace(Metrics().BuildField(1));
        if (request.cellPressureMinimum)
            values.cellPressureMinimum.emplace(Metrics().BuildField(1));
        if ((request.cellSpectralSum && !request.faceSpectralRadius) || (request.cellWaveJump && !request.faceWaveJump) ||
            (request.cellPressureMinimum && !request.facePressureMinimum))
            values._faceScratch.emplace(R::RowLayout::Dense(faces, 3));
        return values;
    }
    const std::vector<index> &FiniteVolume::AuxiliaryFaces() const { return _data->stencilFaces; }
    struct FiniteVolume::AuxiliaryPreparation::State
    {
        std::shared_ptr<const FiniteVolume::Data> data;
        CFV::FVField source;
        AuxiliaryValues buffers;
        CFV::FiniteVolumeRead geometry;
        std::optional<R::Array<real>::ReadLease> owned, ghost;
        std::optional<EquationDofs> derivedSource;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::optional<BoundaryInputs> boundary;
        std::optional<R::Array<real>::ReadLease> boundaryRead;
        RHSScalarView incoming;
        std::array<std::optional<R::Array<real>::WriteLease>, 7> writes;
        std::array<R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>, 7> values;
        std::optional<R::Array<real>::WriteLease> scratchWrite;
        R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows> scratch;
        real time;
        RigidFrame frame;
        int phase = 0;
        bool started = false;

        State(std::shared_ptr<const FiniteVolume::Data> owner, const CFV::FVField &input, const AuxiliaryValues &output, real atTime, const BoundaryInputs *trace)
            : data(std::move(owner)), source(input), buffers(output), geometry(data->metrics.AcquireRead()), time(atTime), frame(data->motion.At(atTime))
        {
            data->partition.Validate();
            owned.emplace(source.Owned().Read());
            if (trace)
            {
                boundary = *trace;
                boundaryRead.emplace(boundary->flow.Read());
                incoming = boundaryRead->View<R::UniformTableRows>();
            }
            if (buffers.derivedOwned || buffers.derivedGhost)
                derivedSource.emplace(EquationDofs::Coupled(data->equation.Schema(), source));
            const auto arrays = AuxiliaryArrays(buffers);
            try
            {
                const std::optional<DerivedFields> *derived[]{&buffers.derivedOwned, &buffers.derivedGhost};
                for (int part = 0; part < 2; ++part)
                    if (*derived[part])
                    {
                        DNDS_check_throw_info((*derived[part])->GhostPart() == (part == 1), "derived auxiliary row part is incorrect");
                        (void)(*derived[part])->Current(*derivedSource, data->equation);
                        (*derived[part])->PreflightRefresh();
                    }
                if (buffers._derivedThermodynamics)
                {
                    DNDS_check_throw_info(buffers.derivedOwned && buffers.derivedGhost, "derived thermodynamics needs both row parts");
                    for (const auto *part : {&*buffers.derivedOwned, &*buffers.derivedGhost})
                        for (auto q : {DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed, DerivedQuantity::DiffusionBound})
                            DNDS_check_throw_info(part->Request().Has(q), "derived thermodynamic quantity is missing");
                }
                for (int k = 0; k < 7; ++k)
                    if (arrays[k])
                    {
                        writes[k].emplace(arrays[k]->Write());
                        values[k] = writes[k]->View<R::UniformTableRows>();
                    }
                const bool needScratch = (arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]);
                if (needScratch)
                {
                    scratchWrite.emplace(buffers._faceScratch->Write());
                    scratch = scratchWrite->View<R::UniformTableRows>();
                }
            }
            catch (...)
            {
                for (auto &write : writes)
                    if (write)
                        write->CancelUnmodified();
                if (scratchWrite)
                    scratchWrite->CancelUnmodified();
                throw;
            }
        }
        ~State()
        {
            if (!started)
            {
                for (auto &write : writes)
                    if (write)
                        write->CancelUnmodified();
                if (scratchWrite)
                    scratchWrite->CancelUnmodified();
            }
        }
        void Means(const FieldView &mean, index first, index last)
        {
            if (!writes[0])
                return;
            for (index iCell = first; iCell < last; ++iCell)
            {
                const auto u = Load(mean, iCell);
                DNDS_check_throw_info(data->equation.Gas().Admissible(u), "invalid mean in auxiliary preparation");
                const auto acoustic = data->equation.PrepareFluxState(u);
                values[0](iCell, 0) = acoustic.flow.pressure;
                values[0](iCell, 1) = acoustic.soundSpeed;
                values[0](iCell, 2) = data->equation.DiffusionScale(u);
            }
        }
        void Faces(const FieldView &mean, const std::vector<index> &faces)
        {
            const bool spectral = writes[1] || writes[4], wave = writes[2] || writes[5], pressure = writes[3] || writes[6];
            if (!spectral && !wave && !pressure)
                return;
            const auto &equation = data->equation;
            const auto &gas = equation.Gas();
            const auto metric = geometry.HostView();
            std::optional<MeanThermodynamicView> derived;
            if (buffers._derivedThermodynamics)
                derived.emplace(MeanThermodynamicView::Derived(*derivedOwned, derivedGhost ? &*derivedGhost : nullptr));
            const auto prepared = [&](index iCell, const FlowState &u)
            {
                if (derived)
                    return Riemann::AcousticState{{u, (*derived)(iCell, 0)}, (*derived)(iCell, 1)};
                if (writes[0])
                    return Riemann::AcousticState{{u, values[0](iCell, 0)}, values[0](iCell, 1)};
                DNDS_check_throw_info(gas.Admissible(u), "invalid mean in auxiliary preparation");
                return equation.PrepareFluxState(u);
            };
            for (const auto iAux : faces)
            {
                const auto iFace = data->stencilFaces[iAux], first = data->faceEntries[iAux];
                const auto &anchor = data->samples[first];
                const auto left = Load(mean, anchor.cell);
                const auto leftData = prepared(anchor.cell, left);
                const real area = data->auxiliaryFaceArea[iAux];
                real convection = 0, diffusion = 0, jump = 0, minimum = leftData.flow.pressure;
                const index count = metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                for (index q = 0; q < count; ++q)
                {
                    const auto &sample = data->samples[first + q];
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const Vector3 point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                    const auto gridVelocity = frame.GridVelocity(point);
                    const auto right = sample.other == UnInitIndex ? FiniteVolumeDetail::BoundaryExterior(sample, gas, left, time, frame, incoming)
                                                                   : RotateState(Load(mean, sample.other), sample.rotation);
                    const auto rightData = sample.other == UnInitIndex ? equation.PrepareFluxState(right) : prepared(sample.other, right);
                    DNDS_check_throw_info(rightData.flow.pressure > 0 && std::isfinite(rightData.soundSpeed),
                                          "invalid right mean in auxiliary preparation: face=" + std::to_string(iFace) +
                                              " cell=" + std::to_string(sample.cell) + " neighbor=" + std::to_string(sample.other) +
                                              " zone=" + std::to_string(sample.zone) + " pressure=" + std::to_string(rightData.flow.pressure) +
                                              " soundSpeed=" + std::to_string(rightData.soundSpeed) + " time=" + std::to_string(time));
                    if (spectral)
                    {
                        convection = std::max({convection, Riemann::SignalSpeed(leftData, normal, gridVelocity), Riemann::SignalSpeed(rightData, normal, gridVelocity)});
                        const real scaleL = derived ? (*derived)(sample.cell, 2) : writes[0] ? values[0](sample.cell, 2)
                                                                                             : equation.DiffusionScale(left);
                        const real scaleR = sample.other != UnInitIndex && derived ? (*derived)(sample.other, 2) : sample.other != UnInitIndex && writes[0] ? values[0](sample.other, 2)
                                                                                                                                                            : equation.DiffusionScale(right);
                        diffusion = std::max(diffusion, std::max(scaleL, scaleR) * area *
                                                            (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                    }
                    if (wave)
                        jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, normal));
                    if (pressure)
                        minimum = std::min(minimum, rightData.flow.pressure);
                }
                if (writes[1])
                {
                    values[1](iAux, 0) = convection;
                    values[1](iAux, 1) = diffusion;
                }
                if (writes[2])
                    values[2](iAux, 0) = jump;
                if (writes[3])
                    values[3](iAux, 0) = minimum;
                if (scratchWrite)
                {
                    scratch(iAux, 0) = convection + diffusion;
                    scratch(iAux, 1) = jump;
                    scratch(iAux, 2) = minimum;
                }
            }
        }
        void Cells(const std::vector<index> &cells)
        {
            if (!writes[4] && !writes[5] && !writes[6])
                return;
            const auto metric = geometry.HostView();
            const real rotationBound = Geom::tPoint{frame.angularVelocity[0], frame.angularVelocity[1], frame.angularVelocity[2]}.norm();
            for (const auto iCell : cells)
            {
                real sum = 0, wave = 0, pressure = std::numeric_limits<real>::max();
                for (index entry = data->auxiliaryCellOffsets[iCell]; entry < data->auxiliaryCellOffsets[iCell + 1]; ++entry)
                {
                    const auto iAux = data->auxiliaryCellFaces[entry];
                    if (writes[4])
                        sum += data->auxiliaryFaceArea[iAux] * (writes[1] ? values[1](iAux, 0) + values[1](iAux, 1) : scratch(iAux, 0));
                    if (writes[5])
                        wave = std::max(wave, writes[2] ? values[2](iAux, 0) : scratch(iAux, 1));
                    if (writes[6])
                        pressure = std::min(pressure, writes[3] ? values[3](iAux, 0) : scratch(iAux, 2));
                }
                if (writes[4])
                    values[4](iCell, 0) = sum + metric.GetCellVol(iCell) * rotationBound;
                if (writes[5])
                    values[5](iCell, 0) = wave;
                if (writes[6])
                    values[6](iCell, 0) = pressure;
            }
        }
        void Core()
        {
            DNDS_check_throw_info(phase == 0, "auxiliary core phase already executed");
            started = true;
            const FieldView mean{owned->View<R::UniformTableRows>(), {}};
            if (buffers.derivedOwned)
            {
                RefreshDerivedFields(*derivedSource, data->equation, *buffers.derivedOwned, true);
                derivedOwned.emplace(buffers.derivedOwned->ReadCurrent(*derivedSource, data->equation));
            }
            Means(mean, 0, data->metrics.NumCell());
            Faces(mean, data->auxiliaryCoreFaces);
            Cells(data->partition.Core());
            phase = 1;
        }
        void Dependent()
        {
            DNDS_check_throw_info(phase == 1, "auxiliary dependent phase requires completed core work");
            // Failure here (an active halo writer) has not started dependent
            // work. The caller may wait for that transfer and retry this phase.
            ghost.emplace(source.Ghost().Read());
            if (buffers.derivedGhost)
            {
                RefreshDerivedFields(*derivedSource, data->equation, *buffers.derivedGhost, true);
                derivedGhost.emplace(buffers.derivedGhost->ReadCurrent(*derivedSource, data->equation));
            }
            const FieldView mean{owned->View<R::UniformTableRows>(), ghost->View<R::UniformTableRows>()};
            Means(mean, data->metrics.NumCell(), data->metrics.NumCellProc());
            Faces(mean, data->auxiliaryDependentFaces);
            Cells(data->partition.HaloDependent());
            phase = 2;
        }
    };
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation &FiniteVolume::AuxiliaryPreparation::operator=(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation::~AuxiliaryPreparation() = default;
    void FiniteVolume::AuxiliaryPreparation::Apply(CFV::ReconstructionRegion region)
    {
        DNDS_check_throw_info(_state, "auxiliary preparation is finished or moved");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core || region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid auxiliary preparation region");
        if (region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core)
            _state->Core();
        if (region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::HaloDependent)
            _state->Dependent();
    }
    void FiniteVolume::AuxiliaryPreparation::Finish(AuxiliaryValues &output)
    {
        DNDS_check_throw_info(_state && _state->phase == 2, "cannot publish incomplete auxiliary preparation");
        const auto expected = AuxiliaryArrays(_state->buffers), actual = AuxiliaryArrays(output);
        for (int k = 0; k < 7; ++k)
            DNDS_check_throw_info(bool(expected[k]) == bool(actual[k]) && (!expected[k] || expected[k]->SharedStorage() == actual[k]->SharedStorage()),
                                  "auxiliary preparation output buffers were replaced");
        const std::optional<CFV::FVField> *a[]{&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum};
        const std::optional<CFV::FVField> *b[]{&_state->buffers.cellSpectralSum, &_state->buffers.cellWaveJump, &_state->buffers.cellPressureMinimum};
        for (int k = 0; k < 3; ++k)
        {
            DNDS_check_throw_info(!*a[k] || (*a[k])->Space().Same((*b[k])->Space()), "auxiliary preparation cell row space was replaced");
            DNDS_check_throw_info(!*a[k] || (*a[k])->Ghost().SharedStorage() == (*b[k])->Ghost().SharedStorage(), "auxiliary preparation ghost buffer was replaced");
        }
        for (auto &write : _state->writes)
            write.reset();
        _state->scratchWrite.reset();
        const auto &source = _state->source;
        AuxiliaryValues::Origin origin{_state->data, source.Owned().SharedStorage(), source.Ghost().SharedStorage(), source.Owned().Version(), source.Ghost().Version(),
                                       source.Owned().ShapeVersion(), source.Ghost().ShapeVersion(), _state->time};
        origin.boundary = _state->boundary;
        for (int k = 0; k < 7; ++k)
            if (actual[k])
            {
                origin.present[k] = true;
                origin.storage[k] = actual[k]->SharedStorage();
                origin.versions[k] = actual[k]->Version();
                origin.shapes[k] = actual[k]->ShapeVersion();
            }
        for (int k = 0; k < 3; ++k)
            origin.ghosts[k] = *a[k] ? (*a[k])->Ghost().SharedStorage() : nullptr;
        const std::optional<DerivedFields> *derived[]{&output.derivedOwned, &output.derivedGhost};
        const std::optional<DerivedFields> *preparedDerived[]{&_state->buffers.derivedOwned, &_state->buffers.derivedGhost};
        for (int part = 0; part < 2; ++part)
        {
            DNDS_check_throw_info(bool(*derived[part]) == bool(*preparedDerived[part]), "derived auxiliary selection changed");
            if (*derived[part])
            {
                origin.derived[part] = (*derived[part])->Values().SharedStorage();
                DNDS_check_throw_info(origin.derived[part] == (*preparedDerived[part])->Values().SharedStorage(), "derived auxiliary buffer replaced");
            }
        }
        output._origin = std::move(origin);
        output._equationOrigin.reset();
        _state.reset();
    }
    FiniteVolume::AuxiliaryPreparation FiniteVolume::BeginAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time, const BoundaryInputs *boundary) const
    {
        output._origin.reset();
        Metrics().ValidateField(state, 5);
        DNDS_check_throw_info(!HasInterfaces() || boundary, "auxiliary preparation needs an incoming interface packet");
        if (boundary)
            ValidateBoundaryInputs(*boundary, time);
        DNDS_check_throw_info(std::isfinite(time) && state.Device() == R::Device::Host(), "auxiliary preparation requires finite boundary time and available CPU inputs");
        const auto arrays = AuxiliaryArrays(output);
        const index faces = index(_data->stencilFaces.size());
        const index rows[7]{Metrics().NumCellProc(), faces, faces, faces, Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCell()};
        const index widths[7]{3, 2, 1, 1, 1, 1, 1};
        for (int k = 0; k < 7; ++k)
            if (arrays[k])
            {
                DNDS_check_throw_info(arrays[k]->GetDevice() == R::Device::Host() && arrays[k]->RowCount() == rows[k] &&
                                          arrays[k]->Layout().Width() == widths[k] && !arrays[k]->Layout().IsVariable(),
                                      "auxiliary output shape or backend mismatch");
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(arrays[k]->Storage(), state.Owned().Storage()) &&
                                          !Algebra::Detail::Overlaps(arrays[k]->Storage(), state.Ghost().Storage()),
                                      "auxiliary output aliases input state");
                for (int j = 0; j < k; ++j)
                    if (arrays[j])
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(arrays[k]->Storage(), arrays[j]->Storage()), "auxiliary outputs overlap");
            }
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
            if (*field)
                Metrics().ValidateField(**field, 1);
        const bool needScratch = (arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]);
        if (needScratch && (!output._faceScratch || output._faceScratch->RowCount() != faces))
            output._faceScratch.emplace(R::RowLayout::Dense(faces, 3));
        return AuxiliaryPreparation(std::make_unique<AuxiliaryPreparation::State>(_data, state, output, time, boundary));
    }
    void FiniteVolume::PrepareAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time, const BoundaryInputs *boundary) const
    {
        auto preparation = BeginAuxValues(state, output, time, boundary);
        preparation.Apply();
        preparation.Finish(output);
    }
    void FiniteVolume::ValidateAuxValues(const CFV::FVField &state, const AuxiliaryValues &values, real time) const
    {
        Metrics().Validate();
        Metrics().ValidateField(state, 5);
        DNDS_check_throw_info(values._origin, "auxiliary values have not been prepared");
        // An active writer has not published its next generation yet. A
        // version comparison alone must not accept that in-flight mutation.
        const auto ownedRead = state.Owned().Read(), ghostRead = state.Ghost().Read();
        const auto &origin = *values._origin;
        DNDS_check_throw_info(!HasInterfaces() || origin.boundary, "auxiliary publication lacks incoming interface provenance");
        if (origin.boundary)
            ValidateBoundaryInputs(*origin.boundary, time);
        DNDS_check_throw_info(origin.owner.get() == _data.get() && time == origin.time &&
                                  origin.ownedStorage == state.Owned().SharedStorage() && origin.ghostStorage == state.Ghost().SharedStorage() &&
                                  origin.ownedVersion == state.Owned().Version() && origin.ghostVersion == state.Ghost().Version() &&
                                  origin.ownedShape == state.Owned().ShapeVersion() && origin.ghostShape == state.Ghost().ShapeVersion(),
                              "stale auxiliary state, time or plan");
        const auto arrays = AuxiliaryArrays(values);
        for (int k = 0; k < 7; ++k)
        {
            DNDS_check_throw_info(bool(arrays[k]) == origin.present[k], "auxiliary output selection changed");
            if (arrays[k])
            {
                const auto read = arrays[k]->Read();
                DNDS_check_throw_info(origin.storage[k] == arrays[k]->SharedStorage() && origin.versions[k] == arrays[k]->Version() && origin.shapes[k] == arrays[k]->ShapeVersion(),
                                      "auxiliary output values changed");
            }
        }
        int iGhost = 0;
        for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
        {
            DNDS_check_throw_info(!*field || (*field)->Space().Same(state.Space()), "auxiliary cell row space changed");
            DNDS_check_throw_info(origin.ghosts[iGhost++] == (*field ? (*field)->Ghost().SharedStorage() : nullptr), "auxiliary ghost storage was rebound");
        }
        if (values.derivedOwned || values.derivedGhost || origin.derived[0] || origin.derived[1])
        {
            const auto typed = EquationDofs::Coupled(_data->equation.Schema(), state);
            const std::optional<DerivedFields> *derived[]{&values.derivedOwned, &values.derivedGhost};
            for (int part = 0; part < 2; ++part)
            {
                DNDS_check_throw_info(bool(origin.derived[part]) == bool(*derived[part]), "derived auxiliary selection changed");
                if (*derived[part])
                {
                    DNDS_check_throw_info((*derived[part])->GhostPart() == (part == 1), "derived auxiliary row part changed");
                    DNDS_check_throw_info(origin.derived[part] == (*derived[part])->Values().SharedStorage(), "derived auxiliary allocation changed");
                    const auto read = (*derived[part])->ReadCurrent(typed, _data->equation);
                }
            }
        }
    }
    real FiniteVolume::StableStep(const CFV::FVField &state, const AuxiliaryValues &values, real cfl, real time) const
    {
        real dt = std::numeric_limits<real>::max();
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "positive CFL required");
            ValidateAuxValues(state, values, time);
            DNDS_check_throw_info(values.cellSpectralSum, "CFL selection requires prepared cell spectral sums");
            auto geometry = Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            const auto read = values.cellSpectralSum->Owned().Read();
            const auto sums = read.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                const real radius = sums(iCell, 0);
                DNDS_check_throw_info(std::isfinite(radius) && radius > 0, "invalid cell spectral sum");
                dt = std::min(dt, cfl * metric.GetCellVol(iCell) / radius);
            } });
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, Metrics().Communicator());
        return dt;
    }
    real FiniteVolume::StableStep(const CFV::FVField &state, real cfl) const
    {
        // Convenience synchronous wrapper. Production Domain owns persistent
        // output buffers and its mean binding; the numerical operator is local.
        auto input = state;
        auto halo = Metrics().BindHalo(input);
        halo.Begin().Wait();
        AuxiliaryRequest request;
        request.meanThermodynamics = request.cellSpectralSum = true;
        auto values = BuildAuxiliaryValues(request);
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        { PrepareAuxValues(state, values); });
        return StableStep(state, values, cfl);
    }
}
