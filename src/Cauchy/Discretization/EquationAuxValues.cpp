#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        template <class Values>
        auto EquationAuxiliaryArrays(Values &values)
        {
            return std::array<const R::Array<real> *, 9>{values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
                                                         values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
                                                         values.faceWaveJump ? &*values.faceWaveJump : nullptr,
                                                         values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
                                                         values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
                                                         values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
                                                         values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr,
                                                         values.derivedOwned ? &values.derivedOwned->Values() : nullptr,
                                                         values.derivedGhost ? &values.derivedGhost->Values() : nullptr};
        }
    }
    struct AuxiliaryValues::EquationOrigin
    {
        struct Stamp
        {
            R::ArrayStorage::Shared storage;
            uint64_t version = 0, shape = 0;
        };
        std::weak_ptr<const void> owner;
        EquationDofOperations::Binding source;
        std::vector<uint64_t> ghostVersions;
        std::array<Stamp, 9> outputs;
        std::array<R::ArrayStorage::Shared, 3> ghosts;
        real time;
        EquationOrigin(std::weak_ptr<const void> plan, const EquationDofs &state, const AuxiliaryValues &values, real atTime)
            : owner(std::move(plan)), source(state.NumericAlias()), time(atTime)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                ghostVersions.push_back(state.PhysicalField(iField).Ghost().Version());
            const auto arrays = EquationAuxiliaryArrays(values);
            for (int k = 0; k < 9; ++k)
                if (arrays[k])
                    outputs[k] = {arrays[k]->SharedStorage(), arrays[k]->Version(), arrays[k]->ShapeVersion()};
            int k = 0;
            for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
                ghosts[k++] = *field ? (*field)->Ghost().SharedStorage() : nullptr;
        }
    };

    struct FiniteVolume::EquationAuxiliaryPreparation::State
    {
        std::shared_ptr<const Data> data;
        EquationDofs source;
        AuxiliaryValues buffers;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        CFV::FiniteVolumeRead geometry;
        EquationDofs::ReadScope owned;
        std::optional<EquationDofs::ReadScope> complete;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::array<std::optional<R::Array<real>::WriteLease>, 7> writes;
        std::array<RHSOutputView, 7> values;
        std::optional<R::Array<real>::WriteLease> scratchWrite;
        RHSOutputView scratch;
        index flowFirst = -1, speciesFirst = -1;
        int independent;
        std::vector<index> vectorStarts;
        real time;
        RigidFrame frame;
        int phase = 0;
        bool started = false;

        State(std::shared_ptr<const Data> plan, const EquationDofs &state, const AuxiliaryValues &output, real atTime,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : data(std::move(plan)), source(state.NumericAlias()), buffers(output), workers(std::move(pool)),
              geometry(data->metrics.AcquireRead()), owned(source.ReadOwned()),
              independent(data->equation.HasMixture() ? data->equation.Mixture().SpeciesNumber() - 1 : 0),
              time(atTime), frame(data->motion.At(time))
        {
            index first = 0;
            for (const auto &block : source.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectorStarts.push_back(first + vector);
                first += index(block.components.size());
            }
            // Validate/reserve the backend before any numerical publication.
            {
                auto evaluator = Evaluator();
            }
            const auto arrays = EquationAuxiliaryArrays(buffers);
            try
            {
                for (const auto *derived : {&buffers.derivedOwned, &buffers.derivedGhost})
                    if (*derived)
                        (*derived)->PreflightRefresh();
                for (int k = 0; k < 7; ++k)
                    if (arrays[k])
                    {
                        auto handle = *arrays[k];
                        writes[k].emplace(handle.Write());
                        values[k] = writes[k]->View<R::UniformTableRows>();
                    }
                if ((arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]))
                {
                    scratchWrite.emplace(buffers._faceScratch->Write());
                    scratch = scratchWrite->View<R::UniformTableRows>();
                }
            }
            catch (...)
            {
                Cancel();
                throw;
            }
        }
        void Cancel()
        {
            for (auto &write : writes)
                if (write)
                    write->CancelUnmodified();
            if (scratchWrite)
                scratchWrite->CancelUnmodified();
        }
        ~State()
        {
            if (!started)
                Cancel();
            else
            {
                for (auto &write : writes)
                    if (write)
                        write->Fail();
                if (scratchWrite)
                    scratchWrite->Fail();
            }
        }
        EquationPointEvaluator Evaluator() const
        {
            return data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true, true, false});
        }
        void RefreshDerived(bool ghost)
        {
            auto &field = ghost ? buffers.derivedGhost : buffers.derivedOwned;
            if (!field)
                return;
            if (workers)
                RefreshDerivedFields(source, data->equation, *field, workers->Acquire(), true);
            else
                RefreshDerivedFields(source, data->equation, *field, true);
            (ghost ? derivedGhost : derivedOwned).emplace(field->ReadCurrent(source, data->equation));
        }
        std::array<real, 3> Properties(EquationFVDetail::PointScratch &point, EquationPointEvaluator &evaluator) const
        {
            const auto flow = point.Flow(flowFirst);
            const auto result = evaluator.Evaluate(flow, point.Species(speciesFirst, independent));
            return {result.thermo.pressure, result.thermo.soundSpeed, result.diffusionBound};
        }
        void Means(const std::vector<EquationBlockReadView> &blocks, index first, index last, EquationPointEvaluator &evaluator)
        {
            if (!writes[0])
                return;
            EquationFVDetail::PointScratch point(source.ComponentNumber());
            for (index iCell = first; iCell < last; ++iCell)
            {
                EquationFVDetail::Gather(blocks, iCell, point);
                const auto result = Properties(point, evaluator);
                for (int j = 0; j < 3; ++j)
                    values[0](iCell, j) = result[j];
            }
        }
        void Faces(const std::vector<EquationBlockReadView> &blocks, const std::vector<index> &faces, EquationPointEvaluator &evaluator)
        {
            const bool spectral = writes[1] || writes[4], wave = writes[2] || writes[5], pressure = writes[3] || writes[6];
            if (!spectral && !wave && !pressure)
                return;
            const auto metric = geometry.HostView();
            std::optional<MeanThermodynamicView> derived;
            if (buffers._derivedThermodynamics)
                derived.emplace(MeanThermodynamicView::Derived(*derivedOwned, derivedGhost ? &*derivedGhost : nullptr));
            EquationFVDetail::PointScratch left(source.ComponentNumber()), right(source.ComponentNumber());
            const auto prepared = [&](index iCell, auto &point)
            {
                if (derived)
                    return std::array<real, 3>{(*derived)(iCell, 0), (*derived)(iCell, 1), (*derived)(iCell, 2)};
                if (writes[0])
                    return std::array<real, 3>{values[0](iCell, 0), values[0](iCell, 1), values[0](iCell, 2)};
                return Properties(point, evaluator);
            };
            for (const auto iAux : faces)
            {
                const auto iFace = data->stencilFaces[iAux], first = data->faceEntries[iAux];
                const auto &anchor = data->samples[first];
                EquationFVDetail::Gather(blocks, anchor.cell, left);
                const auto leftProperties = prepared(anchor.cell, left);
                const Riemann::AcousticState leftData{{left.Flow(flowFirst), leftProperties[0]}, leftProperties[1]};
                real convection = 0, diffusion = 0, jump = 0, minimum = leftProperties[0];
                const auto count = metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                for (index q = 0; q < count; ++q)
                {
                    const auto &sample = data->samples[first + q];
                    if (sample.other == UnInitIndex)
                    {
                        right.values = left.values;
                        EquationFVDetail::AdiabaticBoundary(right, flowFirst, sample, frame, true);
                    }
                    else
                    {
                        EquationFVDetail::Gather(blocks, sample.other, right);
                        EquationFVDetail::RotatePoint(right, sample.rotation, vectorStarts);
                    }
                    const auto rightProperties = sample.other == UnInitIndex ? Properties(right, evaluator) : prepared(sample.other, right);
                    const Riemann::AcousticState rightData{{right.Flow(flowFirst), rightProperties[0]}, rightProperties[1]};
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    if (spectral)
                    {
                        convection = std::max({convection, Riemann::SignalSpeed(leftData, normal, grid), Riemann::SignalSpeed(rightData, normal, grid)});
                        diffusion = std::max(diffusion, std::max(leftProperties[2], rightProperties[2]) * data->auxiliaryFaceArea[iAux] *
                                                            (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                    }
                    if (wave)
                        jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, normal));
                    if (pressure)
                        minimum = std::min(minimum, rightProperties[0]);
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
            DNDS_check_throw_info(phase == 0, "equation auxiliary core already executed");
            started = true;
            RefreshDerived(false);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, owned);
            Means(blocks, 0, data->metrics.NumCell(), evaluator);
            Faces(blocks, data->auxiliaryCoreFaces, evaluator);
            Cells(data->partition.Core());
            phase = 1;
        }
        void Dependent()
        {
            DNDS_check_throw_info(phase == 1, "equation auxiliary dependent phase requires core");
            complete.emplace(source.ReadWithGhost()); // An unfinished halo is retriable before any dependent write.
            RefreshDerived(true);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, *complete);
            Means(blocks, data->metrics.NumCell(), data->metrics.NumCellProc(), evaluator);
            Faces(blocks, data->auxiliaryDependentFaces, evaluator);
            Cells(data->partition.HaloDependent());
            phase = 2;
        }
    };
    FiniteVolume::EquationAuxiliaryPreparation::EquationAuxiliaryPreparation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    FiniteVolume::EquationAuxiliaryPreparation::EquationAuxiliaryPreparation(EquationAuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::EquationAuxiliaryPreparation &FiniteVolume::EquationAuxiliaryPreparation::operator=(EquationAuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::EquationAuxiliaryPreparation::~EquationAuxiliaryPreparation() = default;
    void FiniteVolume::EquationAuxiliaryPreparation::Apply(CFV::ReconstructionRegion region)
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core || region == CFV::ReconstructionRegion::HaloDependent, "invalid auxiliary region");
        if (region != CFV::ReconstructionRegion::HaloDependent)
            _state->Core();
        if (region != CFV::ReconstructionRegion::Core)
            _state->Dependent();
    }
    void FiniteVolume::EquationAuxiliaryPreparation::Finish(AuxiliaryValues &output)
    {
        DNDS_check_throw_info(_state && _state->phase == 2, "cannot publish incomplete equation auxiliary preparation");
        const auto a = EquationAuxiliaryArrays(_state->buffers), b = EquationAuxiliaryArrays(output);
        for (int k = 0; k < 9; ++k)
            DNDS_check_throw_info(bool(a[k]) == bool(b[k]) && (!a[k] || a[k]->SharedStorage() == b[k]->SharedStorage()), "equation auxiliary buffers were replaced");
        int k = 0;
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
        {
            const std::optional<CFV::FVField> *original[]{&_state->buffers.cellSpectralSum, &_state->buffers.cellWaveJump, &_state->buffers.cellPressureMinimum};
            DNDS_check_throw_info(!*field || ((*field)->Space().Same(_state->source.Space()) && (*field)->Ghost().SharedStorage() == (*original[k])->Ghost().SharedStorage()), "equation auxiliary ghost/space changed");
            ++k;
        }
        for (auto &write : _state->writes)
            write.reset();
        _state->scratchWrite.reset();
        output._equationOrigin = std::make_shared<AuxiliaryValues::EquationOrigin>(_state->data, _state->source, output, _state->time);
        output._origin.reset();
        _state.reset();
    }
    FiniteVolume::EquationAuxiliaryPreparation FiniteVolume::BeginAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time,
                                                                            std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        Metrics().Validate();
        _data->partition.Validate();
        _data->equation.ValidateStateSchema(state);
        _data->equation.ResolveKernelProfile(state.Device());
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && std::isfinite(time), "equation auxiliary row space/time differs");
        DNDS_check_throw_info(_data->equation.HasMixture() == bool(workers), "equation auxiliary needs exactly its matching worker backend");
        for (const auto &sample : _data->samples)
            if (sample.other == UnInitIndex)
                DNDS_check_throw_info(sample.boundary.kind == BoundaryKind::Extrapolate || ((sample.boundary.kind == BoundaryKind::SlipWall || sample.boundary.kind == BoundaryKind::NoSlipWall) && !sample.boundary.isothermal), "unsupported equation auxiliary boundary");
        const auto arrays = EquationAuxiliaryArrays(output);
        const index faces = index(_data->stencilFaces.size());
        const index rows[9]{Metrics().NumCellProc(), faces, faces, faces, Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCellProc() - Metrics().NumCell()};
        const index widths[7]{3, 2, 1, 1, 1, 1, 1};
        for (int k = 0; k < 9; ++k)
            if (arrays[k])
            {
                DNDS_check_throw_info(arrays[k]->GetDevice() == R::Device::Host() && arrays[k]->RowCount() == rows[k] && !arrays[k]->Layout().IsVariable() && (k >= 7 || arrays[k]->Layout().Width() == widths[k]), "equation auxiliary output shape/device differs");
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    for (const auto *input : {&state.PhysicalField(iField).Owned(), &state.PhysicalField(iField).Ghost()})
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), arrays[k]->Storage()), "equation auxiliary output aliases state");
                for (int j = 0; j < k; ++j)
                    if (arrays[j])
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(arrays[k]->Storage(), arrays[j]->Storage()), "equation auxiliary outputs overlap");
            }
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
            if (*field)
                Metrics().ValidateField(**field, 1);
        for (int part = 0; part < 2; ++part)
        {
            const auto &derived = part ? output.derivedGhost : output.derivedOwned;
            if (derived)
            {
                DNDS_check_throw_info(derived->GhostPart() == bool(part), "equation derived row part differs");
                (void)derived->Current(state, _data->equation);
            }
        }
        if (output._derivedThermodynamics)
        {
            DNDS_check_throw_info(output.derivedOwned && output.derivedGhost, "derived thermodynamics needs both row parts");
            for (const auto *field : {&*output.derivedOwned, &*output.derivedGhost})
                for (auto q : {DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed, DerivedQuantity::DiffusionBound})
                    DNDS_check_throw_info(field->Request().Has(q), "derived thermodynamic column missing");
        }
        const bool needScratch = (arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]);
        if (needScratch && (!output._faceScratch || output._faceScratch->RowCount() != faces))
            output._faceScratch.emplace(R::RowLayout::Dense(faces, 3));
        auto prepared = std::make_unique<EquationAuxiliaryPreparation::State>(_data, state, output, time, std::move(workers));
        output._equationOrigin.reset();
        output._origin.reset();
        return EquationAuxiliaryPreparation(std::move(prepared));
    }
    void FiniteVolume::PrepareAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time,
                                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        auto scope = BeginAuxValues(state, output, time, std::move(workers));
        scope.Apply();
        scope.Finish(output);
    }
    void FiniteVolume::ValidateAuxValues(const EquationDofs &state, const AuxiliaryValues &values, real time) const
    {
        Metrics().Validate();
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && values._equationOrigin && !values._origin, "equation auxiliary publication is missing or belongs to another path");
        const auto read = state.ReadWithGhost();
        const auto &origin = *values._equationOrigin;
        DNDS_check_throw_info(origin.owner.lock().get() == _data.get() && origin.time == time, "equation auxiliary plan/time is stale");
        origin.source.Check(state, true);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(origin.ghostVersions[iField] == state.PhysicalField(iField).Ghost().Version(), "equation auxiliary ghosts changed");
        const auto arrays = EquationAuxiliaryArrays(values);
        for (int k = 0; k < 9; ++k)
        {
            const auto &stamp = origin.outputs[k];
            DNDS_check_throw_info(bool(stamp.storage) == bool(arrays[k]), "equation auxiliary selection changed");
            if (arrays[k])
            {
                const auto pin = arrays[k]->Read();
                DNDS_check_throw_info(stamp.storage == arrays[k]->SharedStorage() && stamp.version == arrays[k]->Version() && stamp.shape == arrays[k]->ShapeVersion(), "equation auxiliary output changed");
            }
        }
        int k = 0;
        for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
        {
            DNDS_check_throw_info(!*field || (*field)->Space().Same(state.Space()), "equation auxiliary cell space changed");
            DNDS_check_throw_info(origin.ghosts[k++] == (*field ? (*field)->Ghost().SharedStorage() : nullptr), "equation auxiliary cell ghost rebound");
        }
        for (const auto *field : {&values.derivedOwned, &values.derivedGhost})
            if (*field)
            {
                const auto pin = (*field)->ReadCurrent(state, _data->equation);
            }
    }
}
