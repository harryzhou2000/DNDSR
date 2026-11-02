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
        if (request.meanThermodynamics)
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
        return values;
    }
    const std::vector<index> &FiniteVolume::AuxiliaryFaces() const { return _data->stencilFaces; }
    void FiniteVolume::PrepareAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time) const
    {
        output._origin.reset();
        Metrics().ValidateField(state, 5);
        DNDS_check_throw_info(std::isfinite(time) && state.Device() == R::Device::Host(), "auxiliary preparation requires finite boundary time and available CPU inputs");
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
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
        const auto ownedRead = state.Owned().Read(), ghostRead = state.Ghost().Read();
        const FieldView mean{ownedRead.View<R::UniformTableRows>(), ghostRead.View<R::UniformTableRows>()};
        std::array<std::optional<R::Array<real>::WriteLease>, 7> writes;
        std::array<R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>, 7> values;
        try
        {
            for (int k = 0; k < 7; ++k)
                if (arrays[k])
                {
                    writes[k].emplace(arrays[k]->Write());
                    values[k] = writes[k]->View<R::UniformTableRows>();
                }
        }
        catch (...)
        {
            for (auto &write : writes)
                if (write)
                    write->CancelUnmodified();
            throw;
        }
        const auto &equation = _data->equation;
        const auto &gas = equation.Gas();
        if (arrays[0])
            for (index iCell = 0; iCell < Metrics().NumCellProc(); ++iCell)
            {
                const auto u = Load(mean, iCell);
                DNDS_check_throw_info(gas.Admissible(u), "invalid mean in auxiliary preparation");
                const auto data = equation.PrepareFluxState(u);
                values[0](iCell, 0) = data.flow.pressure;
                values[0](iCell, 1) = data.soundSpeed;
                values[0](iCell, 2) = equation.DiffusionScale(u);
            }
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            if (arrays[4])
                values[4](iCell, 0) = 0;
            if (arrays[5])
                values[5](iCell, 0) = 0;
            if (arrays[6])
                values[6](iCell, 0) = std::numeric_limits<real>::max();
        }
        const bool spectral = arrays[1] || arrays[4], wave = arrays[2] || arrays[5], pressure = arrays[3] || arrays[6];
        const auto prepared = [&](index iCell, const FlowState &u)
        {
            if (arrays[0])
                return Riemann::AcousticState{{u, values[0](iCell, 0)}, values[0](iCell, 1)};
            DNDS_check_throw_info(gas.Admissible(u), "invalid mean in auxiliary preparation");
            return equation.PrepareFluxState(u);
        };
        if (spectral || wave || pressure)
            for (index iAux = 0; iAux < faces; ++iAux)
            {
                const auto iFace = _data->stencilFaces[iAux], first = _data->faceEntries[iAux];
                const auto &anchor = _data->samples[first];
                const auto left = Load(mean, anchor.cell);
                const auto leftData = prepared(anchor.cell, left);
                real area = 0, convection = 0, diffusion = 0, jump = 0, minimum = leftData.flow.pressure;
                const index count = metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                for (index q = 0; q < count; ++q)
                    area += metric.GetFaceWeight(_data->samples[first + q].point);
                for (index q = 0; q < count; ++q)
                {
                    const auto &sample = _data->samples[first + q];
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto right = sample.other == UnInitIndex ? sample.boundary.Exterior(gas, left, normal, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time)
                                                                   : RotateState(Load(mean, sample.other), sample.rotation);
                    const auto rightData = sample.other == UnInitIndex ? equation.PrepareFluxState(right) : prepared(sample.other, right);
                    DNDS_check_throw_info(rightData.flow.pressure > 0 && std::isfinite(rightData.soundSpeed), "invalid boundary mean in auxiliary preparation");
                    if (spectral)
                    {
                        convection = std::max({convection, Riemann::SignalSpeed(leftData, normal), Riemann::SignalSpeed(rightData, normal)});
                        const real scaleL = arrays[0] ? values[0](sample.cell, 2) : equation.DiffusionScale(left);
                        const real scaleR = sample.other != UnInitIndex && arrays[0] ? values[0](sample.other, 2) : equation.DiffusionScale(right);
                        diffusion = std::max(diffusion, std::max(scaleL, scaleR) * area *
                                                            (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                    }
                    if (wave)
                        jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, normal));
                    if (pressure)
                        minimum = std::min(minimum, rightData.flow.pressure);
                }
                if (arrays[1])
                {
                    values[1](iAux, 0) = convection;
                    values[1](iAux, 1) = diffusion;
                }
                if (arrays[2])
                    values[2](iAux, 0) = jump;
                if (arrays[3])
                    values[3](iAux, 0) = minimum;
                for (index iCell : {anchor.cell, anchor.other})
                    if (iCell >= 0 && iCell < Metrics().NumCell())
                    {
                        if (arrays[4])
                            values[4](iCell, 0) += area * (convection + diffusion);
                        if (arrays[5])
                            values[5](iCell, 0) = std::max(values[5](iCell, 0), jump);
                        if (arrays[6])
                            values[6](iCell, 0) = std::min(values[6](iCell, 0), minimum);
                    }
            }
        for (auto &write : writes)
            write.reset();
        AuxiliaryValues::Origin origin{_data, state.Owned().Identity(), state.Ghost().Identity(), state.Owned().Version(), state.Ghost().Version(),
                                       state.Owned().ShapeVersion(), state.Ghost().ShapeVersion(), time};
        for (int k = 0; k < 7; ++k)
            if (arrays[k])
            {
                origin.present[k] = true;
                origin.identities[k] = arrays[k]->Identity();
                origin.versions[k] = arrays[k]->Version();
                origin.shapes[k] = arrays[k]->ShapeVersion();
            }
        output._origin = std::move(origin);
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
        DNDS_check_throw_info(origin.owner.get() == _data.get() && time == origin.time &&
                                  origin.ownedIdentity == state.Owned().Identity() && origin.ghostIdentity == state.Ghost().Identity() &&
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
                DNDS_check_throw_info(origin.identities[k] == arrays[k]->Identity() && origin.versions[k] == arrays[k]->Version() && origin.shapes[k] == arrays[k]->ShapeVersion(),
                                      "auxiliary output values changed");
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
