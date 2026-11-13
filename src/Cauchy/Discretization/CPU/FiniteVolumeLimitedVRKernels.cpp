#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    NumericalStatus LimitLVRTarget(RHSScalarView u, RHSScalarView points, RHSOutputView coefficients,
                                   Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                                   const ThermalBounds &bounds, real betaPower, real betaCutoff) noexcept
    {
        const auto finite = OMP::ForRowsStatus(0, points.RowCount(), [&](index iPoint) -> NumericalStatus
                                               {
            for (int j = 0; j < 5; ++j)
                if (!std::isfinite(points(iPoint, j)))
                    return NumericalStatus{NumericalError::InvalidState, 1, -1};
            return {}; });
        if (!finite)
            return finite;
        return OMP::ForRowsStatus(0, u.RowCount(), [&](index iCell) -> NumericalStatus
                                  {
            const auto value = Load(u, iCell);
            if (!(ThermallyAdmissible(value, bounds)))
                return NumericalStatus{NumericalError::InvalidState, 2, -1};
            struct MeanValues
            {
                FlowState value;
                real operator()(index, int j) const { return value[j]; }
            };
            const real beta = ReconstructionFraction(value, MeanValues{value}, points, offsets(iCell, 0), offsets(iCell + 1, 0),
                                                     bounds, betaPower, betaCutoff);
            if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                return NumericalStatus{NumericalError::InvalidState, 3, -1};
            if (beta < 1)
                for (index j = 0; j < coefficients.RowSize(iCell); ++j)
                    coefficients(iCell, j) *= beta;
            return {}; });
    }
    void FillLVRScalars(RHSScalarView u, RHSOutputView s, const PerfectGas &gas)
    {
        OMP::ForRows(0, s.RowCount(), [&](index iCell)
                     {
            const auto state = Load(u, iCell);
            s(iCell, 0) = gas.Pressure(state);
            s(iCell, 1) = state[0]; });
    }
    namespace
    {
        NumericalResult<std::array<real, 3>> FaceSensor(RHSScalarView samples, RHSFieldView means,
                                                        const LimitedVRFace &face, index iFace, const PerfectGas &gas,
                                                        const ThermalBounds &bounds, const CFV::LimitedVRSettings &settings) noexcept
        {
            const auto left = Load(samples, 2 * iFace);
            auto right = Load(samples, 2 * iFace + 1);
            if (face.changesFrame)
                right = RotateState(right, Eigen::Map<const Geom::tGPoint>(face.rotation.data()));
            if (!(ThermallyAdmissible(left, bounds) && ThermallyAdmissible(right, bounds)))
                return NumericalStatus{NumericalError::InvalidState, 1, iFace};
            const auto l = gas.Primitive(left), r = gas.Primitive(right);
            real unLeft = 0, unRight = 0;
            for (int d = 0; d < 3; ++d)
                unLeft += l.velocity[d] * face.normal[d], unRight += r.velocity[d] * face.normal[d];
            const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(l.density, r.density, l.pressure, r.pressure, unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
            // Current v3 uses mean-density support independently of the
            // finalized-O2 base contact gate. Pressure/velocity data here
            // are neutral placeholders for this density-only calculation.
            const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(means(face.cell, 0), means(face.other, 0), 1, 1, 0, 0, 1, 1, settings).contact;
            return std::array<real, 3>{sensor.shock, sensor.contact, support};
        }
    }
    NumericalStatus BuildLVRFaceSensors(RHSScalarView samples, RHSFieldView means,
                                        Runtime::ArrayView<const LimitedVRFace, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces,
                                        const PerfectGas &gas, const ThermalBounds &bounds, const CFV::LimitedVRSettings &settings,
                                        RHSOutputView sensors) noexcept
    {
        DNDS_assert(sensors.RowCount() == faces.RowCount() && sensors.Width() == 3);
        return OMP::ForRowsStatus(0, faces.RowCount(), [&](index iFace) -> NumericalStatus
                                  {
            const auto sensor = FaceSensor(samples, means, faces(iFace, 0), iFace, gas, bounds, settings);
            if (!sensor)
                return sensor.status;
            for (int j = 0; j < 3; ++j)
                sensors(iFace, j) = sensor.value[j];
            return {}; });
    }
    void GatherLVRBranches(Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                           Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceIndices,
                           RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base)
    {
        DNDS_assert(offsets.RowCount() == branches.RowCount() + 1 && base.RowCount() == branches.RowCount());
        OMP::ForRows(0, branches.RowCount(), [&](index iCell)
                     {
            std::array<real, 3> values{};
            for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
                for (int j = 0; j < 3; ++j)
                    values[j] = std::max(values[j], sensors(faceIndices(entry, 0), j));
            for (int j = 0; j < 3; ++j)
                branches(iCell, j) = values[j];
            for (int j = 0; j < 2; ++j)
                base(iCell, j) = values[j]; });
    }
    NumericalStatus BuildLVRBranches(RHSScalarView samples, RHSFieldView means,
                                     Runtime::ArrayView<const LimitedVRFace, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces,
                                     const PerfectGas &gas, const ThermalBounds &bounds,
                                     const CFV::LimitedVRSettings &settings, RHSOutputView g,
                                     Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> b) noexcept
    {
        OMP::ForRows(0, g.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 3; ++j)
                g(iCell, j) = 0; });
        // Serial anchor: each face updates two cell maxima. Production OMP
        // uses BuildLVRFaceSensors then GatherLVRBranches with explicit sensor
        // storage/incidence; separate input/output alone cannot avoid races.
        for (index iFace = 0; iFace < faces.RowCount(); ++iFace)
        {
            const auto &face = faces(iFace, 0);
            const auto sensor = FaceSensor(samples, means, face, iFace, gas, bounds, settings);
            if (!sensor)
                return sensor.status;
            for (const auto iCell : {face.cell, face.other})
                if (iCell < g.RowCount())
                {
                    g(iCell, 0) = std::max(g(iCell, 0), sensor.value[0]);
                    g(iCell, 1) = std::max(g(iCell, 1), sensor.value[1]);
                    g(iCell, 2) = std::max(g(iCell, 2), sensor.value[2]);
                }
        }
        OMP::ForRows(0, g.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 2; ++j)
                b(iCell, j) = g(iCell, j); });
        return {};
    }
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> b,
                      RHSOutputView s, int column)
    {
        OMP::ForRows(0, s.RowCount(), [&](index iCell)
                     { s(iCell, column) = std::max(s(iCell, column), b(iCell, column)); });
    }
    void PublishLVRPenalty(RHSScalarView g,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> h,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> a)
    {
        OMP::ForRows(0, g.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 2; ++j)
                h(iCell, j) = g(iCell, j);
            a(iCell, 0) = std::max(g(iCell, 0), g(iCell, 1)); });
    }
}
