#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include <cmath>
#include "Cauchy/State/CPU/FieldKernels.hpp"

namespace DNDS::Cauchy::NonlinearDetail
{
    /// Freeze Euler's mean-state GMRES component metric before the linear
    /// solve. The preparation is collective; it reads only owned physical
    /// means and never changes halos or the equation. It is independent of
    /// linear block storage, preconditioner and logical field grouping.
    template <class Values, class Geometry>
    std::vector<real> MeanStateScales(Values values, Geometry metric, index components,
                                      index flowFirst, MPI_Comm comm)
    {
        std::vector<real> sum(size_t(components + 1), 0);
        CPU::AccumulateMeanScale(values, metric, components, {sum.data(), index(sum.size())});
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<real>{sum.data(), sum.size()}, comm);
        DNDS_check_throw_info(std::isfinite(sum[components]) && sum[components] > 0, "invalid mean-state metric volume");
        for (index j = 0; j < components; ++j)
            sum[j] /= sum[components] + verySmallReal;
        real squaredMomentum = 0;
        for (int d = 0; d < 3; ++d)
            squaredMomentum += sum[flowFirst + 1 + d] * sum[flowFirst + 1 + d];
        const real momentum = std::sqrt(squaredMomentum);
        for (int d = 0; d < 3; ++d)
            sum[flowFirst + 1 + d] = momentum;
        sum[flowFirst + 4] = momentum * momentum / (sum[flowFirst] + verySmallReal);
        sum.resize(size_t(components));
        // Same numerical zero regularization as Euler's meanScaleInv. This
        // defines the requested metric; it is not a physical-state fallback.
        for (auto &scale : sum)
        {
            scale += verySmallReal;
            DNDS_check_throw_info(std::isfinite(scale) && scale > 0 && std::isfinite(1 / scale), "invalid mean-state component scale");
        }
        return sum;
    }

    inline std::vector<real> LinearScales(const ImplicitSettings &settings, const CFV::FVField &state,
                                          const CFV::FiniteVolumePlan &geometry)
    {
        if (settings.linear.method != "gmres" || settings.linear.componentScaling == "none")
            return {};
        DNDS_check_throw_info(settings.linear.componentScaling == "meanState" && state.Width() == 5, "mean-state metric requires flow5");
        const auto read = state.Owned().Read();
        const auto metric = geometry.AcquireRead();
        return MeanStateScales(read.View<Runtime::UniformTableRows>(), metric.HostView(), 5, 0, geometry.Communicator());
    }

    inline std::vector<real> LinearScales(const ImplicitSettings &settings, const EquationDofs &state,
                                          const CFV::FiniteVolumePlan &geometry)
    {
        if (settings.linear.method != "gmres" || settings.linear.componentScaling == "none")
            return {};
        const auto indices = state.Schema().ResolveComponents({"flow.rho", "flow.rhou", "flow.rhov", "flow.rhow", "flow.rhoE"});
        const index first = indices[0];
        for (int j = 0; j < 5; ++j)
            DNDS_check_throw_info(indices[j] == first + j, "mean-state metric requires an ordered compressible flow5 block");
        const auto read = state.ReadOwned();
        const auto layout = read.ComponentLayout();
        const auto metric = geometry.AcquireRead();
        return MeanStateScales(layout.View(), metric.HostView(), state.ComponentNumber(), first, geometry.Communicator());
    }
}
