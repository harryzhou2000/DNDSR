#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    PhysicalStepLimit FiniteVolume::LimitPhysicalStep(const CFV::FVField &state, const CFV::FVField &residual,
                                                      real dt, real relax, real scale, int compressionPower) const
    {
        namespace R = Runtime;
        const MPIInfo mpi(Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            Metrics().ValidateField(state, 5); Metrics().ValidateField(residual, 5);
            DNDS_check_throw_info(state.Device() == R::Device::Host() && residual.Device() == R::Device::Host() &&
                                      std::isfinite(dt) && dt > 0 && std::isfinite(relax) && relax > 0 && relax <= 1 &&
                                      std::isfinite(scale) && scale > 0 && compressionPower > 0 && compressionPower <= INT_MAX / 2,
                                  "invalid physical timestep PP inputs or unavailable backend");
            for (const auto *input : {&state.Owned(), &state.Ghost()})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), residual.Owned().Storage()), "physical timestep residual aliases its state"); });
        real agreement[]{dt, -dt, relax, -relax, scale, -scale, real(compressionPower), -real(compressionPower)};
        MPI_Allreduce(MPI_IN_PLACE, agreement, 8, MPI_DOUBLE, MPI_MIN, mpi.comm);
        for (int j = 0; j < 8; j += 2)
            DNDS_check_throw_info(agreement[j] == -agreement[j + 1], "physical timestep PP controls differ between ranks");
        PhysicalStepLimit result{dt};
        const auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        MPI::LocalStage(mpi, [&]
                        {
            const auto meanRead = state.Owned().Read(), residualRead = residual.Owned().Read();
            const auto mean = meanRead.View<R::UniformTableRows>(), rhs = residualRead.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
            {
                FlowState value{}, increment{};
                for (int j = 0; j < 5; ++j)
                {
                    value[j] = mean(iCell, j);
                    increment[j] = (dt * scale) * rhs(iCell, j) / metric.GetCellVol(iCell);
                    DNDS_check_throw_info(std::isfinite(increment[j]), "nonfinite physical timestep PP trial increment");
                }
                DNDS_check_throw_info(ThermallyAdmissible(value, ThermalConstraints()), "physical timestep PP requires admissible means");
                // Euler's physical-step caller uses compress=1; RHS/dTau and
                // accepted-correction controls use compress=2 elsewhere.
                const real alpha = FlowIncrementFraction(value, increment, ThermalConstraints(), relax, 1, compressionPower);
                DNDS_check_throw_info(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid physical timestep PP fraction");
                if (alpha < 1)
                {
                    ++result.limitedCells;
                    result.minimumAlpha = std::min(result.minimumAlpha, alpha);
                }
            } });
        MPI_Allreduce(MPI_IN_PLACE, &result.minimumAlpha, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &result.limitedCells, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        if (result.limitedCells)
            result.dt *= result.minimumAlpha;
        return result;
    }
}
