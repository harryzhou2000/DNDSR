#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        void CheckStepControls(real dt, real relax, real scale, int power)
        {
            DNDS_check_throw_info(std::isfinite(dt) && dt > 0 && std::isfinite(relax) && relax > 0 && relax <= 1 &&
                                      std::isfinite(scale) && scale > 0 && power > 0 && power <= INT_MAX / 2,
                                  "invalid physical timestep PP controls");
        }
        void AgreeStepControls(const MPIInfo &mpi, real dt, real relax, real scale, int power)
        {
            real values[]{dt, -dt, relax, -relax, scale, -scale, real(power), -real(power)};
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<real>{values, 8}, mpi.comm);
            for (int j = 0; j < 8; j += 2)
                DNDS_check_throw_info(values[j] == -values[j + 1], "physical timestep PP controls differ between ranks");
        }
        void ReduceStepLimit(PhysicalStepLimit &result, const MPIInfo &mpi)
        {
            result.minimumAlpha = MPI::AllreduceValue<MPI::Reduction::Min>(
                result.minimumAlpha, mpi.comm);
            result.limitedCells = MPI::AllreduceValue<MPI::Reduction::Sum>(
                result.limitedCells, mpi.comm);
            if (result.limitedCells)
                result.dt *= result.minimumAlpha;
        }
    }
    PhysicalStepLimit FiniteVolume::LimitPhysicalStep(const EquationDofs &state, const EquationDofs &residual,
                                                      const ThermalView &thermal, real dt, real relax, real scale, int compressionPower) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            Metrics().Validate(); state.RequireSchema(Equation().Schema());
            state.RequireCompatible(residual); EquationDofOperations::RequireDistinct(state, residual);
            CheckStepControls(dt, relax, scale, compressionPower);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && state.Device() == Runtime::Device::Host() &&
                                      residual.Device() == Runtime::Device::Host() && thermal.components == state.ComponentNumber(),
                                  "physical timestep equation space/backend/thermal width differs");
            index first = 0, flowFirst = -1, speciesFirst = -1, independent = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow") flowFirst = first;
                if (block.name == "species") { speciesFirst = first; independent = index(block.components.size()); }
                first += index(block.components.size());
            }
            DNDS_check_throw_info(thermal.flowFirst == flowFirst && thermal.speciesFirst == speciesFirst && thermal.independent == independent,
                                  "physical timestep thermal component mapping differs");
            DNDS_check_throw_info(std::isfinite(thermal.densityFloor) && thermal.densityFloor > 0,
                                  "invalid physical timestep density bound");
            for (int c = 0; c < 2; ++c)
                DNDS_check_throw_info(std::isfinite(thermal.floors[c]) && thermal.floors[c] >= 0 &&
                                          std::isfinite(thermal.densityCoefficients[c]) &&
                                          (!thermal.speciesEnergies[c].empty()) == Equation().HasMixture(),
                                      "physical timestep thermal coefficients are missing or invalid"); });
        AgreeStepControls(mpi, dt, relax, scale, compressionPower);
        PhysicalStepLimit result{dt};
        const auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        MPI::LocalStage(mpi, [&]
                        {
            const auto meanRead = state.ReadOwned(), rhsRead = residual.ReadOwned();
            const auto meanLayout = meanRead.ComponentLayout(), rhsLayout = rhsRead.ComponentLayout();
            const auto mean = meanLayout.View(), rhs = rhsLayout.View();
            const auto resultLocal = RequireNumericalSuccess(CPU::LimitPhysicalStepRows(mean, rhs, metric, thermal, dt, relax, scale, compressionPower), "LimitPhysicalStepRows");
            result.limitedCells = resultLocal.first; result.minimumAlpha = resultLocal.second; });
        ReduceStepLimit(result, mpi);
        return result;
    }
}
