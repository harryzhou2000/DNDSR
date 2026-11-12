#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    Vector3 FiniteVolume::LastBoundaryForce(const Workspace &work, const std::vector<int> &zones) const
    {
        if (work.fixedFlowRHS)
        {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                                  "fixed-flow boundary force requires a completed RHS evaluation");
            auto force = work.diagnostics.BoundaryForce(zones);
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<real>{force.data(), 3}, Metrics().Communicator());
            return force;
        }
        Vector3 force{};
        const MPIInfo mpi(Metrics().Communicator());
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                "equation boundary force requires a completed RHS evaluation");
            for (const int zone : zones)
                DNDS_check_throw_info(_data->settings.boundaries.count(std::to_string(zone)), "force zone has no boundary condition");
            const auto read = work.boundaryMomentumFlux.Read();
            const auto values = read.View<Runtime::FixedRows<3>>();
            const auto stencilRead = _data->stencil.Read();
            const auto stencil = stencilRead.View();
            force = CPU::AccumulateBoundaryForce(stencil, values, {zones.data(), index(zones.size())}); });
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<real>{force.data(), 3}, mpi.comm);
        return force;
    }

    index FiniteVolume::LastFaceFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        if (work.fixedFlowRHS)
        {
            auto count = work.diagnostics.FaceFallbacks();
            count = MPI::AllreduceValue<MPI::Reduction::Sum>(count, Metrics().Communicator());
            return count;
        }
        return work.faceFallbacks[0] + work.faceFallbacks[1];
    }

    index FiniteVolume::LastCharacteristicFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.characteristicFallbacks;
    }

    const EquationDofs &FiniteVolume::LowOrderResidual(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        DNDS_check_throw_info(work.lowOrderResidual, "low-order residual requires the positivity workspace");
        return *work.lowOrderResidual;
    }

}
