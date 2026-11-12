#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void FiniteVolume::SetBoundaryInputs(Workspace &work, const BoundaryInputs &input) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        work.incoming = AliasBoundaryInputs(input);
    }

    void FiniteVolume::SetBoundaryInputs(MeanWorkspace &work, const BoundaryInputs &input) const
    {
        DNDS_check_throw_info(work.work && work.work->owner == _data, "foreign equation mean workspace");
        work.work->incoming = AliasBoundaryInputs(input);
    }
    std::optional<BoundaryInputs> FiniteVolume::PreparedBoundaryInputs(
        const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data,
                              "foreign equation workspace");
        return work.incoming;
    }

}
