#include "Mesh.hpp"
#include "NativeMeshRefinement.hpp"
#include "Mesh_NativeSmoothingHelpers.hpp"

namespace DNDS::Geom
{
    void UnstructuredMesh::BuildO2FromO1Elevation(UnstructuredMesh &source)
    {
        CheckCollectiveMutationAvailable(true);
        std::optional<Runtime::VersionTracker::ReadLease> retained;
        std::shared_ptr<const NativeMeshState> input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(!versions.SameController(source.versions), "derivation destination requires an independent mesh controller");
            DNDS_check_throw_info(source.HasNativeState() && source.getDim() == getDim(), "native derivation requires matching dimension and native source");
            int relation = MPI_UNEQUAL;
            DNDS_check_throw_info(MPI_Comm_compare(mpi.comm, source.mpi.comm, &relation) == MPI_SUCCESS &&
                                  (relation == MPI_IDENT || relation == MPI_CONGRUENT), "native derivation communicator differs");
            retained.emplace(source.versions.Read());
            input = source.CurrentNativeState(); });
        auto result = NativeMeshRefinement::Elevate(*input, mpi, {}, true);
        ReplaceNativeForSetup(std::move(result.state));
    }

    void UnstructuredMesh::BuildBisectO1FormO2(UnstructuredMesh &source)
    {
        CheckCollectiveMutationAvailable(true);
        std::optional<Runtime::VersionTracker::ReadLease> retained;
        std::shared_ptr<const NativeMeshState> input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(!versions.SameController(source.versions), "derivation destination requires an independent mesh controller");
            DNDS_check_throw_info(source.HasNativeState() && source.getDim() == getDim(), "native derivation requires matching dimension and native source");
            int relation = MPI_UNEQUAL;
            DNDS_check_throw_info(MPI_Comm_compare(mpi.comm, source.mpi.comm, &relation) == MPI_SUCCESS &&
                                  (relation == MPI_IDENT || relation == MPI_CONGRUENT), "native derivation communicator differs");
            retained.emplace(source.versions.Read());
            input = source.CurrentNativeState(); });
        auto result = NativeMeshRefinement::Bisect(*input, mpi);
        ReplaceNativeForSetup(std::move(result.state));
    }

    void UnstructuredMesh::ElevatedNodesGetBoundarySmooth(const std::function<bool(t_index)> &FiFBndIdNeedSmooth)
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(_nativeState, "boundary smoothing requires native mesh authority"); });
        {
            CheckCollectiveMutationAvailable(true);
            const auto stamp = versions.Stamp();
            auto work = NativeMeshSmoothing::PrepareBoundary(_nativeState, mpi, NativeSmoothingFacade::Options(elevationInfo), FiFBndIdNeedSmooth);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { stamp.Validate(); });
            _nativeSmoothing = std::move(work);
            return;
        }
    }
}
