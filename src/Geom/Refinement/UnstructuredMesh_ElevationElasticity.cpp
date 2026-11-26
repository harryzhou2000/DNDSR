#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Geometry/NativeSmoothingHelpers.hpp"

namespace DNDS::Geom
{
    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV2()
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        std::shared_ptr<const UnstructuredMeshState> current;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            current = _nativeState.Load();
            DNDS_check_throw_info(current, "internal smoothing requires native mesh authority"); });
        {
            const auto stamp = versions.Stamp();
            std::shared_ptr<const NativeSmoothingWorkspace> work;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                work = _nativeSmoothing.Load();
                DNDS_check_throw_info(work && work->Source() == current, "native smoothing requires a fresh boundary workspace for the current mesh"); });
            auto prepared = NativeMeshSmoothing::Elasticity(*work, mpi, NativeSmoothingFacade::Options(elevationInfo));
            if (prepared != current)
                PublishNativeGeometry(std::move(prepared), stamp);
            _nativeSmoothing.Reset();
            return;
        }
    }

}
