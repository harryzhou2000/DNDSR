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
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(_nativeState, "internal smoothing requires native mesh authority"); });
        {
            const auto stamp = versions.Stamp();
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { DNDS_check_throw_info(_nativeSmoothing && _nativeSmoothing->Source() == _nativeState, "native smoothing requires a fresh boundary workspace for the current mesh"); });
            auto prepared = NativeMeshSmoothing::Elasticity(*_nativeSmoothing, mpi, NativeSmoothingFacade::Options(elevationInfo));
            if (prepared != _nativeState)
                PublishNativeGeometry(std::move(prepared), stamp);
            _nativeSmoothing.reset();
            return;
        }
    }

}
