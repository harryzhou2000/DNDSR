#include "Mesh.hpp"
#include "Mesh_NativeSmoothingHelpers.hpp"

namespace DNDS::Geom
{
    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV2()
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(_nativeState, "internal smoothing requires native mesh authority"); });
        {
            CheckCollectiveMutationAvailable(true);
            const auto stamp = versions.Stamp();
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(_nativeSmoothing && _nativeSmoothing->Source() == _nativeState, "native smoothing requires a fresh boundary workspace for the current mesh"); });
            auto prepared = NativeMeshSmoothing::Elasticity(*_nativeSmoothing, mpi, NativeSmoothingFacade::Options(elevationInfo));
            if (prepared != _nativeState)
                PublishNativeGeometry(std::move(prepared), stamp);
            _nativeSmoothing.reset();
            return;
        }
    }

}
