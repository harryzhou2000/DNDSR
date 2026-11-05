#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Geometry/NativeSmoothingHelpers.hpp"

namespace DNDS::Geom
{
    void UnstructuredMesh::ElevatedNodesSolveInternalSmooth()
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
            auto prepared = NativeMeshSmoothing::LocalRBF(*_nativeSmoothing, mpi, NativeSmoothingFacade::Options(elevationInfo));
            if (prepared != _nativeState)
                PublishNativeGeometry(std::move(prepared), stamp);
            _nativeSmoothing.reset();
            return;
        }
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1()
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
            auto prepared = NativeMeshSmoothing::GlobalRBF(*_nativeSmoothing, mpi, NativeSmoothingFacade::Options(elevationInfo));
            if (prepared != _nativeState)
                PublishNativeGeometry(std::move(prepared), stamp);
            _nativeSmoothing.reset();
            return;
        }
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1Old()
    {
        const auto communication = PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(false, "retired V1Old compact-kernel smoother has no equivalent native solver; use V1 global RBF explicitly"); });
    }
}
