#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/IO/Metis.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <map>
#include <cmath>
#include <limits>
#include <set>

namespace DNDS::Geom
{
    void UnstructuredMesh::PrepareNativeTopology(bool faces, bool edges, int ghostLayers)
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(_nativeState, "native topology preparation requires a native mesh"); });
        const auto stamp = versions.Stamp();
        auto result = NativeMeshPreparation::PrepareTopology(*_nativeState, mpi, faces, edges, ghostLayers);
        std::shared_ptr<UnstructuredMeshState> prepared;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            stamp.Validate();
            prepared = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*result)); });
        MeshMutationToken mutation(this, versions, Runtime::Change::Topology | Runtime::Change::Layout | Runtime::Change::Distribution | Runtime::Change::Allocation);
        mutation.Commit();
        prepared->BindPreparedEpoch(versions);
        _nativeState = std::move(prepared);
        _nativeSmoothing.reset();
        node2bndNode.clear();
        node2parentNode.clear();
        cell2parentCell.clear();
    }
}
