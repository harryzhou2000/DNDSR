#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <nlohmann/json.hpp>

namespace DNDS::Geom
{
    MeshReadAccess UnstructuredMesh::AcquireRead(MeshReadRequirements requirements, const Runtime::ExecutionStream &stream)
    {
        auto execution = stream.BorrowContext();
        auto result = AcquireRead(requirements, execution.Context());
        result._execution.emplace(std::move(execution));
        return result;
    }
    MeshReadAccess UnstructuredMesh::AcquireRead(MeshReadRequirements requirements, Runtime::ExecutionContext context)
    {
        DNDS_check_throw_info(_nativeState, "mesh read requires native authority");
        auto sourceRead = versions.Read();
        _nativeState->CheckRequirements(requirements);
        MeshReadAccess result(_nativeState->Snapshot(context.device), context, std::move(sourceRead), versions.Stamp());
        result._nativeOwner = _nativeState;
        result.Restrict(requirements);
        return result;
    }

    std::shared_ptr<UnstructuredMesh> UnstructuredMesh::FromNative(const MPIInfo &mpi,
                                                                   std::shared_ptr<const UnstructuredMeshState> state)
    {
        DNDS_check_throw_info(state, "native mesh state is missing");
        state->Validate();
        const auto nodes = state->Entities(EntityKind::Node);
        EntitySpace::CheckCommunicator(nodes._mpi.comm, mpi.comm);
        const auto communication = state->PreparationCommunication();
        const auto retainedMPI = communication ? MPIInfo(communication->Communicator()) : mpi;
        auto result = std::make_shared<UnstructuredMesh>(retainedMPI, state->BorrowHostView().Dimension());
        result->_mpiOwner = communication;
        result->_entityIdentity = nodes._identity;
        result->isPeriodic = state->BorrowHostView().IsPeriodic();
        result->periodicInfo = state->BorrowHostView().Periodic();
        result->_nativeState = state->BindEpoch(result->versions);
        result->_nativeSmoothing.reset();
        return result;
    }

    void UnstructuredMesh::ReplaceNativeForSetup(std::shared_ptr<const UnstructuredMeshState> source)
    {
        const auto communication = source ? source->PreparationCommunication() : PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        std::shared_ptr<UnstructuredMeshState> prepared;
        std::shared_ptr<const uint8_t> identity;
        Periodicity periodic;
        bool nextPeriodic = false;
        MPIInfo nextMPI = mpi;
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            int actualRank = -1, actualSize = 0;
            MPI_Comm_rank(mpi.comm, &actualRank);
            MPI_Comm_size(mpi.comm, &actualSize);
            DNDS_check_throw_info(actualRank == mpi.rank && actualSize == mpi.size && source,
                                  "native setup replacement requires valid MPIInfo and source");
            source->Validate();
            if (communication)
                nextMPI = MPIInfo(communication->Communicator());
            const auto nodes = source->Entities(EntityKind::Node);
            EntitySpace::CheckCommunicator(nodes._mpi.comm, mpi.comm);
            const auto &view = source->BorrowHostView();
            DNDS_check_throw_info(view.Dimension() == getDim(), "native setup replacement dimension differs");
            signature = source->MeshNamespace() + ":" + std::to_string(view.Dimension()) + ":" + std::to_string(view.IsPeriodic());
            for (const auto &[kind, family] : source->_spaces)
            {
                EntitySpace::CheckCommunicator(family.identity._mpi.comm, mpi.comm);
                signature += "k" + std::to_string(int(kind)) + ":";
                for (auto count : family.global->RLengths()) signature += std::to_string(count) + ",";
            }
            for (const auto &[kind, next] : source->IdHighWater()) signature += "i" + std::to_string(int(kind)) + ":" + std::to_string(next) + ";";
            const auto r = source->Requirements();
            signature += "q" + std::to_string(r.faces) + std::to_string(r.inverse) + std::to_string(r.edges) +
                         std::to_string(r.cellFaceNeighbors) + std::to_string(r.wallDistance) + std::to_string(r.allFaceCellsLocal);
            for (size_t slot = 0; slot < size_t(MeshRelation::Count); ++slot)
                signature += view.HasRelation(static_cast<MeshRelation>(slot)) ? "r1" : "r0";
            for (const auto &key : source->RelationKeys())
            { const auto value = key.Signature(); signature += "a" + std::to_string(value.size()) + ":" + value; }
            std::vector<real> frame;
            for (size_t f = 0; f < view.Periodic().rotation.size(); ++f)
            {
                for (int entry = 0; entry < 9; ++entry) frame.push_back(view.Periodic().rotation[f].map().data()[entry]);
                for (int entry = 0; entry < 3; ++entry) frame.push_back(view.Periodic().translation[f].map()[entry]);
                for (int entry = 0; entry < 3; ++entry) frame.push_back(view.Periodic().rotationCenter[f].map()[entry]);
            }
            signature += nlohmann::json(frame).dump();
            signature += source->Provenance() ? source->Provenance()->Signature() : "none";
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native setup replacement metadata too large");
            prepared = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*source));
            // Prepare all allocations and descriptor bindings before publication.
            prepared->BindPreparedEpoch(versions);
            identity = nodes._identity;
            periodic = view.Periodic();
            nextPeriodic = view.IsPeriodic(); });
        int size = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&size, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { expected = mpi.rank == 0 ? signature : std::string(size_t(size), '\0'); });
        MPI_Bcast(expected.data(), size, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(signature == expected, "native setup source metadata differs across ranks"); });
        MeshMutationToken mutation(this, versions, Runtime::Change::All);
        mutation.Commit();
        // Refresh stamps after commit; these operations only rebind shared
        // handles or clear existing storage and cannot allocate.
        prepared->BindPreparedEpoch(versions);
        _entityIdentity = std::move(identity);
        isPeriodic = nextPeriodic;
        periodicInfo = periodic;
        mpi = nextMPI;
        if (communication)
            _mpiOwner = communication;
        _nativeState = std::move(prepared);
        _nativeSmoothing.reset();
        node2bndNode.clear();
        node2parentNode.clear();
        cell2parentCell.clear();
    }

    void UnstructuredMesh::PlaceNative(Runtime::Device device, Runtime::ExecutionContext context)
    {
        auto read = versions.Read();
        DNDS_check_throw_info(_nativeState, "publish native mesh state before placement");
        _nativeState = _nativeState->WithPlacement(device, context);
        _nativeSmoothing.reset();
    }

}
