#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/Connectivity/NativeConnectivity.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"

namespace DNDS::Geom
{
    NativeCellClosure BuildNativeCellClosure(const NativeOwnedMeshData &source, const MPIInfo &mpi)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native cell closure requires a communicator");
        std::optional<NativeRelation> initial;
        std::optional<Runtime::Array<ElemInfo>> metadata;
        std::optional<Runtime::Array<NodePeriodicBits>> periodic;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            const auto cells = source.spaces.at(EntityKind::Cell), nodes = source.spaces.at(EntityKind::Node);
            DNDS_check_throw_info(cells.NumGhost() == 0 && nodes.NumGhost() == 0 && source.cellInfo &&
                                      source.relations[size_t(MeshRelation::CellNodes)],
                                  "construction closure requires authoritative owned cell/node inputs");
            auto rows = *source.relations[size_t(MeshRelation::CellNodes)];
            initial.emplace(Adj::Cell2Node, cells, nodes, std::move(rows));
            metadata = source.cellInfo; });
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            if (source.periodic)
            {
                DNDS_check_throw_info(source.cellNodesPbi, "periodic construction closure requires incidence images");
                periodic = source.cellNodesPbi;
            } });
        int periodicFlags[]{int(bool(periodic)), -int(bool(periodic))};
        MPI_Allreduce(MPI_IN_PLACE, periodicFlags, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(periodicFlags[0] == -periodicFlags[1], "native cell closure periodic metadata differs between ranks");
        NativeConnectivity setup(mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { setup.Register(initial->Read()); });
        auto support = NativeConnectivity::Inverse(initial->Read(), mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { setup.Register(support.Read()); });
        GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Node, support.Kind()}, EntityKind::Cell},
                        {EntityKind::Cell, {Adj::Cell2Node, support.Kind(), Adj::Cell2Node}, EntityKind::Node}}};
        auto publication = setup.BuildGhosted(setup.Evaluate(CompiledGhostTree::compile(spec)));
        const auto schema = Runtime::ElementSchema::For<ElemInfo>();
        auto ghosts = publication.routes.at(EntityKind::Cell).TransferRows(publication.spaces.at(EntityKind::Cell), *metadata, schema);
        ghosts = Runtime::Array<ElemInfo>::FromStorage(Runtime::RowLayout::Dense(ghosts.Rows(), 1), ghosts.Storage());
        NativeCellClosure result{std::move(publication.relations.at(Adj::Cell2Node)), {std::move(*metadata), std::move(ghosts)}};
        if (periodic)
        {
            const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
            auto ghostBits = publication.routes.at(EntityKind::Cell).TransferRows(publication.spaces.at(EntityKind::Cell), *periodic, bits);
            result.cellNodesPbi = Runtime::ArrayPair<NodePeriodicBits>{std::move(*periodic), std::move(ghostBits)};
        }
        return result;
    }

    NativeCellClosure BuildNativeCellClosure(const UnstructuredMeshState &source, const MPIInfo &mpi)
    {
        return BuildNativeCellClosure(NativeMeshAssembler::CaptureOwned(source, mpi), mpi);
    }
}
