#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"

namespace DNDS::Geom
{
    NativeCellClosure BuildNativeCellClosure(const NativeOwnedMeshData &source, const MPIInfo &mpi)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native cell closure requires a communicator");
        std::optional<MeshAdjacency> initial;
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
        // All choices remain registered adjacency/tree operations. Complete
        // CellCells is a precomputed node-star composition, not a solver ring.
        const bool useCellCells = source.cellCellsComplete &&
                                  bool(source.relations[size_t(MeshRelation::CellCells)]);
        const bool useNodeCells = !useCellCells &&
                                  bool(source.relations[size_t(MeshRelation::NodeCells)]);
        int flags[]{int(bool(periodic)), -int(bool(periodic)),
                    int(useCellCells), -int(useCellCells), int(useNodeCells), -int(useNodeCells)};
        MPI_Allreduce(MPI_IN_PLACE, flags, 6, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[1] && flags[2] == -flags[3] && flags[4] == -flags[5],
                              "construction closure inputs differ between ranks");
        AdjacencyRegistry setup(mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { setup.Register(initial->Read()); });
        std::optional<MeshAdjacency> support;
        GhostSpec spec;
        if (useCellCells)
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                const auto cells = source.spaces.at(EntityKind::Cell);
                support.emplace(Adj::Cell2Cell, cells, cells,
                                *source.relations[size_t(MeshRelation::CellCells)]);
                setup.Register(support->Read()); });
            // Owned cones are an explicit branch: an isolated owned cell need
            // not occur in its own neighbor list, but its remote nodes matter.
            spec = GhostSpec{{{EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
                              {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
                              {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node}}};
        }
        else
        {
            if (useNodeCells)
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { support.emplace(Adj::Node2Cell, source.spaces.at(EntityKind::Node),
                                                                  source.spaces.at(EntityKind::Cell),
                                                                  *source.relations[size_t(MeshRelation::NodeCells)]); });
            else
                support.emplace(AdjacencyRegistry::Inverse(initial->Read(), mpi));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { setup.Register(support->Read()); });
            spec = GhostSpec{{{EntityKind::Cell, {Adj::Cell2Node, support->Kind()}, EntityKind::Cell},
                              {EntityKind::Cell, {Adj::Cell2Node, support->Kind(), Adj::Cell2Node}, EntityKind::Node}}};
        }
        const auto tree = CompiledGhostTree::compile(spec);
        const auto discovery = setup.Evaluate(tree);
        auto publication = setup.BuildGhosted(discovery, {Adj::Cell2Node});
        const auto schema = Runtime::ElementSchema::For<ElemInfo>();
        auto ghosts = publication.routes.at(EntityKind::Cell).TransferRows(publication.spaces.at(EntityKind::Cell), *metadata, schema);
        ghosts = Runtime::Array<ElemInfo>::FromStorage(Runtime::RowLayout::Dense(ghosts.RowCount(), 1), ghosts.Storage());
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
