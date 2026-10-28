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
        if ((source.cellCellsComplete && source.relations[size_t(MeshRelation::CellCells)] &&
             source.relations[size_t(MeshRelation::CellNodes)]) ||
            (source.relations[size_t(MeshRelation::NodeCells)] &&
             source.relations[size_t(MeshRelation::CellNodes)]))
        {
            // The primary builder already retained the complete node inverse.
            // Gather its unique node rows directly under the validated setup
            // scope, transfer each node star once, and derive the cell closure
            // without the generic multi-hop registry. Cell2Cell alone is not
            // sufficient: detached setup inputs may deliberately omit remote
            // parent ghosts while retaining authoritative owned node stars.
            const auto cells = source.spaces.at(EntityKind::Cell);
            const auto nodes = source.spaces.at(EntityKind::Node);
            DNDS_check_throw_info(cells.NumGhost() == 0 && source.cellInfo,
                                  "construction closure requires authoritative owned cell inputs");
            RelationReadAccess cellNodesCheck(Adj::Cell2Node, cells, nodes, Adj_PointToGlobal,
                                              *source.relations[size_t(MeshRelation::CellNodes)]);
            std::vector<index> referencedNodes;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto rows = cellNodesCheck._view.owned;
                referencedNodes.reserve(rows.Rows() * 8);
                for (size_t row = 0; row < rows.Rows(); ++row)
                {
                    const auto count = rows.RowSize(row);
                    const auto *values = count ? rows.Row(row) : nullptr;
                    for (size_t slot = 0; slot < count; ++slot)
                    {
                        const auto global = values[slot];
                        DNDS_check_throw_info(global >= 0 && global < nodes.NumGlobal(),
                                              "cell closure requires complete cell-node rows");
                        referencedNodes.push_back(global);
                    }
                } });
            std::vector<index> ghostCells;
            if (source.cellCellsComplete)
            {
                RelationReadAccess cellCellsCheck(Adj::Cell2Cell, cells, cells, Adj_PointToGlobal,
                                                  *source.relations[size_t(MeshRelation::CellCells)]);
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    const auto first = cells.NumOwned() ? cells.LocalToGlobal(0) : 0;
                    auto rows = cellCellsCheck._view.owned;
                    for (size_t cell = 0; cell < rows.Rows(); ++cell)
                    {
                        const auto count = rows.RowSize(cell);
                        const auto *neighbors = count ? rows.Row(cell) : nullptr;
                        for (size_t slot = 0; slot < count; ++slot)
                        {
                            const auto globalCell = neighbors[slot];
                            if (globalCell != UnInitIndex &&
                                (globalCell < first || globalCell - first >= cells.NumOwned()))
                                ghostCells.push_back(globalCell);
                        }
                    } });
            }
            else
            {
                RelationReadAccess nodeCellsCheck(Adj::Node2Cell, nodes, cells, Adj_PointToGlobal,
                                                  *source.relations[size_t(MeshRelation::NodeCells)]);
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    std::sort(referencedNodes.begin(), referencedNodes.end());
                    referencedNodes.erase(std::unique(referencedNodes.begin(), referencedNodes.end()), referencedNodes.end()); });
                std::vector<Runtime::RowRequest> nodeRequests;
                nodeRequests.reserve(referencedNodes.size());
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    for (size_t row = 0; row < referencedNodes.size(); ++row)
                        nodeRequests.push_back({nodes.Owner(referencedNodes[row]),
                                                nodes.OwnerLocalRow(referencedNodes[row]), row}); });
                auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
                Runtime::RowTransferPlan nodeRoute(context, size_t(nodes.NumOwned()), referencedNodes.size(), nodeRequests);
                const auto &ownedNodeCells = *source.relations[size_t(MeshRelation::NodeCells)];
                Runtime::Array<index> fetchedNodeCells(nodeRoute.DestinationLayout(ownedNodeCells.Layout()));
                Runtime::TransferOptions transferOptions;
                transferOptions.mode = Runtime::TransferMode::Transient;
                transferOptions.context = context;
                Runtime::PayloadPlan(nodeRoute, ownedNodeCells.Layout(), fetchedNodeCells.Layout(),
                                     Runtime::ElementSchema::Arithmetic<index>())
                    .Bind(ownedNodeCells, fetchedNodeCells, Runtime::TransferTransport::Direct, transferOptions)
                    .Begin()
                    .Wait();
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    const auto first = cells.NumOwned() ? cells.LocalToGlobal(0) : 0;
                    auto fetched = fetchedNodeCells.Read();
                    auto rows = cellNodesCheck._view.owned;
                    for (size_t cell = 0; cell < rows.Rows(); ++cell)
                    {
                        const auto count = rows.RowSize(cell);
                        const auto *cellNodes = count ? rows.Row(cell) : nullptr;
                        for (size_t slot = 0; slot < count; ++slot)
                        {
                            const auto globalNode = cellNodes[slot];
                            const auto found = std::lower_bound(referencedNodes.begin(), referencedNodes.end(), globalNode);
                            DNDS_check_throw_info(found != referencedNodes.end() && *found == globalNode,
                                                  "cell closure node request mapping is incomplete");
                            const auto requestRow = size_t(found - referencedNodes.begin());
                            for (auto globalCell : fetched.View()[requestRow])
                                if (globalCell != UnInitIndex &&
                                    (globalCell < first || globalCell - first >= cells.NumOwned()))
                                    ghostCells.push_back(globalCell);
                        }
                    } });
            }
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::sort(ghostCells.begin(), ghostCells.end());
                ghostCells.erase(std::unique(ghostCells.begin(), ghostCells.end()), ghostCells.end());
                for (auto global : ghostCells)
                    DNDS_check_throw_info(cells.Owner(global) != mpi.rank && global >= 0 && global < cells.NumGlobal(),
                                          "cell closure discovered an invalid ghost cell"); });
            auto closedCells = cells.WithGhosts(ghostCells);
            RelationGhostRoute route(closedCells);
            auto ghostCellNodes = route.TransferRows(closedCells,
                                                     *source.relations[size_t(MeshRelation::CellNodes)],
                                                     Runtime::ElementSchema::Arithmetic<index>());
            std::vector<index> remoteNodes;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto ghosts = ghostCellNodes.Read();
                for (size_t row = 0; row < ghosts.View().Rows(); ++row)
                    for (auto global : ghosts.View()[row])
                    {
                        DNDS_check_throw_info(global >= 0 && global < nodes.NumGlobal(),
                                              "closed cell contains an invalid node reference");
                        referencedNodes.push_back(global);
                    }
                std::sort(referencedNodes.begin(), referencedNodes.end());
                referencedNodes.erase(std::unique(referencedNodes.begin(), referencedNodes.end()), referencedNodes.end());
                for (auto global : referencedNodes)
                    if (nodes.Owner(global) != mpi.rank)
                        remoteNodes.push_back(global); });
            auto closedNodes = nodes.WithGhosts(std::move(remoteNodes));
            NativeRelation completed = NativeRelation::WithGhostValues(
                Adj::Cell2Node, closedCells, closedNodes,
                *source.relations[size_t(MeshRelation::CellNodes)], std::move(ghostCellNodes));
            const auto schema = Runtime::ElementSchema::For<ElemInfo>();
            auto metadata = route.TransferRows(closedCells, *source.cellInfo, schema);
            metadata = Runtime::Array<ElemInfo>::FromStorage(Runtime::RowLayout::Dense(metadata.Rows(), 1), metadata.Storage());
            NativeCellClosure result{std::move(completed), {*source.cellInfo, std::move(metadata)}};
            if (source.periodic)
            {
                DNDS_check_throw_info(source.cellNodesPbi, "periodic construction closure requires incidence images");
                const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
                auto ghostBits = route.TransferRows(closedCells, *source.cellNodesPbi, bits);
                result.cellNodesPbi = Runtime::ArrayPair<NodePeriodicBits>{*source.cellNodesPbi, std::move(ghostBits)};
            }
            return result;
        }
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
