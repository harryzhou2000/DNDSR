#include "NativeMeshRefinement.hpp"
#include "NativeMeshAssembler.hpp"
#include "NativeMeshClosure.hpp"
#include "NativeConnectivity.hpp"
#include "NativePeriodicMatch.hpp"
#include "FieldTransfer.hpp"

namespace DNDS::Geom
{
    namespace
    {
        using Array = Runtime::Array<index>;
        template <class T>
        Runtime::Array<T> ElevationRows(const std::vector<std::vector<T>> &rows)
        {
            Runtime::RowBuilder builder(rows.size(), sizeof(T));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                std::copy(rows[row].begin(), rows[row].end(), builder.Row<T>(row).data);
            }
            return Runtime::Array<T>::FromSharedStorage(builder.Commit());
        }
        Array ElevationFetch(const Runtime::RowTransferPlan &route, const Array &source,
                             const MPIInfo &mpi, const std::shared_ptr<Runtime::CommunicationContext> &context)
        {
            auto layout = route.DestinationLayout(source.Layout());
            std::optional<Array> result;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { result.emplace(std::move(layout)); });
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = context;
            Runtime::PayloadPlan(route, source.Layout(), result->Layout(), Runtime::ElementSchema::Arithmetic<index>())
                .Bind(source, *result, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return std::move(*result);
        }
    }

    NativeMeshDerivedResult NativeMeshRefinement::Elevate(const NativeMeshState &source, const MPIInfo &mpi,
                                                          NativeMeshBuildOptions options, bool preserveCellIds)
    {
        int expectedPreserve = preserveCellIds;
        MPI_Bcast(&expectedPreserve, 1, MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(expectedPreserve == int(preserveCellIds), "native elevation Cell ID policy differs across ranks"); });
        auto owned = NativeMeshAssembler::CaptureOwned(source, mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto *metadata : {&*owned.cellInfo, &*owned.bndInfo})
            {
                auto read = metadata->Read();
                for (size_t row = 0; row < metadata->Rows(); ++row)
                {
                    const auto element = Elem::Element{read.View()(row, 0).getElemType()};
                    DNDS_check_throw_info(element.GetOrder() == 1 && element.GetNumElev_O1O2() > 0 &&
                                              element.ObtainElevatedElem().GetOrder() == 2,
                                          "native elevation requires supported first-order cells and boundaries");
                }
            } });
        auto closure = BuildNativeCellClosure(source, mpi);
        auto cone = closure.cellNodes.Read();
        auto metadata = closure.cellInfo.ReadCombined();
        const auto cellInfoView = metadata.View();
        std::optional<RelationPeriodicBits> periodic;
        if (closure.cellNodesPbi)
            periodic.emplace(cone, closure.cellNodesPbi->owned, closure.cellNodesPbi->ghost);
        SubEntityQueryPbi query;
        query.numSubEntities = [cellInfoView](index cell)
        { return Elem::Element{cellInfoView(size_t(cell), 0).getElemType()}.GetNumElev_O1O2(); };
        query.describe = [cellInfoView](index cell, int slot)
        {
            const auto span = Elem::Element{cellInfoView(size_t(cell), 0).getElemType()}.ObtainElevNodeSpan(t_index(slot));
            return SubEntityDesc{span.GetNumVertices(), span.GetNumNodes(), t_index(span.type)};
        };
        query.extractNodes = [cellInfoView](index cell, int slot, const std::function<index(int)> &read, index *output)
        {
            auto element = Elem::Element{cellInfoView(size_t(cell), 0).getElemType()};
            std::vector<index> nodes(size_t(element.GetNumNodes()));
            for (size_t i = 0; i < nodes.size(); ++i)
                nodes[i] = read(int(i));
            const auto span = element.ObtainElevNodeSpan(t_index(slot));
            std::vector<index> values(size_t(span.GetNumNodes()));
            element.ExtractElevNodeSpanNodes(t_index(slot), nodes, values);
            std::copy(values.begin(), values.end(), output);
        };
        query.extractPbi = [cellInfoView](index cell, int slot, const std::function<NodePeriodicBits(int)> &read, NodePeriodicBits *output)
        {
            auto element = Elem::Element{cellInfoView(size_t(cell), 0).getElemType()};
            const auto span = element.ObtainElevNodeSpan(t_index(slot));
            std::vector<NodePeriodicBits> nodes(size_t(element.GetNumNodes())), values(size_t(span.GetNumNodes()));
            for (size_t i = 0; i < nodes.size(); ++i)
                nodes[i] = read(int(i));
            element.ExtractElevNodeSpanNodes(t_index(slot), nodes, values);
            std::copy(values.begin(), values.end(), output);
        };
        // Temporary construction spans reuse subentity deduplication. This
        // scratch Edge space includes edge/face/cell spans and is never exposed
        // as the new mesh's edge family; every span becomes one new node.
        auto spans = NativeConnectivity::Interpolate(cone, EntityKind::Edge, query, [mpi](const std::vector<index> &, const std::vector<MPI_int> &ranks, index)
                                                     { return OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == mpi.rank, {}}; }, periodic ? &*periodic : nullptr);
        const auto oldNodes = source.Entities(EntityKind::Node);
        const index nOld = oldNodes.NumOwned();
        index nNew = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(spans.entities.NumOwned() <= std::numeric_limits<index>::max() - nOld,
                                  "native elevated local node count overflows");
            nNew = nOld + spans.entities.NumOwned(); });
        std::vector<index> oldCounts(size_t(mpi.size)), newCounts(size_t(mpi.size)), starts(size_t(mpi.size) + 1);
        MPI_Allgather(&nOld, 1, DNDS_MPI_INDEX, oldCounts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
        MPI_Allgather(&nNew, 1, DNDS_MPI_INDEX, newCounts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                DNDS_check_throw_info(newCounts[size_t(rank)] >= 0 && newCounts[size_t(rank)] <= INT64_MAX - starts[size_t(rank)],
                                      "native elevated node count overflows");
                starts[size_t(rank) + 1] = starts[size_t(rank)] + newCounts[size_t(rank)];
            } });
        auto oldNode = [&](index global)
        { const auto owner = oldNodes.Owner(global); return starts[size_t(owner)] + oldNodes.OwnerLocalRow(global); };
        auto newNode = [&](index span)
        { const auto owner = spans.entities.Owner(span); return starts[size_t(owner)] + oldCounts[size_t(owner)] + spans.entities.OwnerLocalRow(span); };
        const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, nNew);
        const auto cells = nodes.CreateSibling(EntityKind::Cell, source.NumOwned(EntityKind::Cell));
        const auto boundaries = nodes.CreateSibling(EntityKind::Bnd, source.NumOwned(EntityKind::Bnd));
        const auto expandedNodes = closure.cellNodes.Target();
        RelationGhostRoute nodeRoute(expandedNodes);
        auto ghostCoords = nodeRoute.TransferRows(expandedNodes, *owned.coords, Runtime::ElementSchema::Arithmetic<real>());
        auto ghostIds = nodeRoute.TransferRows(expandedNodes, owned.ids.at(EntityKind::Node), Runtime::ElementSchema::Arithmetic<index>());
        Runtime::ArrayPair<real> coordinates{*owned.coords, std::move(ghostCoords)};
        Runtime::ArrayPair<index> sourceIds{owned.ids.at(EntityKind::Node), std::move(ghostIds)};
        auto coordRead = coordinates.ReadCombined();
        auto idRead = sourceIds.ReadCombined();
        Runtime::Array<real> newCoordinates(Runtime::RowLayout::Dense(size_t(nNew), 3));
        std::vector<std::vector<index>> nodeParents(size_t(nNew), std::vector<index>{});
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto output = newCoordinates.Write();
            for (index node = 0; node < nOld; ++node)
            {
                for (size_t component = 0; component < 3; ++component) output.View()(size_t(node), component) = coordRead.View()(size_t(node), component);
                nodeParents[size_t(node)] = {idRead.View()(size_t(node), 0)};
            }
            auto spanNodes = spans.entity2node.Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> spanBits;
            if (spans.entity2nodePbi) spanBits.emplace(spans.entity2nodePbi->Read());
            for (index span = 0; span < spans.entities.NumOwned(); ++span)
            {
                const auto globals = spanNodes.GlobalRow(span);
                tPoint coordinate = tPoint::Zero();
                uint8_t common = 7;
                for (size_t slot = 0; slot < globals.size(); ++slot)
                {
                    const auto global = globals[slot];
                    const auto local = expandedNodes.GlobalToLocal(global);
                    DNDS_check_throw_info(local, "native elevation span node closure missing");
                    tPoint point;
                    for (size_t component = 0; component < 3; ++component) point[component] = coordRead.View()(size_t(*local), component);
                    const auto bits = spanBits ? spanBits->View()(size_t(span), slot) : NodePeriodicBits{0};
                    common &= uint8_t(bits);
                    coordinate += owned.periodic ? owned.periodicInfo.GetCoordByBits(point, bits) : point;
                    nodeParents[size_t(nOld + span)].push_back(idRead.View()(size_t(*local), 0));
                }
                coordinate /= real(globals.size());
                if (owned.periodic) coordinate = owned.periodicInfo.GetCoordBackByBits(coordinate, NodePeriodicBits{common});
                DNDS_check_throw_info(coordinate.allFinite(), "native elevated coordinate is not finite");
                for (size_t component = 0; component < 3; ++component) output.View()(size_t(nOld + span), component) = coordinate[component];
            } });
        std::vector<std::vector<index>> cellNodes(size_t(cells.NumOwned())), boundaryNodes(size_t(boundaries.NumOwned()));
        std::vector<std::vector<NodePeriodicBits>> cellBits(cellNodes.size()), boundaryBits(boundaryNodes.size());
        Runtime::Array<ElemInfo> newCellInfo(Runtime::RowLayout::Dense(size_t(cells.NumOwned()), 1));
        Runtime::Array<ElemInfo> newBoundaryInfo(Runtime::RowLayout::Dense(size_t(boundaries.NumOwned()), 1));
        Array types(Runtime::RowLayout::Dense(size_t(cells.NumOwned()), 1));
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto additions = spans.parent2entity.Read();
            auto output = newCellInfo.Write();
            auto typeWrite = types.Write();
            for (index cell = 0; cell < cells.NumOwned(); ++cell)
            {
                auto &row = cellNodes[size_t(cell)];
                for (auto global : cone.GlobalRow(cell)) row.push_back(oldNode(global));
                for (auto global : additions.GlobalRow(cell)) row.push_back(newNode(global));
                if (periodic)
                {
                    auto &bits = cellBits[size_t(cell)];
                    for (size_t slot = 0; slot < cone.RowSize(cell); ++slot) bits.push_back((*periodic)(cell, slot));
                    auto element = Elem::Element{cellInfoView(size_t(cell), 0).getElemType()};
                    for (t_index span = 0; span < element.GetNumElev_O1O2(); ++span)
                    {
                        const auto sub = element.ObtainElevNodeSpan(span);
                        std::vector<NodePeriodicBits> labels(size_t(sub.GetNumNodes()));
                        element.ExtractElevNodeSpanNodes(span, bits, labels);
                        uint8_t common = 7;
                        for (auto value : labels) common &= uint8_t(value);
                        bits.push_back(NodePeriodicBits{common});
                    }
                }
                auto info = cellInfoView(size_t(cell), 0);
                const auto element = Elem::Element{info.getElemType()};
                typeWrite.View()(size_t(cell), 0) = info.type;
                info.setElemType(element.ObtainElevatedElem().type);
                DNDS_check_throw_info(row.size() == size_t(element.ObtainElevatedElem().GetNumNodes()), "native elevated cell slot count mismatch");
                output.View()(size_t(cell), 0) = info;
            } });
        std::vector<Runtime::RowRequest> requests;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto parents = owned.relations[size_t(MeshRelation::BoundaryCells)]->Read();
            const auto sourceCells = source.Entities(EntityKind::Cell);
            for (index boundary = 0; boundary < boundaries.NumOwned(); ++boundary)
            {
                DNDS_check_throw_info(parents.View().RowSize(size_t(boundary)) == 2,
                                      "native elevation requires two boundary physical slots");
                const auto parent = parents.View()(size_t(boundary), 0);
                requests.push_back({sourceCells.Owner(parent), sourceCells.OwnerLocalRow(parent), size_t(boundary)});
            } });
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        Runtime::RowTransferPlan boundaryRoute(context, size_t(cells.NumOwned()), size_t(boundaries.NumOwned()), requests);
        auto candidateNodes = ElevationFetch(boundaryRoute, closure.cellNodes.Owned(), mpi, context);
        auto candidateSpans = ElevationFetch(boundaryRoute, spans.parent2entity.Owned(), mpi, context);
        auto candidateTypes = ElevationFetch(boundaryRoute, types, mpi, context);
        std::optional<Array> candidateFrames;
        if (periodic)
        {
            std::optional<Array> frames;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto read = owned.cellNodesPbi->Read();
                std::vector<std::vector<index>> rows(owned.cellNodesPbi->Rows());
                for (size_t row = 0; row < rows.size(); ++row) for (auto bits : read.View()[row]) rows[row].push_back(uint8_t(bits));
                frames = ElevationRows(rows); });
            candidateFrames = ElevationFetch(boundaryRoute, *frames, mpi, context);
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto cones = owned.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
            auto boundaryInfo = owned.bndInfo->Read();
            auto parentNodes = candidateNodes.Read(), parentSpans = candidateSpans.Read(), parentTypes = candidateTypes.Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> boundaryImages;
            std::optional<Array::ReadLease> parentImages;
            if (periodic)
            {
                boundaryImages.emplace(owned.bndNodesPbi->Read());
                parentImages.emplace(candidateFrames->Read());
            }
            auto output = newBoundaryInfo.Write();
            for (index boundary = 0; boundary < boundaries.NumOwned(); ++boundary)
            {
                const auto row = size_t(boundary);
                auto info = boundaryInfo.View()(row, 0);
                auto element = Elem::Element{info.getElemType()};
                const auto original = cones.View()[row];
                std::vector<index> globals(original.begin(), original.end());
                for (auto global : globals) boundaryNodes[row].push_back(oldNode(global));
                std::vector<index> originalBits(globals.size(), 0);
                if (periodic)
                    for (size_t slot = 0; slot < globals.size(); ++slot)
                    {
                        const auto value = boundaryImages->View()(row, slot);
                        originalBits[slot] = uint8_t(value);
                        boundaryBits[row].push_back(value);
                    }
                const auto parent = parentNodes.View()[row];
                std::vector<index> parentGlobals(parent.begin(), parent.end());
                std::vector<index> parentBits(parentGlobals.size(), 0);
                if (periodic) for (size_t slot = 0; slot < parentBits.size(); ++slot) parentBits[slot] = parentImages->View()(row, slot);
                auto parentElement = Elem::Element{static_cast<Elem::ElemType>(parentTypes.View()(row, 0))};
                for (t_index slot = 0; slot < element.GetNumElev_O1O2(); ++slot)
                {
                    const auto span = element.ObtainElevNodeSpan(slot);
                    std::vector<index> expected(size_t(span.GetNumNodes()));
                    element.ExtractElevNodeSpanNodes(slot, globals, expected);
                    std::vector<index> expectedBits(expected.size());
                    element.ExtractElevNodeSpanNodes(slot, originalBits, expectedBits);
                    NativePeriodicNodes expectedPattern;
                    uint8_t common = 7;
                    for (size_t node = 0; node < expected.size(); ++node)
                    {
                        expectedPattern.emplace_back(expected[node], uint8_t(expectedBits[node]));
                        common &= uint8_t(expectedBits[node]);
                    }
                    std::sort(expectedPattern.begin(), expectedPattern.end());
                    index found = UnInitIndex;
                    for (t_index parentSlot = 0; parentSlot < parentElement.GetNumElev_O1O2(); ++parentSlot)
                    {
                        const auto candidate = parentElement.ObtainElevNodeSpan(parentSlot);
                        if (candidate.type != span.type) continue;
                        std::vector<index> actual(size_t(candidate.GetNumNodes()));
                        parentElement.ExtractElevNodeSpanNodes(parentSlot, parentGlobals, actual);
                        std::vector<index> actualBits(actual.size());
                        parentElement.ExtractElevNodeSpanNodes(parentSlot, parentBits, actualBits);
                        NativePeriodicNodes actualPattern;
                        for (size_t node = 0; node < actual.size(); ++node) actualPattern.emplace_back(actual[node], uint8_t(actualBits[node]));
                        std::sort(actualPattern.begin(), actualPattern.end());
                        if (actualPattern == expectedPattern)
                        {
                            DNDS_check_throw_info(found == UnInitIndex, "native boundary elevation span is ambiguous");
                            found = parentSpans.View()(row, size_t(parentSlot));
                        }
                    }
                    DNDS_check_throw_info(found != UnInitIndex, "native boundary elevation span is absent from its parent");
                    boundaryNodes[row].push_back(newNode(found));
                    if (periodic) boundaryBits[row].push_back(NodePeriodicBits{common});
                }
                info.setElemType(element.ObtainElevatedElem().type);
                output.View()(row, 0) = info;
            } });
        NativeMeshDerivedResult result;
        std::optional<NativeMeshInput> input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            result.provenance.sourceNamespace = source.MeshNamespace();
            result.provenance.sourceIds.emplace(EntityKind::Node, ElevationRows(nodeParents));
            result.provenance.sourceIds.emplace(EntityKind::Cell, owned.ids.at(EntityKind::Cell));
            result.provenance.sourceIds.emplace(EntityKind::Bnd, owned.ids.at(EntityKind::Bnd));
            result.provenance.sourceKinds = {{EntityKind::Node, EntityKind::Node}, {EntityKind::Cell, EntityKind::Cell}, {EntityKind::Bnd, EntityKind::Bnd}};
            input.emplace(NativeMeshInput{mpi, owned.dimension, cells, nodes, boundaries, ElevationRows(cellNodes), ElevationRows(boundaryNodes),
                                         std::move(newCoordinates), std::move(newCellInfo), std::move(newBoundaryInfo)});
            if (preserveCellIds)
                input->ids.emplace(EntityKind::Cell, owned.ids.at(EntityKind::Cell));
            input->provenance = result.provenance; });
        if (owned.periodic)
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                input->periodic = true;
                input->periodicInfo = owned.periodicInfo;
                input->cellNodesPbi = ElevationRows(cellBits);
                input->boundaryNodesPbi = ElevationRows(boundaryBits); });
        result.state = NativeMeshBuilder::Build(std::move(*input), options);
        return result;
    }
}
