#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Metis.hpp"
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
    GhostSpec ExtendNativeMeshGhostSpecForTopology(const NativeOwnedMeshData &owned, bool faces, bool edges, int ghostLayers)
    {
        auto recipe = owned.ghostSpec ? *owned.ghostSpec : DefaultNativeMeshGhostSpec(owned, ghostLayers);
        auto append = [&](EntityKind anchor, std::vector<AdjKind> hops)
        { recipe.chains.push_back({anchor, hops, hops.back().to}); };
        // Requested new topology explicitly extends its caller's coverage.
        // Existing custom rings/named paths are retained unchanged.
        if (faces)
        {
            append(EntityKind::Cell, {Adj::Cell2Face});
            append(EntityKind::Face, {Adj::Face2Node});
            append(EntityKind::Face, {Adj::Face2Cell});
            append(EntityKind::Face, {Adj::Face2Bnd});
            append(EntityKind::Bnd, {Adj::Bnd2Face});
            append(EntityKind::Cell, {Adj::Cell2Face, Adj::Face2Cell});
        }
        if (edges)
        {
            append(EntityKind::Cell, {Adj::Cell2Edge});
            append(EntityKind::Edge, {Adj::Edge2Node});
            append(EntityKind::Edge, {Adj::Edge2Cell});
        }
        return recipe;
    }

    std::shared_ptr<const UnstructuredMeshState> ConstructNativeMesh(NativeMeshInput input, NativeMeshBuildOptions options)
    {
        const auto mpi = input.mpi;
        auto owned = NativeMeshBuilder::Build(std::move(input), options);
        GhostSpec recipe;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { recipe = DefaultNativeMeshGhostSpec(owned, options.cellLayers, options.completeNodeStar); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }

    std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshTopology(
        const UnstructuredMeshState &source, const MPIInfo &mpi,
        bool faces, bool edges, int ghostLayers, const std::map<EntityKind, index> &generatedIdStarts)
    {
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(ghostLayers > 0, "topology finalization requires positive default cell layers"); });
        auto owned = NativeMeshTopology::Build(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, faces, edges, generatedIdStarts);
        GhostSpec recipe;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { recipe = ExtendNativeMeshGhostSpecForTopology(owned, faces, edges, ghostLayers); });
        owned.ghostSpec = std::move(recipe);
        auto result = AssembleNativeMesh(mpi, std::move(owned), ghostLayers);
        if (!result->Requirements().faces)
            return result;
        std::vector<index> partStarts;
        for (int part = 0; part < source.NLocalParts(); ++part)
            partStarts.push_back(source.LocalPartStart(part));
        partStarts.push_back(source.NumOwned(EntityKind::Cell));
        return result->WithLocalPartitions(std::move(partStarts));
    }

    void UnstructuredMesh::PrepareNativeTopology(bool faces, bool edges, int ghostLayers)
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(_nativeState, "native topology preparation requires a native mesh"); });
        const auto stamp = versions.Stamp();
        auto result = NativeMeshPreparation::PrepareTopology(*_nativeState, mpi, faces, edges, ghostLayers);
        std::shared_ptr<UnstructuredMeshState> prepared;
        FieldTransferDetail::LocalStage(mpi, [&]
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

    namespace
    {
        struct ActiveCommunicator
        {
            MPI_Comm value = MPI_COMM_NULL;
            ~ActiveCommunicator()
            {
                if (value != MPI_COMM_NULL)
                    MPI_Comm_free(&value);
            }
        };
        void Agree(const MPIInfo &mpi, const NativePartitionOptions &options)
        {
            std::array<int, 7> values{options.metisUfactor, options.metisSeed, options.edgeWeightMethod,
                                      options.metisNcuts, options.ghostLayers, options.metisType == "KWAY", int(options.deferNewFacesUntilPreparation)};
            auto expected = values;
            MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(values == expected && (options.metisType == "KWAY" || options.metisType == "RB") &&
                                                                        (options.edgeWeightMethod == 0 || options.edgeWeightMethod == 1) && options.metisUfactor > 0 && options.metisSeed >= 0 &&
                                                                        options.metisNcuts > 0 && options.ghostLayers > 0 &&
                                                                        options.metisSeed <= std::numeric_limits<int>::max() - (options.metisNcuts - 1),
                                                                    "native partition requires matching KWAY/RB, edge weight method 0/1, positive cuts/ufactor/layers and supported seed"); });
        }
    }

    std::shared_ptr<const UnstructuredMeshState> NativeMeshPreparation::PrepareTopology(
        const UnstructuredMeshState &source, const MPIInfo &mpi, bool faces, bool edges, int ghostLayers)
    {
        std::array<int, 5> flags{int(faces), int(edges), ghostLayers, 0, 0};
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            flags[3] = source.Requirements().faces;
            flags[4] = source.Requirements().edges; });
        auto expected = flags;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(flags == expected && ghostLayers > 0, "native preparation topology options differ across ranks"); });
        return PrepareNativeMeshTopology(source, mpi, faces, edges, ghostLayers);
    }

    namespace detail
    {
        namespace
        {
            ConnectivityRows::SubEntityRows PrimaryFaces(const UnstructuredMeshView &view, const EntitySpace &nodes, index count)
            {
                std::vector<index> nodeGlobal(size_t(view.NumNodeProc()));
                for (index node = 0; node < view.NumNodeProc(); ++node)
                    nodeGlobal[size_t(node)] = nodes.LocalToGlobal(node);
                for (index cell = 0; cell < count; ++cell)
                {
                    const auto cone = view.CellNodes(cell);
                    const auto element = Elem::Element{view.GetCellInfo(cell).getElemType()};
                    DNDS_check_throw_info(element.GetDim() == view.Dimension() && cone.size() == size_t(element.GetNumNodes()),
                                          "primary face scratch requires matching complete element cones");
                    if (view.IsPeriodic())
                        DNDS_check_throw_info(view.CellNodePeriodicBits(cell).size() == cone.size(),
                                              "primary face scratch periodic cone shape differs");
                    for (auto node : cone)
                        DNDS_check_throw_info(node >= 0 && node < view.NumNodeProc(), "primary face scratch requires local node closure");
                }
                struct CellTypes
                {
                    UnstructuredMeshView view;
                    const ElemInfo &operator()(size_t cell, size_t) const { return view.GetCellInfo(index(cell)); }
                };
                struct Cone
                {
                    UnstructuredMeshView view;
                    const std::vector<index> &nodeGlobal;
                    index ConvertedIndex(index cell, size_t slot, MeshAdjState) const { return nodeGlobal[size_t(view.CellNodes(cell)[slot])]; }
                };
                struct Periodic
                {
                    UnstructuredMeshView view;
                    NodePeriodicBits operator()(index cell, size_t slot) const { return view.CellNodePeriodicBits(cell)[slot]; }
                };
                const Cone cone{view, nodeGlobal};
                const Periodic periodic{view};
                const auto query = Query(CellTypes{view}, cone, false, view.IsPeriodic() ? &periodic : nullptr);
                return ConnectivityRows::EnumerateSubEntities(
                    [&](index cell, int slot)
                    { return view.CellNodes(cell)[size_t(slot)]; }, query, count, view.NumNodeProc());
            }
        }

        tLocalMatStruct BuildPrimaryLocalCellNeighbors(const UnstructuredMeshState &source)
        {
            source.Validate();
            source.CheckRequirements(MeshReadRequirements::Primary());
            DNDS_check_throw_info(!source.Requirements().faces && !source.HasEntities(EntityKind::Face),
                                  "primary local graph requires no existing Face family");
            auto read = source.Snapshot(Runtime::Device::Host()).AcquireRead();
            const auto view = read.HostView();
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3,
                                  "primary local graph requires dimension two or three");
            tLocalMatStruct graph(size_t(view.NumCell()));
            if (!view.NumCell())
                return graph;
            const auto rows = PrimaryFaces(view, source.Entities(EntityKind::Node), view.NumCellProc());
            for (const auto &parents : rows.entity2parent)
            {
                if (std::none_of(parents.begin(), parents.end(), [&](index cell)
                                 { return cell < view.NumCell(); }))
                    continue;
                DNDS_check_throw_info(!parents.empty() && parents.size() <= 2 &&
                                          (view.IsPeriodic() || parents.size() == 1 || parents[0] != parents[1]),
                                      "native face has missing, repeated or nonmanifold parent incidences");
                if (parents.size() == 2 && parents[0] != parents[1] && parents[0] < view.NumCell() && parents[1] < view.NumCell())
                {
                    graph[size_t(parents[0])].push_back(parents[1]);
                    graph[size_t(parents[1])].push_back(parents[0]);
                }
            }
            for (auto &row : graph)
            {
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
            }
            return graph;
        }

        NativePartitionGraph BuildReaderPartitionGraph(const UnstructuredMeshState &source, const MPIInfo &mpi)
        {
            NativePartitionGraph graph;
            std::optional<MeshReadAccess> read;
            std::optional<EntitySpace> cells, nodes;
            int active = 0;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                source.Validate();
                DNDS_check_throw_info(!source.Requirements().faces, "reader partition scratch requires primary input");
                cells = source.Entities(EntityKind::Cell);
                nodes = source.Entities(EntityKind::Node);
                (void)METIS::indexToIdx(cells->NumGlobal());
                active = cells->NumOwned() ? 1 : 0;
                read.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead());
                DNDS_check_throw_info(read->HostView().Dimension() == 2 || read->HostView().Dimension() == 3,
                                      "reader partition scratch requires dimension two or three"); });
            MPI_Allreduce(MPI_IN_PLACE, &active, 1, MPI_INT, MPI_SUM, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(active <= 1, "reader partition scratch requires at most one active cell rank");
                graph.offsets.push_back(0);
                if (!cells->NumOwned()) return;
                const auto view = read->HostView();
                const index firstCell = cells->LocalToGlobal(0);
                const auto rows = PrimaryFaces(view, *nodes, cells->NumOwned());
                std::vector<std::vector<index>> neighbors(size_t(cells->NumOwned()));
                for (const auto &parents : rows.entity2parent)
                {
                    DNDS_check_throw_info(!parents.empty() && parents.size() <= 2 &&
                                              (view.IsPeriodic() || parents.size() == 1 || parents[0] != parents[1]),
                                          "native face has missing, repeated or nonmanifold parent incidences");
                    if (parents.size() == 2 && parents[0] != parents[1])
                    {
                        neighbors[size_t(parents[0])].push_back(firstCell + parents[1]);
                        neighbors[size_t(parents[1])].push_back(firstCell + parents[0]);
                    }
                }
                for (auto &row : neighbors)
                {
                    CompactAdjacencyRow(row);
                    for (auto neighbor : row) graph.adjacency.push_back(METIS::indexToIdx(neighbor));
                    graph.offsets.push_back(METIS::indexToIdx(size_t_to_signed<index>(graph.adjacency.size())));
                } });
            return graph;
        }

        void ScaleNativePartitionWeights(const MPIInfo &mpi, const std::vector<real> &diameters, NativePartitionGraph &graph)
        {
            real maximum = 0;
            index directedEdges = 0;
            std::vector<index> edgeCounts;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                directedEdges = size_t_to_signed<index>(diameters.size());
                edgeCounts.resize(size_t(mpi.size)); });
            MPI_Allgather(&directedEdges, 1, DNDS_MPI_INDEX, edgeCounts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            for (auto diameter : diameters)
                maximum = std::max(maximum, diameter);
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                directedEdges = 0;
                for (auto count : edgeCounts) directedEdges = CheckedSize::Add(directedEdges, count);
                DNDS_check_throw_info(directedEdges < std::numeric_limits<idx_t>::max(), "weighted partition edge count exceeds backend integer capacity");
                const index capacity = directedEdges ? (index(std::numeric_limits<idx_t>::max()) - 1) / directedEdges : 1;
                // INT_MAX also keeps the floating-to-integer conversion
                // exactly representable on both 32- and 64-bit backends.
                const idx_t cap = idx_t(std::min<index>(INT_MAX, capacity));
                graph.weights.reserve(diameters.size());
                for (auto diameter : diameters)
                {
                    DNDS_check_throw_info(std::isfinite(maximum) && maximum > 0, "partition face diameter maximum is invalid");
                    // Bound the sum of every directed edge weight. The old
                    // INT_MAX scale overflowed 32-bit METIS degree/cut sums.
                    const auto weight = real(cap - 1) * (diameter / maximum) + real(1);
                    DNDS_check_throw_info(std::isfinite(weight) && weight >= 1, "partition edge weight is invalid");
                    graph.weights.push_back(idx_t(std::min(weight, real(cap))));
                } });
        }

        NativePartitionGraph BuildNativePartitionGraph(const UnstructuredMeshState &source, const MPIInfo &mpi, int method)
        {
            NativePartitionGraph graph;
            std::vector<real> diameters;
            std::optional<EntitySpace> cells;
            std::optional<MeshReadAccess> read;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                source.Validate();
                source.CheckRequirements(MeshReadRequirements::CellFlux());
                DNDS_check_throw_info(method == 0 || method == 1, "unsupported native partition edge weight method");
                cells = source.Entities(EntityKind::Cell);
                (void)METIS::indexToIdx(cells->NumGlobal());
                read.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead()); });
            int expected = method;
            MPI_Bcast(&expected, 1, MPI_INT, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(method == expected, "native partition edge weight methods differ across ranks");
                const auto view = read->HostView();
                graph.offsets.push_back(0);
                for (index cell = 0; cell < cells->NumOwned(); ++cell)
                {
                    std::vector<NativeNeighborWeight> neighbors;
                    for (auto face : view.CellFaces(cell))
                    {
                        DNDS_check_throw_info(face >= 0 && face < view.NumFaceProc(), "partition facial cone is unavailable");
                        for (auto neighbor : view.FaceCells(face))
                        {
                            if (neighbor == UnInitIndex) continue;
                            DNDS_check_throw_info(neighbor >= 0 && neighbor < view.NumCellProc(), "partition face support is unavailable");
                            const auto target = cells->LocalToGlobal(neighbor);
                            if (target == cells->LocalToGlobal(cell)) continue;
                            real diameter = 0;
                            if (method)
                            {
                                const auto cone = view.FaceNodes(face);
                                for (size_t i = 0; i < cone.size(); ++i)
                                {
                                    DNDS_check_throw_info(cone[i] >= 0 && cone[i] < view.NumNodeProc(), "partition face coordinate is unavailable");
                                    const auto point = view.GetCoordNodeOnFace(face, rowsize(i));
                                    DNDS_check_throw_info(point.allFinite(), "partition face coordinate is not finite");
                                    for (size_t j = 0; j < i; ++j)
                                    {
                                        const auto distance = (point - view.GetCoordNodeOnFace(face, rowsize(j))).norm();
                                        DNDS_check_throw_info(std::isfinite(distance), "partition face diameter is not finite");
                                        diameter = std::max(diameter, distance);
                                    }
                                }
                                DNDS_check_throw_info(diameter > 0, "weighted partition face has no positive diameter");
                            }
                            neighbors.emplace_back(target, diameter);
                        }
                    }
                    CompactAdjacencyRow(neighbors);
                    for (const auto &[neighbor, diameter] : neighbors)
                    {
                        graph.adjacency.push_back(METIS::indexToIdx(neighbor));
                        if (method) diameters.push_back(diameter);
                    }
                    graph.offsets.push_back(METIS::indexToIdx(size_t_to_signed<index>(graph.adjacency.size())));
                } });
            if (method)
                ScaleNativePartitionWeights(mpi, diameters, graph);
            return graph;
        }
    }

    namespace detail
    {
        std::vector<MPI_int> PartitionNativeGraph(const EntitySpace &cells, const MPIInfo &mpi,
                                                  NativePartitionGraph graph, const NativePartitionOptions &options)
        {
            std::vector<idx_t> xadj, adjacency, adjacencyWeights, part;
            std::vector<index> counts(size_t(mpi.size));
            const index owned = cells.NumOwned();
            MPI_Allgather(&owned, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            idx_t global = 0, nParts = 0;
            std::vector<idx_t> offsets;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                global = METIS::indexToIdx(cells.NumGlobal());
                nParts = METIS::indexToIdx(std::min<index>(mpi.size, global));
                offsets.push_back(0);
                for (auto count : counts)
                    if (count) offsets.push_back(offsets.back() + METIS::indexToIdx(count));
                DNDS_check_throw_info(offsets.back() == global, "native partition cell ownership is inconsistent");
                part.resize(std::max<size_t>(size_t(owned), 1)); });
            xadj = std::move(graph.offsets);
            adjacency = std::move(graph.adjacency);
            adjacencyWeights = std::move(graph.weights);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                if (adjacency.empty()) adjacency.push_back(0);
                if (options.edgeWeightMethod && adjacencyWeights.empty()) adjacencyWeights.push_back(1); });
            std::vector<real_t> weights;
            std::vector<idx_t> candidate;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                weights.assign(size_t(nParts), nParts ? real_t(1) / real_t(nParts) : real_t(0));
                candidate.resize(part.size()); });
            ActiveCommunicator active;
            MPI_Comm_split(mpi.comm, owned ? 1 : MPI_UNDEFINED, mpi.rank, &active.value);
            int status = METIS_OK;
            if (owned && nParts > 1)
            {
                int activeSize = 0;
                MPI_Comm_size(active.value, &activeSize);
                idx_t nCon = 1, zero = 0, edgeCut = 0;
                if (activeSize == 1)
                {
                    std::array<idx_t, METIS_NOPTIONS> settings{};
                    METIS_SetDefaultOptions(settings.data());
                    // Preserve MeshPartitionCell2Cell's source-reader policy. The
                    // defaults differ between KWAY and RB (notably refinement and
                    // contiguity); seed/cut/imbalance alone do not define the same
                    // legacy partition.
                    settings[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
                    settings[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
                    settings[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
                    settings[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
                    settings[METIS_OPTION_NCUTS] = options.metisNcuts;
                    settings[METIS_OPTION_NITER] = 10;
                    settings[METIS_OPTION_UFACTOR] = options.metisUfactor;
                    settings[METIS_OPTION_MINCONN] = 1;
                    settings[METIS_OPTION_CONTIG] = 1;
                    settings[METIS_OPTION_SEED] = options.metisSeed;
                    settings[METIS_OPTION_NUMBERING] = 0;
                    settings[METIS_OPTION_DBGLVL] = METIS_DBG_TIME;
                    idx_t vertices = global;
                    auto partition = options.metisType == "RB" ? METIS_PartGraphRecursive : METIS_PartGraphKway;
                    status = partition(&vertices, &nCon, xadj.data(), adjacency.data(), nullptr, nullptr,
                                       options.edgeWeightMethod ? adjacencyWeights.data() : nullptr,
                                       &nParts, nullptr, nullptr, settings.data(), &edgeCut, part.data());
                }
                else
                {
                    std::array<real_t, 1> tolerance{real_t(1) + real_t(options.metisUfactor) / real_t(1000)};
                    idx_t weighted = options.edgeWeightMethod ? 1 : 0;
                    std::array<idx_t, 3> settings{1, 0, idx_t(options.metisSeed)};
                    bool best = false;
                    for (int cut = 0; cut < options.metisNcuts; ++cut)
                    {
                        settings[2] = idx_t(options.metisSeed + cut);
                        idx_t objective = 0;
                        status = ParMETIS_V3_PartKway(offsets.data(), xadj.data(), adjacency.data(), nullptr,
                                                      options.edgeWeightMethod ? adjacencyWeights.data() : nullptr, &weighted, &zero, &nCon, &nParts, weights.data(), tolerance.data(), settings.data(),
                                                      &objective, candidate.data(), &active.value);
                        if (status != METIS_OK)
                            break;
                        if (!best || objective < edgeCut)
                        {
                            best = true;
                            edgeCut = objective;
                            part = candidate;
                        }
                    }
                }
            }
            std::vector<MPI_int> targets;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(status == METIS_OK, "native METIS/ParMETIS partition failed");
                for (index row = 0; row < owned; ++row)
                {
                    const auto target = nParts > 1 ? part[size_t(row)] : 0;
                    DNDS_check_throw_info(target >= 0 && target < mpi.size, "partition returned invalid rank");
                    targets.push_back(MPI_int(target));
                } });
            return targets;
        }
    }

    NativeOwnedMeshTransitionResult NativeMeshPreparation::PartitionOwned(
        const NativeOwnedMeshData &source, const MPIInfo &mpi, const NativePartitionOptions &options)
    {
        Agree(mpi, options);
        int active = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto &[kind, space] : source.spaces)
                DNDS_check_throw_info(space.Valid() && space.NumGhost() == 0, "owned partition requires owned-only input spaces");
            active = source.spaces.at(EntityKind::Cell).NumOwned() ? 1 : 0; });
        MPI_Allreduce(MPI_IN_PLACE, &active, 1, MPI_INT, MPI_SUM, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(options.metisType != "RB" || active <= 1, "owned RB partition requires one reader cell rank"); });
        const bool primaryCellGraph = options.edgeWeightMethod == 0 && source.relations[size_t(MeshRelation::CellCells)];
        auto graph = primaryCellGraph
                         ? detail::BuildOwnedPrimaryCellGraph(source, mpi)
                         : detail::BuildOwnedPartitionGraph(source, mpi, options.edgeWeightMethod);
        auto carried = source;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            std::vector<size_t> sizes(size_t(graph.offsets.empty() ? 0 : graph.offsets.size() - 1));
            for (size_t row = 0; row < sizes.size(); ++row)
                sizes[row] = size_t(graph.offsets[row + 1] - graph.offsets[row]);
            carried.namedRelations.insert_or_assign(
                NativeSetupFacialCellNeighborsKey(),
                Runtime::BuildArrayRows<index>(sizes.size(), [&](size_t row) { return sizes[row]; },
                                               [&](size_t row, auto output)
                                               {
                                                   for (size_t slot = 0; slot < output.size(); ++slot)
                                                       output[slot] = index(graph.adjacency[size_t(graph.offsets[row]) + slot]);
                                               })); });
        if (mpi.size == 1)
        {
            // One rank already owns every row. Keep the reader's established
            // family order and IDs; the graph cache is retained for the local
            // reorder that follows. There is no field or topology migration.
            NativeOwnedMeshTransitionResult result;
            result.sourceSpaces = source.spaces;
            result.owned = std::move(carried);
            return result;
        }
        auto targets = detail::PartitionNativeGraph(source.spaces.at(EntityKind::Cell), mpi, std::move(graph), options);
        ReorderInput input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            input.explicitMaps.push_back({EntityKind::Cell, std::move(targets)});
            if (!source.spaces.count(EntityKind::Face))
            {
                const auto side = source.periodic
                    ? AdjKind::Composition(Adj::Bnd2Node, Adj::Cell2Node.Inverted(), RelationSelection::Exact, false,
                                           "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched)
                    : Adj::Bnd2Cell;
                input.follows = {{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell},
                                 {EntityKind::Bnd, EntityKind::Cell, side}};
            } });
        return NativeMeshTransition::BuildOwned(carried, mpi, input);
    }

    NativeMeshTransitionResult NativeMeshPreparation::Partition(
        const UnstructuredMeshState &source, const MPIInfo &mpi, const NativePartitionOptions &options)
    {
        Agree(mpi, options);
        int activeRanks = 0;
        std::array<int, 2> facial{};
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            activeRanks = source.NumOwned(EntityKind::Cell) ? 1 : 0;
            facial = {int(source.Requirements().faces), int(source.HasEntities(EntityKind::Face))}; });
        MPI_Allreduce(MPI_IN_PLACE, &activeRanks, 1, MPI_INT, MPI_SUM, mpi.comm);
        auto expectedFacial = facial;
        MPI_Bcast(expectedFacial.data(), int(expectedFacial.size()), MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(facial == expectedFacial, "native partition facial inventory differs across ranks");
                                          DNDS_check_throw_info(options.metisType != "RB" || activeRanks <= 1,
                                                                "native recursive partition requires all cells on one reader rank before distribution"); });
        // Preserve the completed API's family/identity behavior, but retain
        // owned data through topology and migration. The exact graph itself
        // needs only temporary facial incidence, never a published mesh.
        const bool readerGraph = !facial[0] && !facial[1] && activeRanks <= 1 && options.edgeWeightMethod == 0;
        auto owned = NativeMeshAssembler::CaptureOwned(source, mpi);
        const bool completeBefore = !facial[0] && !readerGraph;
        if (completeBefore)
            owned = NativeMeshTopology::Build(std::move(owned), mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd, EntityKind::Face, EntityKind::Edge})
                if (source.HasEntities(kind))
                {
                    DNDS_check_throw_info(source.NumOwned(kind) == owned.spaces.at(kind).NumOwned(), "topology preparation changed original owned extent");
                    auto ids = owned.ids.at(kind).Read();
                    for (index row = 0; row < source.NumOwned(kind); ++row)
                        DNDS_check_throw_info(source.OriginalEntityId(kind, row) == ids.View()(size_t(row), 0),
                                              "topology preparation changed original owned row order");
                } });
        const bool primaryCellGraph = !source.Requirements().faces && options.edgeWeightMethod == 0 &&
                                      owned.relations[size_t(MeshRelation::CellCells)];
        auto graph = readerGraph
                         ? detail::BuildReaderPartitionGraph(source, mpi)
                     : source.Requirements().faces
                         ? detail::BuildNativePartitionGraph(source, mpi, options.edgeWeightMethod)
                     : primaryCellGraph
                         ? detail::BuildOwnedPrimaryCellGraph(owned, mpi)
                         : detail::BuildOwnedPartitionGraph(owned, mpi, options.edgeWeightMethod);
        auto targets = detail::PartitionNativeGraph(owned.spaces.at(EntityKind::Cell), mpi, std::move(graph), options);
        ReorderInput input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            input.explicitMaps.push_back({EntityKind::Cell, std::move(targets)});
            if (!source.HasEntities(EntityKind::Face))
            {
                input.follows = {{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell},
                                 {EntityKind::Bnd, EntityKind::Cell, NativeBoundaryPlacementRelation(source)}};
                if (completeBefore) input.follows.push_back({EntityKind::Face, EntityKind::Cell, Adj::Face2Cell});
            } });
        auto moved = NativeMeshTransition::BuildOwned(owned, mpi, input);
        const bool completeAfter = readerGraph && !options.deferNewFacesUntilPreparation;
        if (completeAfter)
            moved.owned = NativeMeshTopology::Build(std::move(moved.owned), mpi);
        if (completeBefore || completeAfter)
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { moved.owned.ghostSpec = ExtendNativeMeshGhostSpecForTopology(moved.owned, true, false, options.ghostLayers); });
        NativeMeshTransitionResult result;
        result.transfers = std::move(moved.transfers);
        result.sourceSpaces = std::move(moved.sourceSpaces);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            // External fields validate the caller's original publication,
            // never the detached owned map or a transient completion map.
            for (const auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd, EntityKind::Face, EntityKind::Edge})
                if (source.HasEntities(kind)) result.sourceSpaces.insert_or_assign(kind, source.Entities(kind)); });
        result.state = AssembleNativeMesh(mpi, std::move(moved.owned), options.ghostLayers);
        return result;
    }
}
