#include "NativeMeshBuilder.hpp"
#include "NativePeriodicMatch.hpp"
#include "NativeConnectivity.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        /// Agree on recoverable local errors before the next MPI stage. This
        /// includes local relation construction, which validates retained rows.
        template <class F>
        auto LocalStage(const MPIInfo &mpi, const char *stage, F &&action)
        {
            using Result = decltype(action());
            std::optional<Result> result;
            std::string detail;
            try
            {
                result.emplace(action());
            }
            catch (const std::exception &error)
            {
                detail = error.what();
            }
            catch (...)
            {
                detail = "unknown local error";
            }
            int valid = bool(result);
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(valid, std::string(stage) + ": native mesh setup failed on a rank" +
                                             (detail.empty() ? std::string{} : "; " + detail));
            return std::move(*result);
        }

        template <class T>
        void CheckRows(const Runtime::Array<T> &values, index count, std::optional<size_t> width = {})
        {
            DNDS_check_throw_info(values.GetDevice() == Runtime::Device::Host() && values.Rows() == size_t(count),
                                  "native input host placement or owned row count mismatch");
            auto read = values.Read();
            if (width)
                for (size_t row = 0; row < read.View().Rows(); ++row)
                    DNDS_check_throw_info(read.View().RowSize(row) == *width, "native input logical row width mismatch");
        }

        /// Uniform semantic components use a table view at publication. A
        /// metadata-only reinterpretation preserves the original value buffer
        /// and access controller, including aliases held by Python callers.
        template <class T>
        Runtime::Array<T> UniformTable(Runtime::Array<T> values, size_t width)
        {
            const auto layout = values.Layout();
            if (!layout.IsVariable())
            {
                DNDS_check_throw_info(layout.Width() == width, "native uniform input table width mismatch");
                return values;
            }
            for (size_t row = 0; row < layout.Rows(); ++row)
            {
                DNDS_check_throw_info(layout.RowSize(row) == width, "native uniform input logical width mismatch");
                if (layout.IsRagged())
                    DNDS_check_throw_info(layout.Offset(row) == CheckedSize::Multiply(row, width),
                                          "native uniform CSR input is not tightly contiguous");
            }
            if (layout.IsRagged())
                DNDS_check_throw_info(layout.Elements() == CheckedSize::Multiply(layout.Rows(), width),
                                      "native uniform CSR input has an invalid terminal offset");
            const auto stride = layout.IsPadded() && layout.Rows() ? layout.Stride() : width;
            return Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(layout.Rows(), width, stride), values.Storage());
        }

        void CheckCone(const Runtime::Array<index> &cone, const Runtime::Array<ElemInfo> &info,
                       const EntitySpace &source, const EntitySpace &nodes, int dimension, bool boundary,
                       const Runtime::Array<NodePeriodicBits> *periodic = nullptr)
        {
            CheckRows(cone, source.NumOwned());
            CheckRows(info, source.NumOwned(), 1);
            auto entries = cone.Read();
            auto elements = info.Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> bits;
            if (periodic)
            {
                CheckRows(*periodic, source.NumOwned());
                bits.emplace(periodic->Read());
            }
            for (size_t row = 0; row < entries.View().Rows(); ++row)
            {
                const auto elementInfo = elements.View()(row, 0);
                DNDS_check_throw_info(elementInfo.type > Elem::UnknownElem && elementInfo.type < Elem::ElemType_NUM,
                                      "native input contains an unsupported element type");
                const Elem::Element element{elementInfo.getElemType()};
                DNDS_check_throw_info(element.GetDim() == dimension &&
                                          entries.View().RowSize(row) == size_t(element.GetNumNodes()),
                                      "native input cone differs from its element type/dimension");
                DNDS_check_throw_info(!boundary || FaceIDIsExternalBC(elementInfo.zone) || (periodic && FaceIDIsPeriodic(elementInfo.zone)),
                                      "native boundary requires an external or periodic zone");
                DNDS_check_throw_info(!bits || bits->View().RowSize(row) == entries.View().RowSize(row), "periodic cone slots differ from node slots");
                std::set<std::pair<index, uint8_t>> unique;
                for (size_t slot = 0; slot < entries.View().RowSize(row); ++slot)
                {
                    const auto node = entries.View()(row, slot);
                    const auto frame = bits ? uint8_t(bits->View()(row, slot)) : uint8_t(0);
                    DNDS_check_throw_info(node >= 0 && node < nodes.NumGlobal() && frame < 8 && unique.emplace(node, frame).second,
                                          "native input node/image is invalid or repeated in an ordered cone");
                }
            }
        }

        Runtime::Array<index> IndexRows(const std::vector<std::vector<index>> &rows, std::optional<size_t> width = {})
        {
            Runtime::RowBuilder builder(rows.size(), sizeof(index), width ? Runtime::RowLayoutKind::Table : Runtime::RowLayoutKind::CSR, width.value_or(0));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                std::copy(rows[row].begin(), rows[row].end(), builder.Row<index>(row).data);
            }
            return Runtime::Array<index>::FromSharedStorage(builder.Commit());
        }

        std::pair<NativeRelation, NativeRelation> PeriodicBoundaryRelations(
            const NativeMeshInput &input, const RelationReadAccess &candidates, const AdjKind &exactKind)
        {
            const auto mpi = input.mpi;
            std::vector<Runtime::RowRequest> requests;
            std::vector<index> candidateCells;
            std::vector<size_t> starts;
            std::optional<Runtime::Array<index>> types, bits;
            LocalStage(mpi, "periodic boundary candidate preparation", [&]
                       {
                starts.push_back(0);
                for (index boundary = 0; boundary < candidates.NumOwned(); ++boundary)
                {
                    for (auto cell : candidates.GlobalRow(boundary))
                    {
                        requests.push_back({input.cells.Owner(cell), input.cells.OwnerLocalRow(cell), candidateCells.size()});
                        candidateCells.push_back(cell);
                    }
                    starts.push_back(candidateCells.size());
                }
                std::vector<std::vector<index>> cellTypes(size_t(input.cells.NumOwned())), cellBits(cellTypes.size());
                auto info = input.cellInfo.Read();
                auto pbi = input.cellNodesPbi->Read();
                for (size_t row = 0; row < cellTypes.size(); ++row)
                {
                    cellTypes[row] = {info.View()(row, 0).type};
                    for (auto value : pbi.View()[row]) cellBits[row].push_back(uint8_t(value));
                }
                types = IndexRows(cellTypes, 1);
                bits = IndexRows(cellBits);
                return true; });
            auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            Runtime::RowTransferPlan route(context, size_t(input.cells.NumOwned()), candidateCells.size(), requests);
            auto fetch = [&](const Runtime::Array<index> &values)
            {
                auto layout = route.DestinationLayout(values.Layout());
                auto output = LocalStage(mpi, "periodic boundary payload allocation", [&]
                                         { return Runtime::Array<index>(layout); });
                Runtime::TransferOptions options;
                options.mode = Runtime::TransferMode::Transient;
                options.context = context;
                Runtime::PayloadPlan(route, values.Layout(), output.Layout(), Runtime::ElementSchema::Arithmetic<index>())
                    .Bind(values, output, Runtime::TransferTransport::Direct, options)
                    .Begin()
                    .Wait();
                return output;
            };
            auto cones = fetch(input.cellNodes), metadata = fetch(*types), frames = fetch(*bits);
            return LocalStage(mpi, "periodic physical boundary occurrences", [&]
                              {
                auto nodes = cones.Read(), info = metadata.Read(), pbi = frames.Read();
                auto boundaryNodes = input.boundaryNodes.Read();
                auto boundaryFrames = input.boundaryNodesPbi->Read();
                auto boundaryInfo = input.boundaryInfo.Read();
                std::vector<std::vector<index>> exact(size_t(input.boundaries.NumOwned())), physical(exact.size());
                for (size_t boundary = 0; boundary < exact.size(); ++boundary)
                {
                    NativePeriodicNodes expected;
                    for (size_t slot = 0; slot < boundaryNodes.View().RowSize(boundary); ++slot)
                        expected.emplace_back(boundaryNodes.View()(boundary, slot), uint8_t(boundaryFrames.View()(boundary, slot)));
                    struct Occurrence { index cell; t_index slot; uint8_t frame; };
                    std::vector<Occurrence> matches;
                    for (size_t candidate = starts[boundary]; candidate < starts[boundary + 1]; ++candidate)
                    {
                        Elem::Element element{static_cast<Elem::ElemType>(info.View()(candidate, 0))};
                        const auto values = nodes.View()[candidate], labels = pbi.View()[candidate];
                        std::vector<index> parent(values.begin(), values.end()), parentFrames(labels.begin(), labels.end());
                        for (t_index face = 0; face < element.GetNumFaces(); ++face)
                        {
                            const auto sub = element.ObtainFace(face);
                            if (sub.type != boundaryInfo.View()(boundary, 0).type) continue;
                            std::vector<index> subNodes(size_t(sub.GetNumNodes())), subFrames(subNodes.size());
                            element.ExtractFaceNodes(face, parent, subNodes);
                            element.ExtractFaceNodes(face, parentFrames, subFrames);
                            NativePeriodicNodes actual;
                            for (size_t slot = 0; slot < subNodes.size(); ++slot) actual.emplace_back(subNodes[slot], uint8_t(subFrames[slot]));
                            auto frame = NativePeriodicFrame(std::move(actual), expected);
                            if (frame) matches.push_back({candidateCells[candidate], face, *frame});
                        }
                    }
                    auto side = std::find_if(matches.begin(), matches.end(), [](const auto &entry) { return entry.frame == 0; });
                    DNDS_check_throw_info(side != matches.end() && std::count_if(matches.begin(), matches.end(), [](const auto &entry) { return entry.frame == 0; }) == 1,
                                          "boundary must match exactly one full element face in its physical frame");
                    exact[boundary] = {side->cell};
                    physical[boundary] = {side->cell, UnInitIndex};
                    const auto zone = boundaryInfo.View()(boundary, 0).zone;
                    if (FaceIDIsPeriodic(zone))
                    {
                        const uint8_t mask = uint8_t(1U << ((-int(zone) - 1) % 3));
                        DNDS_check_throw_info(matches.size() == 2, "periodic boundary must have exactly two physical face occurrences");
                        const auto &other = matches[side == matches.begin() ? 1 : 0];
                        DNDS_check_throw_info(other.frame == mask, "periodic partner frame differs from its boundary group");
                        physical[boundary][1] = other.cell;
                    }
                }
                return std::make_pair(NativeRelation(exactKind, input.boundaries, input.cells, IndexRows(exact)),
                                      NativeRelation(Adj::Bnd2Cell, input.boundaries, input.cells, IndexRows(physical, 2))); });
        }

        void ValidateBoundaryFaces(const NativeMeshInput &input, const RelationReadAccess &containment)
        {
            if (!input.boundaries.NumGlobal())
                return;
            const auto mpi = input.mpi;
            std::vector<Runtime::RowRequest> requests;
            Runtime::Array<index> types(Runtime::RowLayout::Dense(size_t(input.cells.NumOwned()), 1));
            LocalStage(mpi, "boundary face request preparation", [&]
                       {
                auto metadata = input.cellInfo.Read();
                auto write = types.Write();
                for (index cell = 0; cell < input.cells.NumOwned(); ++cell)
                    write.View()(size_t(cell), 0) = metadata.View()(size_t(cell), 0).type;
                for (index boundary = 0; boundary < containment.NumOwned(); ++boundary)
                {
                    DNDS_check_throw_info(containment.RowSize(boundary) == 1,
                                          "external boundary must have exactly one all-node containing cell");
                    const auto cell = containment.ReadEntry(boundary, 0).global;
                    requests.push_back({input.cells.Owner(cell), input.cells.OwnerLocalRow(cell), size_t(boundary)});
                }
                return true; });
            // Containment may name a remote cell that is not in the initial
            // cell closure. Fetch just those candidate rows in global node IDs;
            // no complete remote mesh or additional entity map is needed.
            auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            Runtime::RowTransferPlan route(context, size_t(input.cells.NumOwned()), size_t(input.boundaries.NumOwned()),
                                           requests, {input.cells.Stamp(), input.boundaries.Stamp()});
            auto fetch = [&](const Runtime::Array<index> &source)
            {
                Runtime::Array<index> received(route.DestinationLayout(source.Layout()));
                Runtime::TransferOptions options;
                options.mode = Runtime::TransferMode::Transient;
                options.context = context;
                Runtime::PayloadPlan(route, source.Layout(), received.Layout(), Runtime::ElementSchema::Arithmetic<index>())
                    .Bind(source, received, Runtime::TransferTransport::Direct, options)
                    .Begin()
                    .Wait();
                return received;
            };
            auto candidateNodes = fetch(input.cellNodes);
            auto candidateTypes = fetch(types);
            LocalStage(mpi, "exact boundary face validation", [&]
                       {
                auto nodes = candidateNodes.Read(), elementTypes = candidateTypes.Read();
                auto boundaryNodes = input.boundaryNodes.Read();
                auto boundaryInfo = input.boundaryInfo.Read();
                for (size_t boundary = 0; boundary < size_t(input.boundaries.NumOwned()); ++boundary)
                {
                    Elem::Element cell{static_cast<Elem::ElemType>(elementTypes.View()(boundary, 0))};
                    const auto source = nodes.View()[boundary];
                    std::vector<index> parent(source.begin(), source.end());
                    const auto target = boundaryNodes.View()[boundary];
                    std::vector<index> expected(target.begin(), target.end());
                    std::sort(expected.begin(), expected.end());
                    bool matched = false;
                    for (t_index face = 0; face < cell.GetNumFaces(); ++face)
                    {
                        const auto element = cell.ObtainFace(face);
                        if (element.type != boundaryInfo.View()(boundary, 0).type)
                            continue;
                        std::vector<index> actual(size_t(element.GetNumNodes()));
                        cell.ExtractFaceNodes(face, parent, actual);
                        std::sort(actual.begin(), actual.end());
                        matched = matched || actual == expected;
                    }
                    DNDS_check_throw_info(matched, "external boundary nodes do not form a face of their containing cell");
                }
                return true; });
        }

        std::vector<index> ReferencedGhostNodes(const NativeMeshInput &input)
        {
            std::set<index> ghosts;
            for (const auto *array : {&input.cellNodes, &input.boundaryNodes})
            {
                auto read = array->Read();
                for (size_t row = 0; row < read.View().Rows(); ++row)
                    for (auto global : read.View()[row])
                        if (input.nodes.Owner(global) != input.mpi.rank)
                            ghosts.insert(global);
            }
            return {ghosts.begin(), ghosts.end()};
        }

        GhostSpec PrimaryClosure(const AdjKind &cellNeighbors, const AdjKind &nodeCells,
                                 const AdjKind &nodeBoundaries, const AdjKind &boundaryCells,
                                 NativeMeshBuildOptions options)
        {
            GhostSpec spec;
            auto append = [&](EntityKind anchor, std::vector<AdjKind> hops, EntityKind target)
            { spec.chains.push_back({anchor, std::move(hops), target}); };
            auto cells = [&](EntityKind anchor, const std::vector<AdjKind> &path)
            {
                append(anchor, path, EntityKind::Cell);
                auto nodes = path;
                nodes.push_back(Adj::Cell2Node);
                append(anchor, nodes, EntityKind::Node);
                auto boundaries = nodes;
                boundaries.push_back(nodeBoundaries);
                append(anchor, boundaries, EntityKind::Bnd);
                boundaries.push_back(Adj::Bnd2Node);
                append(anchor, boundaries, EntityKind::Node);
            };
            append(EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node);
            append(EntityKind::Cell, {Adj::Cell2Node, nodeBoundaries}, EntityKind::Bnd);
            append(EntityKind::Cell, {Adj::Cell2Node, nodeBoundaries, Adj::Bnd2Node}, EntityKind::Node);
            std::vector<AdjKind> layers;
            for (int layer = 0; layer < options.cellLayers; ++layer)
            {
                layers.push_back(cellNeighbors);
                cells(EntityKind::Cell, layers);
            }
            append(EntityKind::Bnd, {Adj::Bnd2Node}, EntityKind::Node);
            append(EntityKind::Bnd, {Adj::Bnd2Node, nodeBoundaries}, EntityKind::Bnd);
            append(EntityKind::Bnd, {Adj::Bnd2Node, nodeBoundaries, Adj::Bnd2Node}, EntityKind::Node);
            cells(EntityKind::Bnd, {boundaryCells});
            if (options.completeNodeStar)
            {
                cells(EntityKind::Node, {nodeCells});
                append(EntityKind::Node, {nodeBoundaries}, EntityKind::Bnd);
                append(EntityKind::Node, {nodeBoundaries, Adj::Bnd2Node}, EntityKind::Node);
            }
            return spec;
        }

        /// Preserve ordered cone slots and explicit absent/unavailable states.
        /// Only outer support/neighbor rows may omit a valid global target.
        Runtime::ArrayPair<index> LocalRows(const RelationReadAccess &relation, bool cone, bool completeOwned,
                                            bool externalBoundarySlot = false)
        {
            auto convert = [&](index first, index count)
            {
                Runtime::RowBuilder builder(size_t(count), sizeof(index),
                                            externalBoundarySlot ? Runtime::RowLayoutKind::Table : Runtime::RowLayoutKind::CSR,
                                            externalBoundarySlot ? 2 : 0);
                for (index row = 0; row < count; ++row)
                {
                    const auto source = first + row;
                    DNDS_check_throw_info(!externalBoundarySlot || relation.RowSize(source) == 1,
                                          "external boundary must have exactly one all-node containing cell");
                    builder.ResizeRow(size_t(row), externalBoundarySlot ? 2 : relation.RowSize(source), false);
                    auto output = builder.Row<index>(size_t(row));
                    if (externalBoundarySlot)
                        output[1] = UnInitIndex;
                    for (size_t slot = 0; slot < relation.RowSize(source); ++slot)
                        output[slot] = relation.ConvertedIndex(source, slot, Adj_PointToLocal,
                                                               !cone && (!completeOwned || source >= relation.NumOwned()));
                }
                return Runtime::Array<index>::FromSharedStorage(builder.Commit());
            };
            return {convert(0, relation.NumOwned()), convert(relation.NumOwned(), relation.Rows() - relation.NumOwned())};
        }

        template <class T>
        Runtime::ArrayPair<T> Companion(const NativeGhostPublication &publication, EntityKind kind,
                                        Runtime::Array<T> owned, const Runtime::ElementSchema &schema)
        {
            const auto &space = publication.spaces.at(kind);
            auto ghost = publication.routes.at(kind).TransferRows(space, owned, schema);
            if (!owned.Layout().IsVariable())
            {
                // Generic row transfer discovers tightly packed CSR rows.
                // Uniform companion views also promise a component width,
                // including on empty ranks. Rebind only the logical layout;
                // the same Buffer retains its values and access controller.
                const auto width = owned.Layout().Width();
                const auto layout = ghost.Layout();
                DNDS_check_throw_info(layout.IsRagged() || (!layout.IsVariable() && layout.Stride() == width),
                                      "native uniform companion transfer is not tightly packed");
                for (size_t row = 0; row < layout.Rows(); ++row)
                    DNDS_check_throw_info(layout.RowSize(row) == width,
                                          "native uniform companion received a different component width");
                ghost = Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(layout.Rows(), width), ghost.Storage());
            }
            return {std::move(owned), std::move(ghost)};
        }

        Runtime::Array<index> CombinedIds(const Runtime::ArrayPair<index> &pair)
        {
            auto read = pair.ReadCombined();
            Runtime::Array<index> result(Runtime::RowLayout::Dense(read.View().Rows(), 1));
            auto write = result.Write();
            for (size_t row = 0; row < read.View().Rows(); ++row)
                write.View()(row, 0) = read.View()(row, 0);
            return result;
        }
    }

    std::shared_ptr<const NativeMeshState> NativeMeshBuilder::Build(NativeMeshInput input, NativeMeshBuildOptions options)
    {
        const auto mpi = input.mpi;
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native mesh builder requires a communicator");
        LocalStage(mpi, "input validation", [&]
                   {
            int actualRank = -1, actualSize = 0;
            MPI_Comm_rank(mpi.comm, &actualRank);
            MPI_Comm_size(mpi.comm, &actualSize);
            DNDS_check_throw_info(mpi.rank == actualRank && mpi.size == actualSize &&
                                      (input.dimension >= 1 && input.dimension <= 3) && options.cellLayers >= 1,
                                  "invalid native mesh MPI/dimension/layer options");
            DNDS_check_throw_info(input.periodic ? bool(input.cellNodesPbi && input.boundaryNodesPbi) : !input.cellNodesPbi && !input.boundaryNodesPbi,
                                  "native periodic flag and ordered cone companions disagree");
            for (const auto *space : {&input.cells, &input.nodes, &input.boundaries})
            {
                space->Validate();
                DNDS_check_throw_info(space->MeshIdentity() == input.nodes.MeshIdentity() && !space->NumGhost(),
                                      "native input spaces must be sibling owned-only families");
            }
            DNDS_check_throw_info(input.cells.Kind() == EntityKind::Cell && input.nodes.Kind() == EntityKind::Node &&
                                      input.boundaries.Kind() == EntityKind::Bnd, "native input entity kinds are invalid");
            DNDS_check_throw_info(input.dimension != 1 || input.boundaries.NumGlobal() == 0,
                                  "native one-dimensional primary meshes require an empty boundary family; point elements are unsupported");
            CheckRows(input.coordinates, input.nodes.NumOwned(), 3);
            CheckCone(input.cellNodes, input.cellInfo, input.cells, input.nodes, input.dimension, false, input.cellNodesPbi ? &*input.cellNodesPbi : nullptr);
            CheckCone(input.boundaryNodes, input.boundaryInfo, input.boundaries, input.nodes, input.dimension - 1, true, input.boundaryNodesPbi ? &*input.boundaryNodesPbi : nullptr);
            input.coordinates = UniformTable(std::move(input.coordinates), 3);
            input.cellInfo = UniformTable(std::move(input.cellInfo), 1);
            input.boundaryInfo = UniformTable(std::move(input.boundaryInfo), 1);
            // Register validates each space's communicator against the caller
            // before WithGhosts enters that descriptor's own collectives.
            NativeConnectivity validate(mpi);
            validate.Register(RelationReadAccess(Adj::Cell2Node, input.cells, input.nodes, Adj_PointToGlobal, input.cellNodes));
            validate.Register(RelationReadAccess(Adj::Bnd2Node, input.boundaries, input.nodes, Adj_PointToGlobal, input.boundaryNodes));
            for (const auto &[kind, ids] : input.ids)
            {
                DNDS_check_throw_info(kind == EntityKind::Cell || kind == EntityKind::Node || kind == EntityKind::Bnd,
                                      "native input IDs name an unsupported family");
                const auto &space = kind == EntityKind::Cell ? input.cells : kind == EntityKind::Node ? input.nodes : input.boundaries;
                CheckRows(ids, space.NumOwned(), 1);
                auto read = ids.Read();
                std::set<index> unique;
                for (size_t row = 0; row < ids.Rows(); ++row)
                {
                    const auto id = read.View()(row, 0);
                    DNDS_check_throw_info(id >= 0 && id != UnInitIndex && unique.insert(id).second,
                                          "native input IDs are invalid or locally duplicated");
                }
            }
            return true; });
        int idMask = 0;
        for (const auto &[kind, unused] : input.ids)
            idMask |= 1 << int(kind);
        int settings[]{input.dimension, options.cellLayers, int(options.completeNodeStar), idMask, int(input.periodic)};
        int minimum[5], maximum[5];
        MPI_Allreduce(settings, minimum, 5, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(settings, maximum, 5, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 5, maximum), "native mesh build options differ between ranks");
        if (input.periodic)
        {
            std::array<real, 60> values{}, root{};
            LocalStage(mpi, "periodic geometry validation", [&]
                       {
                size_t offset = 0;
                for (size_t frame = 0; frame < 4; ++frame)
                {
                    const auto rotation = input.periodicInfo.rotation[frame].map();
                    DNDS_check_throw_info(rotation.allFinite() && (rotation.transpose() * rotation - tGPoint::Identity()).norm() < 1e-10 &&
                                              std::abs(rotation.determinant() - 1) < 1e-10 && input.periodicInfo.translation[frame].map().allFinite() &&
                                              input.periodicInfo.rotationCenter[frame].map().allFinite(), "invalid native rigid periodic frame");
                    for (int i = 0; i < 9; ++i) values[offset++] = rotation.data()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = input.periodicInfo.translation[frame].map()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = input.periodicInfo.rotationCenter[frame].map()[i];
                }
                return true; });
            root = values;
            MPI_Bcast(root.data(), int(root.size()), DNDS_MPI_REAL, 0, mpi.comm);
            LocalStage(mpi, "periodic geometry agreement", [&]
                       { DNDS_check_throw_info(values == root, "periodic geometry differs between ranks"); return true; });
        }

        auto referenced = LocalStage(mpi, "initial node closure", [&]
                                     { return ReferencedGhostNodes(input); });
        const auto nodes = input.nodes.WithGhosts(std::move(referenced));
        auto cellNodes = LocalStage(mpi, "cell cone", [&]
                                    { return NativeRelation(Adj::Cell2Node, input.cells, nodes, input.cellNodes); });
        auto boundaryNodes = LocalStage(mpi, "boundary cone", [&]
                                        { return NativeRelation(Adj::Bnd2Node, input.boundaries, nodes, input.boundaryNodes); });
        auto nodeCells = NativeConnectivity::Inverse(cellNodes.Read(), mpi);
        auto nodeBoundaries = NativeConnectivity::Inverse(boundaryNodes.Read(), mpi);
        RelationGhostRoute initialNodeRoute(nodes);
        nodeCells = nodeCells.WithGhostRows(initialNodeRoute);
        nodeBoundaries = nodeBoundaries.WithGhostRows(initialNodeRoute);
        auto cellCells = LocalStage(mpi, "cell neighbor composition", [&]
                                    { return NativeConnectivity::Compose(cellNodes.Read(), nodeCells.Read(), true); });
        auto containment = LocalStage(mpi, "boundary cell containment", [&]
                                      { return NativeConnectivity::ComposeAll(boundaryNodes.Read(), nodeCells.Read()); });
        const auto boundaryCellKind = containment.Read().Kind();
        std::optional<NativeRelation> periodicExact, physicalBoundary;
        if (input.periodic)
        {
            const auto exactKind = AdjKind::Composition(boundaryNodes.Read().Kind(), nodeCells.Read().Kind(), RelationSelection::Exact, false,
                                                        "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched);
            auto matched = PeriodicBoundaryRelations(input, containment.Read(), exactKind);
            periodicExact.emplace(std::move(matched.first));
            physicalBoundary.emplace(std::move(matched.second));
        }
        else
        {
            LocalStage(mpi, "external boundary incidence", [&]
                       {
            auto read = containment.Read();
            for (index row = 0; row < read.NumOwned(); ++row)
                DNDS_check_throw_info(read.RowSize(row) == 1, "external boundary must have exactly one all-node containing cell");
            return true; });
            ValidateBoundaryFaces(input, containment.Read());
        }
        NativeConnectivity connectivity(mpi);
        LocalStage(mpi, "topology registration", [&]
                   {
            connectivity.Register(cellNodes.Read());
            connectivity.Register(boundaryNodes.Read());
            connectivity.Register(nodeCells.Read());
            connectivity.Register(nodeBoundaries.Read());
            connectivity.Register(cellCells.Read());
            connectivity.Register(containment.Read());
            if (periodicExact) connectivity.Register(periodicExact->Read());
            if (physicalBoundary) connectivity.Register(physicalBoundary->Read());
            return true; });
        auto tree = LocalStage(mpi, "primary closure schedule", [&]
                               { return CompiledGhostTree::compile(PrimaryClosure(cellCells.Read().Kind(), nodeCells.Read().Kind(),
                                                                                  nodeBoundaries.Read().Kind(), physicalBoundary ? Adj::Bnd2Cell : boundaryCellKind, options)); });
        auto publication = connectivity.BuildGhosted(connectivity.Evaluate(tree));
        auto data = LocalStage(mpi, "local topology publication", [&]
                               {
            NativeMeshData result;
            result.identity = std::move(input.identity);
            result.provenance = std::move(input.provenance);
            result.dimension = input.dimension;
            result.periodic = input.periodic;
            result.periodicInfo = input.periodicInfo;
            auto relation = [&](MeshRelation slot, const AdjKind &kind, bool cone, bool completeOwned)
            {
                auto rows = LocalRows(publication.relations.at(kind).Read(), cone, completeOwned);
                result.namedRelations.emplace(kind, NativeNamedRelation{rows, Adj_PointToLocal});
                result.relations[size_t(slot)] = std::move(rows);
            };
            relation(MeshRelation::CellNodes, Adj::Cell2Node, true, true);
            relation(MeshRelation::BoundaryNodes, Adj::Bnd2Node, true, true);
            relation(MeshRelation::CellCells, cellCells.Read().Kind(), false, true);
            // The native containment relation stays sorted/set-valued. Only
            // this explicit facade slot adds the external second-cell sentinel.
            result.relations[size_t(MeshRelation::BoundaryCells)] = LocalRows(
                publication.relations.at(physicalBoundary ? Adj::Bnd2Cell : boundaryCellKind).Read(), false, true, !physicalBoundary);
            result.namedRelations.emplace(boundaryCellKind, NativeNamedRelation{
                LocalRows(publication.relations.at(boundaryCellKind).Read(), false, !physicalBoundary), Adj_PointToLocal});
            if (periodicExact)
            {
                result.namedRelations.emplace(periodicExact->Read().Kind(), NativeNamedRelation{LocalRows(publication.relations.at(periodicExact->Read().Kind()).Read(), false, true), Adj_PointToLocal});
                result.namedRelations.emplace(Adj::Bnd2Cell, NativeNamedRelation{*result.relations[size_t(MeshRelation::BoundaryCells)], Adj_PointToLocal});
            }
            relation(MeshRelation::NodeCells, nodeCells.Read().Kind(), false, options.completeNodeStar);
            relation(MeshRelation::NodeBoundaries, nodeBoundaries.Read().Kind(), false, options.completeNodeStar);
            return result; });
        data.coords = Companion(publication, EntityKind::Node, input.coordinates, Runtime::ElementSchema::Arithmetic<real>());
        if (input.periodic)
        {
            const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
            data.cellNodesPbi = Companion(publication, EntityKind::Cell, *input.cellNodesPbi, bits);
            data.bndNodesPbi = Companion(publication, EntityKind::Bnd, *input.boundaryNodesPbi, bits);
        }
        const auto elemSchema = Runtime::ElementSchema::For<ElemInfo>();
        data.cellInfo = Companion(publication, EntityKind::Cell, input.cellInfo, elemSchema);
        data.bndInfo = Companion(publication, EntityKind::Bnd, input.boundaryInfo, elemSchema);
        std::map<EntityKind, EntitySpace> spaces;
        std::map<EntityKind, Runtime::Array<index>> ids;
        for (auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd})
        {
            const auto &space = publication.spaces.at(kind);
            spaces.emplace(kind, space);
            auto owned = LocalStage(mpi, "persistent ID initialization", [&]
                                    {
                const auto supplied = input.ids.find(kind);
                if (supplied != input.ids.end()) return supplied->second;
                Runtime::Array<index> generated(Runtime::RowLayout::Dense(size_t(space.NumOwned()), 1));
                auto write = generated.Write();
                for (index row = 0; row < space.NumOwned(); ++row)
                    write.View()(size_t(row), 0) = space.LocalToGlobal(row);
                return generated; });
            auto pair = Companion(publication, kind, std::move(owned), Runtime::ElementSchema::Arithmetic<index>());
            ids.emplace(kind, LocalStage(mpi, "persistent ID publication", [&]
                                         { return CombinedIds(pair); }));
        }
        MeshReadRequirements requirements;
        requirements.inverse = true;
        return LocalStage(mpi, "native state publication", [&]
                          { return NativeMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), requirements); });
    }
}
