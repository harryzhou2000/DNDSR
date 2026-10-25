#include "NativeMeshExtraction.hpp"
#include "NativeMeshAssembler.hpp"
#include "FieldTransfer.hpp"
#include "NativePeriodicMatch.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        using Array = Runtime::Array<index>;
        Array Rows(const std::vector<std::vector<index>> &rows)
        {
            Runtime::RowBuilder builder(rows.size(), sizeof(index));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                std::copy(rows[row].begin(), rows[row].end(), builder.Row<index>(row).data);
            }
            return Array::FromSharedStorage(builder.Commit());
        }
        Array Fetch(const Runtime::RowTransferPlan &route, const Array &source, const MPIInfo &mpi,
                    const std::shared_ptr<Runtime::CommunicationContext> &context)
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

    NativeMeshDerivedResult NativeMeshExtraction::ExtractBoundary(const NativeMeshState &source, const MPIInfo &mpi,
                                                                  NativeMeshBuildOptions options)
    {
        auto owned = NativeMeshAssembler::CaptureOwned(source, mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(owned.dimension >= 2 && owned.dimension <= 3,
                                  "native boundary extraction requires two- or three-dimensional input");
            auto metadata = owned.bndInfo->Read();
            for (size_t row = 0; row < owned.bndInfo->Rows(); ++row)
                DNDS_check_throw_info(FaceIDIsExternalBC(metadata.View()(row, 0).zone) ||
                                          (owned.periodic && FaceIDIsPeriodic(metadata.View()(row, 0).zone)),
                                      "native boundary extraction requires external or declared periodic boundary zones"); });
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto oldNodes = owned.spaces.at(EntityKind::Node);
        const auto oldBoundaries = owned.spaces.at(EntityKind::Bnd);
        std::vector<std::vector<index>> cones;
        std::vector<std::vector<NodePeriodicBits>> images;
        std::vector<size_t> selectedBoundaries;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto read = owned.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
            auto info = owned.bndInfo->Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> bits;
            if (owned.periodic) bits.emplace(owned.bndNodesPbi->Read());
            for (size_t row = 0; row < size_t(oldBoundaries.NumOwned()); ++row)
            {
                if (FaceIDIsPeriodic(info.View()(row, 0).zone)) continue;
                selectedBoundaries.push_back(row);
                const auto values = read.View()[row];
                cones.emplace_back(values.begin(), values.end());
                images.emplace_back();
                if (bits) for (auto value : bits->View()[row]) images.back().push_back(value);
            } });
        if (owned.requirements.faces)
        {
            const auto faces = owned.spaces.at(EntityKind::Face);
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto boundaryFaces = owned.relations[size_t(MeshRelation::BoundaryFaces)]->Read();
                for (size_t row = 0; row < cones.size(); ++row)
                {
                    const auto boundary = selectedBoundaries[row];
                    DNDS_check_throw_info(boundaryFaces.View().RowSize(boundary) <= 1, "boundary extraction requires at most one mapped physical face");
                    if (boundaryFaces.View().RowSize(boundary) == 0) continue;
                    const auto face = boundaryFaces.View()(boundary, 0);
                    if (face == UnInitIndex) continue;
                    requests.push_back({faces.Owner(face), faces.OwnerLocalRow(face), row});
                } });
            Runtime::RowTransferPlan route(context, size_t(faces.NumOwned()), cones.size(), requests);
            auto orderedFaces = Fetch(route, *owned.relations[size_t(MeshRelation::FaceNodes)], mpi, context);
            std::optional<Array> faceFrames;
            if (owned.periodic)
            {
                std::optional<Array> values;
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    auto read = owned.faceNodesPbi->Read();
                    std::vector<std::vector<index>> rows(owned.faceNodesPbi->Rows());
                    for (size_t row = 0; row < rows.size(); ++row) for (auto bits : read.View()[row]) rows[row].push_back(uint8_t(bits));
                    values = Rows(rows); });
                faceFrames = Fetch(route, *values, mpi, context);
            }
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto read = orderedFaces.Read();
                std::optional<Array::ReadLease> frames;
                if (faceFrames) frames.emplace(faceFrames->Read());
                for (size_t row = 0; row < cones.size(); ++row)
                {
                    const auto values = read.View()[row];
                    if (!values.size()) continue;
                    std::vector<index> expected = cones[row], actual(values.begin(), values.end());
                    std::sort(expected.begin(), expected.end());
                    std::sort(actual.begin(), actual.end());
                    DNDS_check_throw_info(actual == expected, "mapped boundary face has different nodes");
                    if (frames)
                    {
                        NativePeriodicNodes facePattern, boundaryPattern;
                        for (size_t slot = 0; slot < values.size(); ++slot)
                        {
                            facePattern.emplace_back(values[slot], uint8_t(frames->View()(row, slot)));
                            boundaryPattern.emplace_back(cones[row][slot], uint8_t(images[row][slot]));
                        }
                        const auto delta = NativePeriodicFrame(std::move(facePattern), std::move(boundaryPattern));
                        DNDS_check_throw_info(delta, "mapped boundary face images disagree with physical boundary");
                        for (size_t slot = 0; slot < values.size(); ++slot) images[row][slot] = NodePeriodicBits{uint8_t(frames->View()(row, slot) ^ *delta)};
                    }
                    cones[row].assign(values.begin(), values.end());
                } });
        }
        std::vector<index> requested;
        std::vector<Runtime::RowRequest> requests;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            std::set<index> unique;
            for (const auto &row : cones) unique.insert(row.begin(), row.end());
            requested.assign(unique.begin(), unique.end());
            for (size_t row = 0; row < requested.size(); ++row)
                requests.push_back({oldNodes.Owner(requested[row]), oldNodes.OwnerLocalRow(requested[row]), row}); });
        Runtime::RowTransferPlan discovery(context, size_t(oldNodes.NumOwned()), requested.size(), requests);
        std::vector<size_t> selected;
        std::vector<std::vector<size_t>> send;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            std::set<size_t> unique;
            for (const auto &peer : discovery.SendRows()) unique.insert(peer.begin(), peer.end());
            selected.assign(unique.begin(), unique.end());
            send = discovery.SendRows();
            for (auto &peer : send)
                for (auto &row : peer)
                    row = size_t(std::lower_bound(selected.begin(), selected.end(), row) - selected.begin()); });
        const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(selected.size()));
        const auto cells = nodes.CreateSibling(EntityKind::Cell, index(selectedBoundaries.size()));
        const auto boundaries = nodes.CreateSibling(EntityKind::Bnd, 0);
        std::optional<Array> labels;
        std::optional<Runtime::Array<real>> coordinates;
        NativeMeshDerivedResult result;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            labels.emplace(Runtime::RowLayout::Dense(selected.size(), 1));
            coordinates.emplace(Runtime::RowLayout::Dense(selected.size(), 3));
            auto writeLabels = labels->Write();
            auto writeCoords = coordinates->Write();
            auto originalCoords = owned.coords->Read();
            auto originalIds = owned.ids.at(EntityKind::Node).Read();
            std::vector<std::vector<index>> parentNodes(selected.size());
            for (size_t row = 0; row < selected.size(); ++row)
            {
                writeLabels.View()(row, 0) = nodes.LocalToGlobal(index(row));
                for (size_t component = 0; component < 3; ++component)
                    writeCoords.View()(row, component) = originalCoords.View()(selected[row], component);
                parentNodes[row] = {originalIds.View()(selected[row], 0)};
            }
            result.provenance.sourceNamespace = source.MeshNamespace();
            result.provenance.sourceKinds = {{EntityKind::Node, EntityKind::Node}, {EntityKind::Cell, EntityKind::Bnd}};
            result.provenance.sourceIds.emplace(EntityKind::Node, Rows(parentNodes));
            auto boundaryIds = owned.ids.at(EntityKind::Bnd).Read();
            std::vector<std::vector<index>> parentBoundaries;
            for (auto boundary : selectedBoundaries) parentBoundaries.push_back({boundaryIds.View()(boundary, 0)});
            result.provenance.sourceIds.emplace(EntityKind::Cell, Rows(parentBoundaries)); });
        // Discovery names original owner rows. Remap only its send rows into
        // compact owned order; the per-peer correspondence remains exact.
        auto route = Runtime::RowTransferPlan::Known(context, selected.size(), requested.size(), std::move(send), discovery.ReceiveRows());
        auto newGlobals = Fetch(route, *labels, mpi, context);
        std::optional<NativeMeshInput> input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto read = newGlobals.Read();
            for (auto &row : cones)
                for (auto &global : row)
                {
                    const auto found = std::lower_bound(requested.begin(), requested.end(), global);
                    DNDS_check_throw_info(found != requested.end() && *found == global, "native extraction node mapping is incomplete");
                    global = read.View()(size_t(found - requested.begin()), 0);
                }
            Runtime::Array<ElemInfo> info(Runtime::RowLayout::Dense(selectedBoundaries.size(), 1));
            { auto sourceInfo = owned.bndInfo->Read(); auto write = info.Write(); for (size_t row = 0; row < selectedBoundaries.size(); ++row) write.View()(row, 0) = sourceInfo.View()(selectedBoundaries[row], 0); }
            input.emplace(NativeMeshInput{mpi, owned.dimension - 1, cells, nodes, boundaries, Rows(cones), Rows({}),
                                         std::move(*coordinates), std::move(info),
                                         Runtime::Array<ElemInfo>(Runtime::RowLayout::Dense(0, 1))});
            input->provenance = result.provenance;
            if (owned.periodic)
            {
                Runtime::RowBuilder builder(images.size(), sizeof(NodePeriodicBits));
                for (size_t row = 0; row < images.size(); ++row)
                {
                    builder.ResizeRow(row, images[row].size());
                    std::copy(images[row].begin(), images[row].end(), builder.Row<NodePeriodicBits>(row).data);
                }
                input->periodic = true;
                input->periodicInfo = owned.periodicInfo;
                input->cellNodesPbi = Runtime::Array<NodePeriodicBits>::FromSharedStorage(builder.Commit());
                input->boundaryNodesPbi = Runtime::Array<NodePeriodicBits>(Runtime::RowLayout::Ragged({}));
            } });
        result.state = NativeMeshBuilder::Build(std::move(*input), options);
        return result;
    }
}
