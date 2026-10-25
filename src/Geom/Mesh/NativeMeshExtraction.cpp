#include "NativeMeshExtraction.hpp"
#include "NativeMeshAssembler.hpp"
#include "FieldTransfer.hpp"
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
            DNDS_check_throw_info(!owned.periodic && owned.dimension >= 2 && owned.dimension <= 3,
                                  "native boundary extraction requires nonperiodic two- or three-dimensional input");
            auto metadata = owned.bndInfo->Read();
            for (size_t row = 0; row < owned.bndInfo->Rows(); ++row)
                DNDS_check_throw_info(!FaceIDIsPeriodic(metadata.View()(row, 0).zone),
                                      "native boundary extraction does not support periodic boundary zones"); });
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto oldNodes = owned.spaces.at(EntityKind::Node);
        const auto oldBoundaries = owned.spaces.at(EntityKind::Bnd);
        std::vector<std::vector<index>> cones;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto read = owned.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
            cones.resize(size_t(oldBoundaries.NumOwned()));
            for (size_t row = 0; row < cones.size(); ++row)
            {
                const auto values = read.View()[row];
                cones[row].assign(values.begin(), values.end());
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
                    DNDS_check_throw_info(boundaryFaces.View().RowSize(row) <= 1, "boundary extraction requires at most one mapped physical face");
                    if (boundaryFaces.View().RowSize(row) == 0) continue;
                    const auto face = boundaryFaces.View()(row, 0);
                    if (face == UnInitIndex) continue;
                    requests.push_back({faces.Owner(face), faces.OwnerLocalRow(face), row});
                } });
            Runtime::RowTransferPlan route(context, size_t(faces.NumOwned()), cones.size(), requests);
            auto orderedFaces = Fetch(route, *owned.relations[size_t(MeshRelation::FaceNodes)], mpi, context);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto read = orderedFaces.Read();
                for (size_t row = 0; row < cones.size(); ++row)
                {
                    const auto values = read.View()[row];
                    if (!values.size()) continue;
                    std::vector<index> expected = cones[row], actual(values.begin(), values.end());
                    std::sort(expected.begin(), expected.end());
                    std::sort(actual.begin(), actual.end());
                    DNDS_check_throw_info(actual == expected, "mapped boundary face has different nodes");
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
        const auto cells = nodes.CreateSibling(EntityKind::Cell, oldBoundaries.NumOwned());
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
            result.provenance.sourceIds.emplace(EntityKind::Cell, owned.ids.at(EntityKind::Bnd)); });
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
            input.emplace(NativeMeshInput{mpi, owned.dimension - 1, cells, nodes, boundaries, Rows(cones), Rows({}),
                                         std::move(*coordinates), *owned.bndInfo,
                                         Runtime::Array<ElemInfo>(Runtime::RowLayout::Dense(0, 1))});
            input->provenance = result.provenance; });
        result.state = NativeMeshBuilder::Build(std::move(*input), options);
        return result;
    }
}
