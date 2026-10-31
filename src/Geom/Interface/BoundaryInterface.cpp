#include "BoundaryInterface.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Geom
{
    namespace R = Runtime;
    namespace Detail
    {
        struct BoundaryPatchData
        {
            ssp<UnstructuredMesh> mesh;
            R::VersionStamp stamp;
            EntitySpace cells;
            std::vector<BoundaryElementRecord> elements;
            std::optional<SurfacePatch> surface;
            explicit BoundaryPatchData(ssp<UnstructuredMesh> m)
                : mesh(std::move(m)), stamp(mesh->versions.Stamp(R::Change::Topology | R::Change::Geometry | R::Change::Distribution | R::Change::Layout)),
                  cells(mesh->Entities(EntityKind::Cell)) {}
        };
        struct InterfacePointMapData
        {
            BoundaryPatch donor;
            std::vector<InterfacePointMatch> matches;
            explicit InterfacePointMapData(BoundaryPatch patch) : donor(std::move(patch)) {}
        };
        struct DistributedInterfaceMapData
        {
            BoundaryPatch donor;
            std::vector<DistributedInterfaceMatch> matches;
            explicit DistributedInterfaceMapData(BoundaryPatch patch) : donor(std::move(patch)) {}
        };
    }
    BoundaryStencilPlan BoundaryStencilPlan::Prepare(UnstructuredMesh &mesh, const GhostSpec &spec, MeshReadRequirements requirements)
    {
        AdjacencyRegistry registry(mesh.getMPI());
        registry.RegisterMesh(mesh, requirements);
        std::optional<CompiledGhostTree> program;
        MPI::LocalStage(mesh.getMPI(), [&]
                        { program.emplace(CompiledGhostTree::compile(spec)); });
        auto result = registry.BuildGhosted(registry.Evaluate(*program));
        BoundaryStencilPlan plan;
        plan._publication = std::make_shared<const AdjacencyGhostPublication>(std::move(result));
        return plan;
    }
    MeshAdjacencyRead BoundaryStencilPlan::Read(const AdjacencyDescriptor &descriptor) const
    {
        DNDS_check_throw_info(_publication, "boundary stencil is unprepared");
        const auto found = _publication->relations.find(descriptor);
        DNDS_check_throw_info(found != _publication->relations.end(), "adjacency is absent from boundary stencil publication");
        return found->second.Read();
    }
    BoundaryPatch BoundaryPatch::Create(ssp<UnstructuredMesh> mesh, const std::vector<index> &boundaries, RigidPose pose, BoundingBoxBackend backend)
    {
        DNDS_check_throw_info(mesh, "boundary patch requires a mesh");
        pose.Validate();
        auto meshRead = mesh->AcquireRead(MeshReadRequirements::CellFlux());
        const auto view = meshRead.HostView();
        auto data = std::make_shared<Detail::BoundaryPatchData>(mesh);
        data->elements.resize(boundaries.size());
        std::vector<size_t> lengths(boundaries.size());
        std::vector<SurfaceElementRecord> elements(boundaries.size());
        auto sorted = boundaries;
        std::sort(sorted.begin(), sorted.end());
        DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "boundary patch contains duplicate elements");
        for (index i = 0; i < index(boundaries.size()); ++i)
        {
            const index iBnd = boundaries[size_t(i)];
            DNDS_check_throw_info(iBnd >= 0 && iBnd < view.NumBnd(), "patch selection must name owned boundary rows");
            const auto links = view.BoundaryFaces(iBnd);
            DNDS_check_throw_info(links.size() == 1 && links[0] >= 0 && links[0] < view.NumFaceProc(), "boundary patch requires a linked face");
            const index face = links[0];
            DNDS_check_throw_info(FaceIDIsExternalBC(view.GetFaceZone(face)), "periodic interface images require explicit image selection");
            const index cell = view.FaceCells(face)[0];
            DNDS_check_throw_info(cell >= 0 && cell < view.NumCellProc(), "boundary patch lacks adjacent volume cell");
            const auto faces = view.CellFaces(cell);
            const auto slot = std::find(faces.begin(), faces.end(), face) - faces.begin();
            DNDS_check_throw_info(slot < faces.size(), "boundary patch face absent from cell incidence");
            const auto type = view.GetFaceElement(face).type;
            const index id = mesh->OriginalEntityId(EntityKind::Bnd, iBnd);
            data->elements[size_t(i)] = {iBnd, face, cell, slot, id, type};
            elements[size_t(i)] = {id, type};
            lengths[size_t(i)] = 3 * view.FaceNodes(face).size();
        }
        R::Array<real> coordinates(R::RowLayout::Ragged(lengths));
        {
            auto write = coordinates.Write();
            const auto coords = write.View<R::CSRRows>();
            tSmallCoords faceCoordinates;
            for (index i = 0; i < index(boundaries.size()); ++i)
            {
                view.GetCoordsOnFace(data->elements[size_t(i)].face, faceCoordinates);
                std::copy_n(faceCoordinates.data(), 3 * faceCoordinates.cols(), coords[i].data());
            }
        }
        data->surface.emplace(SurfacePatch::Create(std::move(elements), std::move(coordinates), pose, backend));
        return BoundaryPatch(std::move(data));
    }
    BoundaryPatch BoundaryPatch::WithPose(RigidPose pose) const
    {
        Validate();
        auto next = std::make_shared<Detail::BoundaryPatchData>(*_data);
        next->surface.emplace(_data->surface->WithPose(pose));
        return BoundaryPatch(std::move(next));
    }
    BoundaryPatch BoundaryPatch::WithBackend(BoundingBoxBackend backend) const
    {
        Validate();
        auto next = std::make_shared<Detail::BoundaryPatchData>(*_data);
        next->surface.emplace(_data->surface->WithBackend(backend));
        return BoundaryPatch(std::move(next));
    }
    BoundingBoxBackend BoundaryPatch::Backend() const { return _data->surface->Backend(); }
    void BoundaryPatch::Validate() const
    {
        DNDS_check_throw_info(_data && _data->stamp.Valid() && _data->cells.Valid(), "boundary patch geometry/order is stale");
    }
    index BoundaryPatch::Size() const { return index(_data->elements.size()); }
    BoundaryElementRecord BoundaryPatch::Element(index i) const
    {
        Validate();
        DNDS_check_throw_info(i >= 0 && i < Size(), "boundary patch element index out of range");
        return _data->elements[size_t(i)];
    }
    const EntitySpace &BoundaryPatch::Cells() const { return _data->cells; }
    RigidPose BoundaryPatch::Pose() const { return _data->surface->Pose(); }
    SurfacePatch BoundaryPatch::SurfaceGeometry() const
    {
        Validate();
        return *_data->surface;
    }

    InterfacePointMap InterfacePointMap::Match(BoundaryPatch donor, const R::Array<real> &points,
                                               const R::Array<index> &queryIds, InterfaceSearchSettings settings)
    {
        donor.Validate();
        DNDS_check_throw_info(donor._data->mesh->getMPI().size == 1, "distributed interface search requires DistributedInterfacePointMap");
        auto meshRead = donor._data->mesh->AcquireRead(MeshReadRequirements::CellFlux());
        donor.Validate();
        const auto map = SurfacePointMap::Match(donor.SurfaceGeometry(), points, queryIds, settings);
        auto data = std::make_shared<Detail::InterfacePointMapData>(donor);
        data->matches.reserve(size_t(map.Size()));
        for (index iQuery = 0; iQuery < map.Size(); ++iQuery)
        {
            const auto match = map.Get(iQuery);
            const auto record = donor.Element(match.element);
            data->matches.push_back({match.queryId, match.element, record.boundary, record.cell, match.elementId, match.reference, match.projectedPoint, match.surfacePoint, match.distance});
        }
        return InterfacePointMap(std::move(data));
    }
    void InterfacePointMap::Validate() const
    {
        DNDS_check_throw_info(_data, "unprepared interface map");
        _data->donor.Validate();
    }
    index InterfacePointMap::Size() const { return index(_data->matches.size()); }
    InterfacePointMatch InterfacePointMap::Get(index i) const
    {
        Validate();
        DNDS_check_throw_info(i >= 0 && i < Size(), "interface map index out of range");
        return _data->matches[size_t(i)];
    }
    const BoundaryPatch &InterfacePointMap::Donor() const { return _data->donor; }

    DistributedInterfacePointMap DistributedInterfacePointMap::Match(BoundaryPatch donor, const R::Array<real> &points,
                                                                     const R::Array<index> &queryIds, InterfaceRoutingMode mode, InterfaceSearchSettings settings)
    {
        const auto mpi = donor._data->mesh->getMPI();
        std::optional<MeshReadAccess> meshRead;
        MPI::LocalStage(mpi, [&]
                        {
            donor.Validate();
            meshRead.emplace(donor._data->mesh->AcquireRead(MeshReadRequirements::CellFlux()));
            DNDS_check_throw_info(mode == InterfaceRoutingMode::CoarseBoxes || mode == InterfaceRoutingMode::AllGatherBoundary,"unknown interface routing mode"); });
        const auto map = DistributedSurfacePointMap::Match(mpi, donor.SurfaceGeometry(), points, queryIds,
                                                           mode == InterfaceRoutingMode::CoarseBoxes ? SurfaceRoutingMode::CoarseBoxes : SurfaceRoutingMode::AllGatherSurface, settings);
        // Attach mesh semantics only AFTER generic geometric location. The
        // locator's selected-element route is also reusable for other clients'
        // element data, rather than embedding volume rows in search packets.
        R::Array<index> links, received;
        MPI::LocalStage(mpi, [&]
                        {
            links = R::Array<index>(R::RowLayout::Dense(donor.Size(), 5));
            received = R::Array<index>(R::RowLayout::Dense(map.Size(), 5));
            auto write = links.Write();
            for (index iElement = 0; iElement < donor.Size(); ++iElement)
            {
                const auto record = donor.Element(iElement);
                const auto global = donor.Cells().LocalToGlobal(record.cell);
                const index row[]{record.boundary,record.cell,record.originalId,donor.Cells().Owner(global),donor.Cells().OwnerLocalRow(global)};
                std::copy_n(row,5,write.View()[iElement].data());
            } });
        const auto route = map.BuildElementDataRoute();
        auto binding = R::PayloadPlan(route, links.Layout(), received.Layout(), R::ElementSchema::Arithmetic<index>()).Bind(links, received, R::TransferTransport::Direct, {R::TransferMode::Transient, route.Context(), 0});
        binding.Begin().Wait();
        std::shared_ptr<Detail::DistributedInterfaceMapData> data;
        MPI::LocalStage(mpi, [&]
                        {
            data = std::make_shared<Detail::DistributedInterfaceMapData>(donor);
            data->matches.reserve(size_t(map.Size()));
            auto read = received.Read();
            const auto rows = read.View<R::FixedRows<5>>();
            for (index iQuery = 0; iQuery < map.Size(); ++iQuery)
            {
                const auto match = map.Get(iQuery);
                DNDS_check_throw_info(rows(iQuery,2) == match.elementId,"selected surface metadata ID mismatch");
                data->matches.push_back({{match.queryId,match.element,rows(iQuery,0),rows(iQuery,1),match.elementId,
                    match.reference,match.projectedPoint,match.surfacePoint,match.distance},match.geometryRank,int(rows(iQuery,3)),rows(iQuery,4)});
            } });
        return DistributedInterfacePointMap(std::move(data));
    }
    void DistributedInterfacePointMap::Validate() const
    {
        DNDS_check_throw_info(_data, "unprepared distributed interface map");
        _data->donor.Validate();
    }
    void DistributedInterfacePointMap::ValidateFor(const BoundaryPatch &active) const
    {
        Validate();
        active.Validate();
        DNDS_check_throw_info(active._data == _data->donor._data, "active interface patch or pose differs from map");
    }
    index DistributedInterfacePointMap::Size() const { return index(_data->matches.size()); }
    const BoundaryPatch &DistributedInterfacePointMap::Donor() const { return _data->donor; }
    DistributedInterfaceMatch DistributedInterfacePointMap::Get(index iQuery) const
    {
        Validate();
        DNDS_check_throw_info(iQuery >= 0 && iQuery < Size(), "distributed interface query outside map");
        return _data->matches[size_t(iQuery)];
    }
}
