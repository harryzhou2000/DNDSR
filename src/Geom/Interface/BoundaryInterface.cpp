#include "BoundaryInterface.hpp"
#include "Geom/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/SparseBatchExchange.hpp"

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
            RigidPose pose;
            std::vector<BoundaryElementRecord> elements;
            R::Array<real> coordinates;
            std::shared_ptr<const BoundingBoxTree> search;
            BoundaryPatchData(ssp<UnstructuredMesh> m, RigidPose p)
                : mesh(std::move(m)), stamp(mesh->versions.Stamp(R::Change::Topology | R::Change::Geometry | R::Change::Distribution | R::Change::Layout)),
                  cells(mesh->Entities(EntityKind::Cell)), pose(std::move(p)) {}
            void BuildIndex()
            {
                auto read = coordinates.Read();
                const auto coords = read.View<R::CSRRows>();
                std::vector<SpatialBox> boxes(elements.size());
                for (index i = 0; i < index(elements.size()); ++i)
                    for (index j = 0; j < coords[i].size(); j += 3)
                        boxes[i].Add(pose.Apply({coords(i, j), coords(i, j + 1), coords(i, j + 2)}));
                search = std::make_shared<BoundingBoxTree>(boxes);
            }
        };
        struct InterfacePointMapData
        {
            BoundaryPatch donor;
            std::vector<InterfacePointMatch> matches;
            explicit InterfacePointMapData(BoundaryPatch patch) : donor(std::move(patch)) {}
        };
    }
    void RigidPose::Validate() const
    {
        DNDS_check_throw_info(rotation.allFinite() && translation.allFinite() &&
                                  (rotation.transpose() * rotation).isApprox(tGPoint::Identity(), 1e-12) &&
                                  std::abs(rotation.determinant() - 1) < 1e-12,
                              "interface pose must be a proper rigid transform");
    }
    void InterfaceSearchSettings::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(maxGap) && maxGap >= 0 && std::isfinite(referenceTolerance) && referenceTolerance >= 0 &&
                                  std::isfinite(projectionTolerance) && projectionTolerance > 0 && maxIterations > 0,
                              "invalid interface search tolerances");
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
    BoundaryPatch BoundaryPatch::Create(ssp<UnstructuredMesh> mesh, const std::vector<index> &boundaries, RigidPose pose)
    {
        DNDS_check_throw_info(mesh, "boundary patch requires a mesh");
        pose.Validate();
        auto meshRead = mesh->AcquireRead(MeshReadRequirements::CellFlux());
        const auto view = meshRead.HostView();
        auto data = std::make_shared<Detail::BoundaryPatchData>(mesh, pose);
        data->elements.resize(boundaries.size());
        std::vector<size_t> lengths(boundaries.size());
        auto sorted = boundaries;
        std::sort(sorted.begin(), sorted.end());
        DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "boundary patch contains duplicate elements");
        for (index i = 0; i < index(boundaries.size()); ++i)
        {
            const index iBnd = boundaries[i];
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
            DNDS_check_throw_info(type == Elem::Line2 || type == Elem::Tri3 || type == Elem::Quad4,
                                  "initial interface search supports Line2/Tri3/Quad4 geometry");
            data->elements[i] = {iBnd, face, cell, slot, mesh->OriginalEntityId(EntityKind::Bnd, iBnd), type};
            lengths[i] = 3 * view.FaceNodes(face).size();
        }
        data->coordinates = R::Array<real>(R::RowLayout::Ragged(lengths));
        {
            auto write = data->coordinates.Write();
            const auto coords = write.View<R::CSRRows>();
            tSmallCoords faceCoordinates;
            for (index i = 0; i < index(boundaries.size()); ++i)
            {
                view.GetCoordsOnFace(data->elements[i].face, faceCoordinates);
                for (index j = 0; j < faceCoordinates.cols(); ++j)
                    for (int d = 0; d < 3; ++d)
                        coords(i, 3 * j + d) = faceCoordinates(d, j);
                const Elem::Element element{data->elements[i].type};
                const Elem::Quadrature q(element, 2);
                for (int iG = 0; iG < q.GetNumPoints(); ++iG)
                {
                    const auto &shape = Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(2).at(iG);
                    const auto jac = Elem::ShapeJacobianCoordD01Nj(faceCoordinates, shape);
                    const real measure = element.GetDim() == 1 ? jac.col(0).norm() : jac.col(0).cross(jac.col(1)).norm();
                    DNDS_check_throw_info(std::isfinite(measure) && measure > 0, "degenerate interface element");
                }
            }
        }
        data->BuildIndex();
        return BoundaryPatch(std::move(data));
    }
    BoundaryPatch BoundaryPatch::WithPose(RigidPose pose) const
    {
        Validate();
        pose.Validate();
        auto next = std::make_shared<Detail::BoundaryPatchData>(*_data);
        next->pose = std::move(pose);
        next->BuildIndex();
        return BoundaryPatch(std::move(next));
    }
    void BoundaryPatch::Validate() const
    {
        DNDS_check_throw_info(_data && _data->stamp.Valid() && _data->cells.Valid(), "boundary patch geometry/order is stale");
    }
    index BoundaryPatch::Size() const { return index(_data->elements.size()); }
    BoundaryElementRecord BoundaryPatch::Element(index i) const
    {
        Validate();
        DNDS_check_throw_info(i >= 0 && i < Size(), "boundary patch element index out of range");
        return _data->elements[i];
    }
    const EntitySpace &BoundaryPatch::Cells() const { return _data->cells; }
    RigidPose BoundaryPatch::Pose() const { return _data->pose; }
    namespace
    {
        bool Inside(Elem::ElemType type, const tPoint &p, real tolerance)
        {
            if (type == Elem::Tri3)
                return p[0] >= -tolerance && p[1] >= -tolerance && p[0] + p[1] <= 1 + tolerance;
            return std::abs(p[0]) <= 1 + tolerance && (type == Elem::Line2 || std::abs(p[1]) <= 1 + tolerance);
        }
        bool Project(Elem::ElemType type, const tSmallCoords &coords, const tPoint &query,
                     const InterfaceSearchSettings &settings, tPoint &reference, tPoint &point)
        {
            const Elem::Element element{type};
            const int dim = element.GetDim();
            reference = type == Elem::Tri3 ? tPoint{1. / 3, 1. / 3, 0} : tPoint::Zero();
            Elem::tD01Nj shape;
            bool converged = false;
            for (int iteration = 0; iteration < settings.maxIterations; ++iteration)
            {
                element.GetD01Nj(reference, shape);
                point = Elem::PPhysicsCoordD01Nj(coords, shape);
                const auto jac = Elem::ShapeJacobianCoordD01Nj(coords, shape);
                const auto tangent = jac.leftCols(dim);
                const Eigen::Matrix2d gram = dim == 2 ? Eigen::Matrix2d(tangent.transpose() * tangent) : Eigen::Matrix2d::Identity();
                tPoint step = tPoint::Zero();
                if (dim == 1)
                {
                    const real length2 = tangent.col(0).squaredNorm();
                    if (!(length2 > 0))
                        return false;
                    step[0] = tangent.col(0).dot(query - point) / length2;
                }
                else
                {
                    if (!(gram.determinant() > 0))
                        return false;
                    step.head<2>() = gram.ldlt().solve(tangent.transpose() * (query - point));
                }
                if (!step.allFinite())
                    return false;
                reference += step;
                if (step.norm() <= settings.projectionTolerance)
                {
                    converged = true;
                    break;
                }
            }
            if (!converged || !Inside(type, reference, settings.referenceTolerance))
                return false;
            element.GetD01Nj(reference, shape);
            point = Elem::PPhysicsCoordD01Nj(coords, shape);
            return (query - point).norm() <= settings.maxGap;
        }
    }
    InterfacePointMap InterfacePointMap::Match(BoundaryPatch donor, const R::Array<real> &points,
                                               const R::Array<index> &queryIds, InterfaceSearchSettings settings)
    {
        donor.Validate();
        settings.Validate();
        DNDS_check_throw_info(donor._data->mesh->getMPI().size == 1, "distributed interface search is not implemented; no implicit global gather");
        auto meshRead = donor._data->mesh->AcquireRead(MeshReadRequirements::CellFlux());
        donor.Validate();
        DNDS_check_throw_info(points.GetDevice() == R::Device::Host() && queryIds.GetDevice() == R::Device::Host() &&
                                  !points.Layout().IsVariable() && points.Layout().Width() == 3 &&
                                  !queryIds.Layout().IsVariable() && queryIds.Layout().Width() == 1 && points.RowCount() == queryIds.RowCount(),
                              "interface query arrays must be CPU points[N,3] and IDs[N,1]");
        auto pointRead = points.Read();
        auto idRead = queryIds.Read();
        const auto queries = pointRead.View<R::FixedRows<3>>();
        const auto ids = idRead.View<R::FixedRows<1>>();
        std::vector<index> sorted(size_t(queryIds.RowCount()));
        for (index i = 0; i < queryIds.RowCount(); ++i)
            sorted[i] = ids(i, 0);
        std::sort(sorted.begin(), sorted.end());
        DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "interface query IDs must be unique");
        auto data = std::make_shared<Detail::InterfacePointMapData>(donor);
        data->matches.resize(size_t(points.RowCount()));
        auto coordRead = donor._data->coordinates.Read();
        const auto coords = coordRead.View<R::CSRRows>();
        tSmallCoords placed;
        for (index i = 0; i < points.RowCount(); ++i)
        {
            const tPoint query{queries(i, 0), queries(i, 1), queries(i, 2)};
            DNDS_check_throw_info(query.allFinite(), "interface query contains nonfinite coordinates");
            bool found = false, interior = false;
            auto &best = data->matches[i];
            donor._data->search->Query(query, settings.maxGap, [&](index candidate)
                                       {
                const auto &element = donor._data->elements[candidate];
                placed.resize(3, coords[candidate].size() / 3);
                for (index j = 0; j < placed.cols(); ++j)
                    placed.col(j) = donor._data->pose.Apply({coords(candidate, 3 * j), coords(candidate, 3 * j + 1), coords(candidate, 3 * j + 2)});
                tPoint reference, projected;
                if (!Project(element.type, placed, query, settings, reference, projected)) return;
                const real distance = (query - projected).norm();
                // A zero acceptance tolerance must still distinguish an exact
                // shared edge/vertex from overlapping element interiors.
                const bool candidateInterior = Inside(element.type, reference,
                    -std::max(settings.referenceTolerance, 32 * std::numeric_limits<real>::epsilon()));
                const bool tied = found && std::abs(distance - best.distance) <= settings.projectionTolerance;
                DNDS_check_throw_info(!(tied && interior && candidateInterior), "ambiguous overlapping interface element interiors");
                if (!found || distance < best.distance - settings.projectionTolerance || (tied && element.originalId < best.donorOriginalId))
                {
                    best = {ids(i, 0), candidate, element.boundary, element.cell, element.originalId,
                            reference, projected, donor._data->pose.Inverse(projected), distance};
                    found = true, interior = candidateInterior;
                } });
            DNDS_check_throw_info(found, "uncovered interface query ID " + std::to_string(ids(i, 0)));
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
        return _data->matches[i];
    }
    const BoundaryPatch &InterfacePointMap::Donor() const { return _data->donor; }

    namespace Detail
    {
        struct DistributedInterfaceMapData
        {
            BoundaryPatch donor;
            std::vector<DistributedInterfaceMatch> matches;
            explicit DistributedInterfaceMapData(BoundaryPatch patch) : donor(std::move(patch)) {}
        };
    }

    DistributedInterfacePointMap DistributedInterfacePointMap::Match(BoundaryPatch donor,
                                                                     const R::Array<real> &points, const R::Array<index> &queryIds, InterfaceRoutingMode mode,
                                                                     InterfaceSearchSettings settings)
    {
        const auto mpi = donor._data->mesh->getMPI();
        auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
        std::shared_ptr<MeshReadAccess> meshRead;
        MPI::LocalStage(mpi, [&]
                        {
            donor.Validate();
            settings.Validate();
            meshRead = std::make_shared<MeshReadAccess>(donor._data->mesh->AcquireRead(MeshReadRequirements::CellFlux()));
            DNDS_check_throw_info(mode == InterfaceRoutingMode::CoarseBoxes || mode == InterfaceRoutingMode::AllGatherBoundary,
                                  "unknown interface routing mode");
            DNDS_check_throw_info(points.GetDevice() == R::Device::Host() && queryIds.GetDevice() == R::Device::Host() &&
                                      points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                      queryIds.Layout().Kind() == R::RowLayoutKind::Table && queryIds.Layout().Width() == 1 &&
                                      points.RowCount() == queryIds.RowCount(), "distributed interface requires host point/ID tables"); });
        // Both ranks' branch choice and geometric tolerances affect protocol
        // behavior, so prove agreement before selecting either path.
        real controls[]{real(int(mode)), settings.maxGap, settings.referenceTolerance, settings.projectionTolerance, real(settings.maxIterations)};
        real minimum[5], maximum[5];
        MPI_Allreduce(controls, minimum, 5, DNDS_MPI_REAL, MPI_MIN, mpi.comm);
        MPI_Allreduce(controls, maximum, 5, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        for (int i = 0; i < 5; ++i)
            DNDS_check_throw_info(minimum[i] == maximum[i], "distributed interface settings differ across ranks");
        std::optional<R::Array<real>::ReadLease> pointRead;
        std::optional<R::Array<index>::ReadLease> idRead;
        MPI::LocalStage(mpi, [&]
                        {
            pointRead.emplace(points.Read());
            idRead.emplace(queryIds.Read()); });
        const auto queries = pointRead->View<R::FixedRows<3>>();
        const auto ids = idRead->View<R::FixedRows<1>>();
        MPI::LocalStage(mpi, [&]
                        {
            std::vector<index> keys;
            keys.reserve(size_t(queryIds.RowCount()));
            for (index iQuery = 0; iQuery < queryIds.RowCount(); ++iQuery)
            {
                DNDS_check_throw_info(tPoint(queries(iQuery, 0), queries(iQuery, 1), queries(iQuery, 2)).allFinite(),
                                      "nonfinite distributed interface query");
                keys.push_back(ids(iQuery, 0));
            }
            std::sort(keys.begin(), keys.end());
            DNDS_check_throw_info(std::adjacent_find(keys.begin(), keys.end()) == keys.end(), "duplicate local interface query ID"); });

        // Fixed record schemas, expressed as two SOA tables sharing one route:
        // integer metadata preserves exact IDs; floating geometry never encodes
        // an integer ID as a double. Only selected boundary coordinates travel.
        R::Array<index> elements;
        R::Array<real> geometry;
        SpatialBox coarse;
        MPI::LocalStage(mpi, [&]
                        {
            elements = R::Array<index>(R::RowLayout::Dense(donor.Size(), 9));
            geometry = R::Array<real>(R::RowLayout::Dense(donor.Size(), 24));
            auto eWrite = elements.Write();
            auto gWrite = geometry.Write();
            auto coordinateRead = donor._data->coordinates.Read();
            const auto coords = coordinateRead.View<R::CSRRows>();
            const auto e = eWrite.View<R::FixedRows<9>>();
            const auto g = gWrite.View<R::FixedRows<24>>();
            for (index iElement = 0; iElement < donor.Size(); ++iElement)
            {
                const auto record = donor.Element(iElement);
                const auto global = donor.Cells().LocalToGlobal(record.cell);
                const index values[]{iElement, record.boundary, record.cell, record.originalId, index(record.type),
                                     coords[iElement].size() / 3, donor.Cells().Owner(global),
                                     donor.Cells().OwnerLocalRow(global), mpi.rank};
                std::copy_n(values, 9, e[iElement].data());
                std::fill(g[iElement].begin(), g[iElement].end(), 0);
                std::copy(coords[iElement].begin(), coords[iElement].end(), g[iElement].begin());
                for (int row = 0; row < 3; ++row)
                {
                    g(iElement, 21 + row) = donor._data->pose.translation[row];
                    for (int col = 0; col < 3; ++col)
                        g(iElement, 12 + 3 * row + col) = donor._data->pose.rotation(row, col);
                }
                for (index node = 0; node < coords[iElement].size() / 3; ++node)
                    coarse.Add(donor._data->pose.Apply({coords(iElement, 3 * node), coords(iElement, 3 * node + 1), coords(iElement, 3 * node + 2)}));
            } });
        R::Array<real> candidatePoints;
        R::Array<index> candidateKeys;
        std::vector<int> origins;
        if (mode == InterfaceRoutingMode::AllGatherBoundary)
        {
            std::vector<std::vector<index>> all(size_t(mpi.size));
            for (auto &peer : all)
            {
                peer.resize(size_t(donor.Size()));
                std::iota(peer.begin(), peer.end(), index(0));
            }
            const auto route = R::SparseBatchExchange::Build(context, donor.Size(), std::move(all));
            elements = route.Exchange(elements);
            geometry = route.Exchange(geometry);
            candidatePoints = points;
            candidateKeys = R::Array<index>(R::RowLayout::Dense(points.RowCount(), 2));
            auto write = candidateKeys.Write();
            for (index iQuery = 0; iQuery < points.RowCount(); ++iQuery)
                write.View()(iQuery, 0) = iQuery, write.View()(iQuery, 1) = ids(iQuery, 0);
            origins.assign(size_t(points.RowCount()), mpi.rank);
        }
        else
        {
            std::vector<real> boxes(size_t(mpi.size) * 6);
            real local[6];
            for (int d = 0; d < 3; ++d)
                local[d] = coarse.lower[d], local[d + 3] = coarse.upper[d];
            MPI_Allgather(local, 6, DNDS_MPI_REAL, boxes.data(), 6, DNDS_MPI_REAL, mpi.comm);
            std::vector<std::vector<index>> destinations(size_t(mpi.size));
            R::Array<index> keys(R::RowLayout::Dense(points.RowCount(), 2));
            {
                auto write = keys.Write();
                for (index iQuery = 0; iQuery < points.RowCount(); ++iQuery)
                {
                    write.View()(iQuery, 0) = iQuery, write.View()(iQuery, 1) = ids(iQuery, 0);
                    const tPoint point{queries(iQuery, 0), queries(iQuery, 1), queries(iQuery, 2)};
                    for (int peer = 0; peer < mpi.size; ++peer)
                    {
                        SpatialBox box;
                        for (int d = 0; d < 3; ++d)
                            box.lower[d] = boxes[6 * peer + d], box.upper[d] = boxes[6 * peer + d + 3];
                        if ((box.lower.array() <= box.upper.array()).all() && box.SquaredDistance(point) <= sqr(settings.maxGap))
                            destinations[peer].push_back(iQuery);
                    }
                }
            }
            const auto route = R::SparseBatchExchange::Build(context, points.RowCount(), std::move(destinations));
            candidatePoints = route.Exchange(points);
            candidateKeys = route.Exchange(keys);
            origins.resize(size_t(candidatePoints.RowCount()));
            for (int peer = 0; peer < mpi.size; ++peer)
                std::fill(origins.begin() + route.ReceiveOffsets()[peer], origins.begin() + route.ReceiveOffsets()[peer + 1], peer);
        }

        struct Proposal
        {
            std::array<index, 10> keys;
            std::array<real, 10> values;
            int origin;
        };
        std::vector<Proposal> proposals;
        MPI::LocalStage(mpi, [&]
                        {
            auto eRead = elements.Read(), keyRead = candidateKeys.Read();
            auto gRead = geometry.Read(), qRead = candidatePoints.Read();
            const auto e = eRead.View<R::FixedRows<9>>();
            const auto keys = keyRead.View<R::FixedRows<2>>();
            const auto g = gRead.View<R::FixedRows<24>>();
            const auto q = qRead.View<R::FixedRows<3>>();
            std::vector<SpatialBox> bounds(size_t(elements.RowCount()));
            for (index iElement = 0; iElement < elements.RowCount(); ++iElement)
                for (index node = 0; node < e(iElement, 5); ++node)
                {
                    tPoint world;
                    for (int row = 0; row < 3; ++row)
                    {
                        world[row] = g(iElement, 21 + row);
                        for (int col = 0; col < 3; ++col)
                            world[row] += g(iElement, 12 + 3 * row + col) * g(iElement, 3 * node + col);
                    }
                    bounds[iElement].Add(world);
                }
            BoundingBoxTree tree(bounds);
            tSmallCoords coordinates;
            for (DNDS::index iQuery = 0; iQuery < candidatePoints.RowCount(); ++iQuery)
            {
                const tPoint world{q(iQuery, 0), q(iQuery, 1), q(iQuery, 2)};
                tree.Query(world, settings.maxGap, [&](DNDS::index iElement)
                {
                    RigidPose pose;
                    for (int row = 0; row < 3; ++row)
                    {
                        pose.translation[row] = g(iElement, 21 + row);
                        for (int col = 0; col < 3; ++col)
                            pose.rotation(row, col) = g(iElement, 12 + 3 * row + col);
                    }
                    coordinates.resize(3, e(iElement, 5));
                    for (DNDS::index node = 0; node < coordinates.cols(); ++node)
                        coordinates.col(node) = tPoint{g(iElement, 3 * node), g(iElement, 3 * node + 1), g(iElement, 3 * node + 2)};
                    tPoint reference, donorPoint;
                    const auto type = Elem::ElemType(e(iElement, 4));
                    if (!Project(type, coordinates, pose.Inverse(world), settings, reference, donorPoint))
                        return;
                    const auto projected = pose.Apply(donorPoint);
                    const bool interior = Inside(type, reference, -std::max(settings.referenceTolerance, 32 * std::numeric_limits<real>::epsilon()));
                    proposals.push_back({{keys(iQuery, 0), keys(iQuery, 1), e(iElement, 0), e(iElement, 1), e(iElement, 2),
                                          e(iElement, 3), e(iElement, 8), e(iElement, 6), e(iElement, 7), index(interior)},
                                         {reference[0], reference[1], reference[2], projected[0], projected[1], projected[2],
                                          donorPoint[0], donorPoint[1], donorPoint[2], (world - projected).norm()}, origins[iQuery]});
                });
            } });
        R::Array<index> proposalKeys(R::RowLayout::Dense(proposals.size(), 10));
        R::Array<real> proposalValues(R::RowLayout::Dense(proposals.size(), 10));
        std::vector<std::vector<index>> replies(size_t(mpi.size));
        {
            auto keyWrite = proposalKeys.Write();
            auto valueWrite = proposalValues.Write();
            for (index iProposal = 0; iProposal < index(proposals.size()); ++iProposal)
            {
                std::copy(proposals[iProposal].keys.begin(), proposals[iProposal].keys.end(), keyWrite.View()[iProposal].begin());
                std::copy(proposals[iProposal].values.begin(), proposals[iProposal].values.end(), valueWrite.View()[iProposal].begin());
                replies[proposals[iProposal].origin].push_back(iProposal);
            }
        }
        const auto replyRoute = R::SparseBatchExchange::Build(context, index(proposals.size()), std::move(replies));
        auto returnedKeys = replyRoute.Exchange(proposalKeys);
        auto returnedValues = replyRoute.Exchange(proposalValues);
        auto data = std::make_shared<Detail::DistributedInterfaceMapData>(donor);
        MPI::LocalStage(mpi, [&]
                        {
            data->matches.resize(size_t(points.RowCount()));
            std::vector<bool> found(size_t(points.RowCount()), false);
            std::vector<index> interiors(size_t(points.RowCount()), 0);
            std::vector<real> closest(size_t(points.RowCount()), std::numeric_limits<real>::infinity());
            auto keyRead = returnedKeys.Read();
            auto valueRead = returnedValues.Read();
            const auto keys = keyRead.View<R::FixedRows<10>>();
            const auto values = valueRead.View<R::FixedRows<10>>();
            // Determine the nearest distance first. Counting interior matches
            // independently of the selected seam ID avoids traversal-order
            // dependence when a seam proposal precedes two interior proposals.
            for (index iProposal = 0; iProposal < returnedKeys.RowCount(); ++iProposal)
            {
                const index iQuery = keys(iProposal, 0);
                DNDS_check_throw_info(iQuery >= 0 && iQuery < points.RowCount() && keys(iProposal, 1) == ids(iQuery, 0), "interface proposal/query key mismatch");
                closest[iQuery] = std::min(closest[iQuery], values(iProposal, 9));
            }
            for (index iProposal = 0; iProposal < returnedKeys.RowCount(); ++iProposal)
            {
                const index iQuery = keys(iProposal, 0);
                DNDS_check_throw_info(iQuery >= 0 && iQuery < points.RowCount() && keys(iProposal, 1) == ids(iQuery, 0), "interface proposal/query key mismatch");
                auto &best = data->matches[iQuery];
                if (values(iProposal, 9) > closest[iQuery] + settings.projectionTolerance)
                    continue;
                interiors[iQuery] += keys(iProposal, 9) != 0;
                DNDS_check_throw_info(interiors[iQuery] <= 1, "ambiguous distributed interface interiors");
                if (!found[iQuery] || keys(iProposal, 5) < best.geometry.donorOriginalId)
                {
                    best = {{ids(iQuery, 0), keys(iProposal, 2), keys(iProposal, 3), keys(iProposal, 4), keys(iProposal, 5),
                             {values(iProposal, 0), values(iProposal, 1), values(iProposal, 2)},
                             {values(iProposal, 3), values(iProposal, 4), values(iProposal, 5)},
                             {values(iProposal, 6), values(iProposal, 7), values(iProposal, 8)}, values(iProposal, 9)},
                            int(keys(iProposal, 6)), int(keys(iProposal, 7)), keys(iProposal, 8)};
                    found[iQuery] = true;
                }
            }
            for (index iQuery = 0; iQuery < points.RowCount(); ++iQuery)
                DNDS_check_throw_info(found[iQuery], "uncovered distributed interface query " + std::to_string(ids(iQuery, 0))); });
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
        return _data->matches[iQuery];
    }
}
