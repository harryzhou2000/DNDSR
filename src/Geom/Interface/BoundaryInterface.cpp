#include "BoundaryInterface.hpp"
#include "Geom/Quadrature.hpp"
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
}
