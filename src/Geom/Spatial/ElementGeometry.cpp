#include "ElementGeometry.hpp"

namespace DNDS::Geom
{
    void RigidPose::Validate() const
    {
        DNDS_check_throw_info(rotation.allFinite() && translation.allFinite() &&
                                  (rotation.transpose() * rotation).isApprox(tGPoint::Identity(), 1e-12) &&
                                  std::abs(rotation.determinant() - 1) < 1e-12,
                              "surface pose must be a proper rigid transform");
    }
    void PointProjectionSettings::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(maxGap) && maxGap >= 0 && std::isfinite(referenceTolerance) && referenceTolerance >= 0 &&
                                  std::isfinite(projectionTolerance) && projectionTolerance > 0 && maxIterations > 0,
                              "invalid point projection tolerances");
    }
    ElementGeometryView::ElementGeometryView(Elem::ElemType type, const real *coordinates, index nodes)
        : _type(type), _coordinates(coordinates), _nodes(nodes)
    {
        DNDS_check_throw_info((type == Elem::Line2 || type == Elem::Tri3 || type == Elem::Quad4) &&
                                  coordinates && nodes == Elem::Element{type}.GetNumNodes(),
                              "unsupported surface element or coordinate shape");
    }
    bool ElementGeometryView::ContainsReference(const tPoint &reference, real tolerance) const
    {
        DNDS_check_throw_info(reference.allFinite() && std::isfinite(tolerance), "invalid reference containment input");
        if (_type == Elem::Tri3)
            return reference[0] >= -tolerance && reference[1] >= -tolerance && reference[0] + reference[1] <= 1 + tolerance;
        return std::abs(reference[0]) <= 1 + tolerance && (_type == Elem::Line2 || std::abs(reference[1]) <= 1 + tolerance);
    }
    tPoint ElementGeometryView::Point(const tPoint &reference) const
    {
        DNDS_check_throw_info(reference.allFinite(), "nonfinite surface reference point");
        Elem::tD01Nj shape;
        Elem::Element{_type}.GetD01Nj(reference, shape);
        return Elem::PPhysicsCoordD01Nj(Coordinates(), shape);
    }
    tGPoint ElementGeometryView::Jacobian(const tPoint &reference) const
    {
        DNDS_check_throw_info(reference.allFinite(), "nonfinite surface reference point");
        Elem::tD01Nj shape;
        Elem::Element{_type}.GetD01Nj(reference, shape);
        return Elem::ShapeJacobianCoordD01Nj(Coordinates(), shape);
    }
    SpatialBox ElementGeometryView::Bounds(RigidPose pose, real referenceTolerance) const
    {
        pose.Validate();
        DNDS_check_throw_info(std::isfinite(referenceTolerance) && referenceTolerance >= 0 && Coordinates().allFinite(), "invalid surface bounds input");
        SpatialBox box;
        for (index iNode = 0; iNode < _nodes; ++iNode)
            box.Add(pose.Apply(Coordinates().col(iNode)));
        const real padding = ReferencePadding((box.upper - box.lower).norm(), referenceTolerance);
        box.lower.array() -= padding;
        box.upper.array() += padding;
        return box;
    }
    real ElementGeometryView::ReferencePadding(real diameter, real referenceTolerance)
    {
        const real padding = 4 * referenceTolerance * (1 + referenceTolerance) * diameter;
        DNDS_check_throw_info(std::isfinite(padding) && diameter >= 0 && referenceTolerance >= 0, "surface reference envelope is not representable");
        return padding;
    }
    void ElementGeometryView::Validate() const
    {
        DNDS_check_throw_info(Coordinates().allFinite(), "nonfinite surface coordinates");
        // P1/simple bilinear validation at corners and center. This rejects
        // sampled degeneracy, not every possible fold of an arbitrary quad.
        const std::vector<tPoint> samples = _type == Elem::Tri3
                                                ? std::vector<tPoint>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1. / 3, 1. / 3, 0}}
                                            : _type == Elem::Line2
                                                ? std::vector<tPoint>{{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}}
                                                : std::vector<tPoint>{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 0}};
        for (const auto &sample : samples)
        {
            const auto jac = Jacobian(sample);
            const real measure = _type == Elem::Line2 ? jac.col(0).norm() : jac.col(0).cross(jac.col(1)).norm();
            DNDS_check_throw_info(std::isfinite(measure) && measure > 0, "degenerate surface element");
        }
    }
    PointProjection ElementGeometryView::Project(const tPoint &query, PointProjectionSettings settings) const
    {
        settings.Validate();
        DNDS_check_throw_info(query.allFinite() && Coordinates().allFinite(), "nonfinite point projection input");
        PointProjection result;
        if (_type == Elem::Tri3)
            result.reference = {1. / 3, 1. / 3, 0};
        bool converged = false;
        for (int iteration = 0; iteration < settings.maxIterations; ++iteration)
        {
            result.iterations = iteration + 1;
            result.point = Point(result.reference);
            const auto jac = Jacobian(result.reference);
            const tPoint residual = query - result.point;
            tPoint step = tPoint::Zero();
            if (_type == Elem::Line2)
            {
                const real length2 = jac.col(0).squaredNorm();
                if (!(length2 > 0) || !std::isfinite(length2))
                {
                    result.status = PointProjectionStatus::Degenerate;
                    return result;
                }
                step[0] = jac.col(0).dot(residual) / length2;
            }
            else
            {
                const Eigen::Matrix<real, 3, 2> tangent = jac.leftCols<2>();
                const Eigen::Matrix2d gram = tangent.transpose() * tangent;
                if (!(gram.determinant() > 0) || !gram.allFinite())
                {
                    result.status = PointProjectionStatus::Degenerate;
                    return result;
                }
                step.head<2>() = gram.ldlt().solve(tangent.transpose() * residual);
            }
            if (!step.allFinite())
                return result;
            result.reference += step;
            if (!result.reference.allFinite())
                return result;
            if (step.norm() <= settings.projectionTolerance)
            {
                converged = true;
                break;
            }
        }
        result.point = Point(result.reference);
        const auto jac = Jacobian(result.reference);
        const int dim = _type == Elem::Line2 ? 1 : 2;
        result.distance = (query - result.point).norm();
        result.stationarity = (jac.leftCols(dim).transpose() * (query - result.point)).norm();
        if (dim == 2)
        {
            const tPoint normal = jac.col(0).cross(jac.col(1));
            if (!(normal.norm() > 0) || !normal.allFinite())
            {
                result.status = PointProjectionStatus::Degenerate;
                return result;
            }
            result.normal = normal.normalized();
        }
        if (!converged || !result.point.allFinite() || !std::isfinite(result.distance) || !std::isfinite(result.stationarity))
            return result;
        if (!ContainsReference(result.reference, settings.referenceTolerance))
            result.status = PointProjectionStatus::OutsideReference;
        else if (result.distance > settings.maxGap)
            result.status = PointProjectionStatus::GapExceeded;
        else
            result.status = PointProjectionStatus::Success;
        return result;
    }
}
