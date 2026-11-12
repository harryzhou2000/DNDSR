#include "Cauchy/Motion/RigidMotion.hpp"
#include <Eigen/Geometry>

namespace DNDS::Cauchy
{
    namespace
    {
        Geom::tPoint Point(const Vector3 &v) { return {v[0], v[1], v[2]}; }
        Vector3 Vector(const Geom::tPoint &v) { return {v[0], v[1], v[2]}; }
    }
    void RigidMotionSettings::Validate(int dimension) const
    {
        DNDS_check_throw_info(dimension == 2 || dimension == 3, "rigid motion requires a 2D/3D domain");
        for (const auto &v : {angularVelocity, translationVelocity, center})
            for (real component : v)
                DNDS_check_throw_info(std::isfinite(component), "nonfinite rigid-motion parameter");
        DNDS_check_throw_info(dimension != 2 || (angularVelocity[0] == 0 && angularVelocity[1] == 0 && translationVelocity[2] == 0),
                              "planar rigid motion requires z-axis rotation and in-plane translation");
    }
    RigidMotion::RigidMotion(RigidMotionSettings settings, int dimension) : _settings(settings) { _settings.Validate(dimension); }
    RigidFrame RigidMotion::At(real time) const
    {
        DNDS_check_throw_info(std::isfinite(time), "nonfinite rigid-frame stage time");
        const auto omega = Point(_settings.angularVelocity), velocity = Point(_settings.translationVelocity), center = Point(_settings.center);
        RigidFrame frame;
        frame.angularVelocity = _settings.angularVelocity;
        frame.center = _settings.center;
        const real speed = omega.norm();
        DNDS_check_throw_info(std::isfinite(speed * time), "rigid rotation angle overflow");
        frame.moving = speed != 0 || velocity.squaredNorm() != 0;
        if (speed != 0)
            frame.pose.rotation = Eigen::AngleAxis<real>(speed * time, omega / speed).toRotationMatrix();
        frame.pose.translation = center + velocity * time - frame.pose.rotation * center;
        frame.centerVelocity = Vector(frame.pose.rotation.transpose() * velocity);
        frame.pose.Validate();
        return frame;
    }
    Vector3 RigidFrame::GridVelocity(const Vector3 &point) const
    {
        if (!moving)
            return {};
        return Vector(Point(centerVelocity) + Point(angularVelocity).cross(Point(point) - Point(center)));
    }
    RigidFrameParameters RigidFrame::KernelParameters() const
    {
        RigidFrameParameters result;
        for (int j = 0; j < 3; ++j)
        {
            for (int i = 0; i < 3; ++i)
                result.rotation[i + 3 * j] = pose.rotation(i, j);
            result.translation[j] = pose.translation[j];
        }
        result.angularVelocity = angularVelocity;
        result.centerVelocity = centerVelocity;
        result.center = center;
        result.moving = moving;
        return result;
    }
    Vector3 RigidFrame::WorldVector(const Vector3 &v) const { return moving ? Vector(pose.rotation * Point(v)) : v; }
    Vector3 RigidFrame::LocalVector(const Vector3 &v) const { return moving ? Vector(pose.rotation.transpose() * Point(v)) : v; }
    Vector3 RigidFrame::WorldPoint(const Vector3 &p) const { return moving ? Vector(pose.Apply(Point(p))) : p; }
    FlowState RigidFrame::WorldState(FlowState u) const
    {
        const auto m = WorldVector({u[1], u[2], u[3]});
        for (int d = 0; d < 3; ++d)
            u[1 + d] = m[d];
        return u;
    }
    FlowState RigidFrame::LocalState(FlowState u) const
    {
        const auto m = LocalVector({u[1], u[2], u[3]});
        for (int d = 0; d < 3; ++d)
            u[1 + d] = m[d];
        return u;
    }
    FlowState RigidFrame::RotationSource(const FlowState &u) const
    {
        if (!moving)
            return {};
        const Geom::tPoint s = -Point(angularVelocity).cross(Geom::tPoint{u[1], u[2], u[3]});
        return {0, s[0], s[1], s[2], 0};
    }
    std::array<real, 25> RigidFrame::NegativeSourceJacobian() const
    {
        std::array<real, 25> matrix{};
        const auto &w = angularVelocity;
        matrix[1 + 5 * 2] = -w[2];
        matrix[1 + 5 * 3] = w[1];
        matrix[2 + 5 * 1] = w[2];
        matrix[2 + 5 * 3] = -w[0];
        matrix[3 + 5 * 1] = -w[1];
        matrix[3 + 5 * 2] = w[0];
        return matrix;
    }
}
