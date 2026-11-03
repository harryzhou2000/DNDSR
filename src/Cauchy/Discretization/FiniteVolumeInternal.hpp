#pragma once

// Private shared preparation data. Public callers use FiniteVolume and explicit
// Workspace/AuxiliaryValues owners; this header is not a mesh or solver API.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    struct Sample
    {
        index cell, other, point;
        int side;
        int zone;
        Geom::tPoint normal, ownPoint, otherPoint;
        Geom::tGPoint rotation;
        BoundaryCondition boundary;
        index auxiliaryFace = UnInitIndex;
        real viscousDistance = 0;
        index face = UnInitIndex;
        index boundaryPoint = UnInitIndex;
    };
    template <class View>
    FlowState Load(const View &input, index iCell)
    {
        FlowState state;
        for (int j = 0; j < 5; ++j)
            state[j] = input(iCell, j);
        return state;
    }
    inline FlowState RotateState(FlowState state, const Geom::tGPoint &rotation)
    {
        const Geom::tPoint momentum{state[1], state[2], state[3]};
        const Geom::tPoint rotated = rotation * momentum;
        for (int d = 0; d < 3; ++d)
            state[1 + d] = rotated[d];
        return state;
    }
    inline FlowState BoundaryFaceValue(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                       real time, const RigidFrame &frame, RHSScalarView incoming)
    {
        if (sample.boundary.kind == BoundaryKind::Interface)
            return Load(incoming, sample.boundaryPoint);
        return sample.boundary.FaceValue(gas, inside, {sample.normal[0], sample.normal[1], sample.normal[2]},
                                         {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time, &frame);
    }
    inline FlowState BoundaryExterior(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                      real time, const RigidFrame &frame, RHSScalarView incoming)
    {
        if (sample.boundary.kind == BoundaryKind::Interface)
            return Load(incoming, sample.boundaryPoint);
        return sample.boundary.Exterior(gas, inside, {sample.normal[0], sample.normal[1], sample.normal[2]},
                                        {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time, &frame);
    }
}

namespace DNDS::Cauchy
{
    namespace Detail
    {
        struct BoundaryPointLayout
        {
            std::weak_ptr<Geom::UnstructuredMesh> mesh;
            Runtime::VersionStamp stamp;
            std::vector<BoundaryPoint> points;
        };
    }
    struct FiniteVolume::Data
    {
        using Sample = FiniteVolumeDetail::Sample;
        ssp<Geom::UnstructuredMesh> mesh;
        DomainSettings settings;
        EquationSystem equation;
        RigidMotion motion;
        Riemann::RiemannFlux flux;
        BoundarySystem boundary;
        CFV::FiniteVolumePlan metrics;
        CFV::VariationalReconstructionPlan reconstruction;
        CFV::GradientPlan gradient;
        Geom::EntityWorkPartition partition;
        std::vector<index> offsets;
        std::vector<Sample> samples;
        std::vector<index> boundaryEntries;
        std::shared_ptr<Detail::BoundaryPointLayout> boundaryLayout;
        std::vector<index> gradientBoundaryPoints;
        bool hasInterfaces = false;
        std::vector<index> stencilFaces;
        std::vector<index> faceEntries;
        std::vector<index> auxiliaryCoreFaces, auxiliaryDependentFaces;
        std::vector<index> auxiliaryCellOffsets, auxiliaryCellFaces;
        std::vector<real> auxiliaryFaceArea;
        std::optional<CFV::PolynomialSamplingPlan> positivitySamples;
        std::shared_ptr<const LimitedVRGeometry> limitedVRGeometry;
        std::vector<index> positivityOffsets;
        ThermalBounds thermalBounds;
        Data(ssp<Geom::UnstructuredMesh> source, DomainSettings controls, EquationSystem physics)
            : mesh(std::move(source)), settings(std::move(controls)), equation(std::move(physics)), motion(settings.motion, settings.mesh.dimension), flux(Riemann::RiemannFlux::Resolve(settings.numerics.flux)),
              boundary(settings.boundaries, equation.Gas()), metrics(CFV::FiniteVolumePlan::ConstructMetrics(mesh, settings.numerics.cfv.intOrder)),
              reconstruction(CFV::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings.numerics.cfv)),
              gradient(CFV::GradientPlan::Prepare(*mesh, metrics, settings.numerics.gradient == "greenGauss" ? CFV::GradientMethod::GreenGauss : settings.numerics.gradient == "ggmp" ? CFV::GradientMethod::GGMP
                                                                                                                                                                                      : CFV::GradientMethod::LeastSquares)),
              partition(metrics.CellWorkPartition()), thermalBounds(equation.PrepareThermalBounds(settings.numerics.densityFloor, settings.numerics.temperatureFloor, settings.numerics.pressureFloor)) {}
    };
}
