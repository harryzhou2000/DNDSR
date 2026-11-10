#pragma once

// Private shared preparation data. Public callers use FiniteVolume and explicit
// Workspace/AuxiliaryValues owners; this header is not a mesh or solver API.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Communication/CommunicationWorker.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Immutable point-flux dispatch resolved with the FV plan. Flow5 and the
    /// extensible equation adapter consume the same policy; numerical kernels
    /// retain their specialized storage and arithmetic.
    struct FluxKernelPolicy
    {
        Riemann::Requirements required;
        Riemann::FluxSettings settings;
        bool rusanov = false;
        bool transverseSignal = false;
        bool pressureStencil = false;
        bool noRsOnWall = false;
        bool noGRPOnWall = false;
    };
    inline FluxKernelPolicy ResolveFluxKernelPolicy(const DomainSettings &settings, const Riemann::RiemannFlux &flux)
    {
        return {flux.RequiredInputs(), settings.numerics.riemann.KernelSettings(),
                flux.Kind() == Riemann::Solver::Rusanov, flux.UsesTransverseSignal(), flux.UsesPressureStencil(),
                settings.numerics.noRsOnWall, settings.numerics.noGRPOnWall};
    }
    inline CFV::ReconstructionCapabilities ReconstructionSupport(const DomainSettings &settings)
    {
        const auto &numerics = settings.numerics;
        const bool limitedVR = numerics.limiter == "LVR" &&
                               (numerics.limitedVR.alphaMax > 0 || numerics.limitedVR.contactAlphaMax > 0);
        auto support = CFV::ReconstructionCapabilities::ForSweeps(numerics.limiter == "WBAP" || limitedVR);
        support.sweeps = numerics.reconstruction == "variational";
        return support;
    }
    inline Runtime::RowLayout FieldLayout(const Runtime::RowLayout &prototype, index rows, index width)
    {
        if (prototype.Kind() == Runtime::RowLayoutKind::ColMajorTable)
            return Runtime::RowLayout::ColMajor(rows, width, prototype.PaddingOptions());
        if (prototype.Kind() == Runtime::RowLayoutKind::TiledColMajorTable)
            return Runtime::RowLayout::TiledColMajor(rows, width, prototype.Stride());
        return Runtime::RowLayout::Dense(rows, width);
    }
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
        /// Dense column slot in this cell's canonical Jacobian row. Filled
        /// once after JacobianColumns() is constructed; UnInitIndex for
        /// diagonal/boundary samples and non-Jacobian quadrature samples.
        index jacobianColumn = UnInitIndex;
        /// Row in the incoming packet used by this sample: RHS flow for
        /// quadrature samples, faceFlow for centered Jacobian samples.
        index boundaryPoint = UnInitIndex;
        /// Dense RHSLocalDiagnostics force slot, resolved once during setup.
        /// Non-wall and interior samples keep UnInitIndex.
        index forceSlot = UnInitIndex;
        /// Exact setup-time frame test. Ordinary and translated-periodic faces
        /// keep components unchanged; even a small nonidentity rotation remains
        /// active. Default true keeps hand-built samples conservative.
        bool changesFrame = true;
    };
    template <class View>
    FlowState Load(const View &input, index iCell)
    {
        FlowState state;
        const auto row = input[iCell];
        for (int j = 0; j < 5; ++j)
            state[j] = row[j];
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
    /// Physical displacement from a cell average's barycenter, in that cell's
    /// own axes. The neighbor center already carries the periodic coordinate
    /// image; only vectors need the inverse frame map. No Taylor basis here.
    inline Geom::tPoint DirectFVDisplacement(const CFV::FiniteVolumeHostView &metric, const Sample &sample, bool other)
    {
        if (!other)
            return sample.ownPoint - metric.GetCellBary(sample.cell);
        const Geom::tPoint delta = sample.ownPoint - sample.otherPoint;
        return (sample.changesFrame ? Geom::tPoint(sample.rotation.transpose() * delta) : delta) +
               metric.GetCellCent(sample.other) - metric.GetCellBary(sample.other);
    }
    /// Canonical rank-owned cell-face-cell columns, retaining remote columns
    /// and folding self-periodic incidences into the eventual diagonal.
    inline std::vector<std::vector<index>> JacobianColumns(const std::vector<index> &offsets, const std::vector<Sample> &samples)
    {
        std::vector<std::vector<index>> columns(offsets.size() - 1);
        for (index iCell = 0; iCell < index(columns.size()); ++iCell)
        {
            auto &row = columns[iCell];
            for (index entry = offsets[iCell]; entry < offsets[iCell + 1]; ++entry)
            {
                const auto other = samples[entry].other;
                if (other != UnInitIndex && other != iCell)
                    row.push_back(other);
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        return columns;
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
    /// GradientPlan consumes a face trace; Euler's GG/LS construction consumes
    /// a mirrored exterior at twice the wall distance. This conservative mean
    /// is the matching trace, including its energy/species convention. Incoming
    /// interface packets already are prescribed point values, not ghost states.
    inline FlowState BoundaryGradientValue(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                           real time, const RigidFrame &frame, RHSScalarView incoming)
    {
        auto result = BoundaryExterior(sample, gas, inside, time, frame, incoming);
        if (sample.boundary.kind != BoundaryKind::Interface)
            for (int j = 0; j < 5; ++j)
                result[j] = .5 * (inside[j] + result[j]);
        return result;
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
            std::vector<BoundaryPoint> faces;
        };
    }
    struct FiniteVolume::Data
    {
        using Sample = FiniteVolumeDetail::Sample;
        ssp<Geom::UnstructuredMesh> mesh;
        DomainSettings settings;
        std::shared_ptr<Runtime::CommunicationWorker> communicationWorker;
        EquationSystem equation;
        RigidMotion motion;
        Riemann::RiemannFlux flux;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        BoundarySystem boundary;
        CFV::FiniteVolumePlan metrics;
        CFV::VariationalReconstructionPlan reconstruction;
        CFV::GradientPlan gradient;
        Geom::EntityWorkPartition partition;
        std::vector<index> offsets;
        std::vector<Sample> samples;
        // Coarse exact dispatch avoids growing the frequently traversed Sample
        // structure. One rotated image retains the general Jacobian path for
        // the whole plan; ordinary/translated meshes skip identity products.
        bool hasRotatedFrames = false;
        /// One oriented mean-geometry face per cell incidence. Approximate
        /// Jacobians never traverse the RHS quadrature samples above.
        std::vector<index> jacobianOffsets;
        std::vector<Sample> jacobianSamples;
        std::shared_ptr<const std::vector<std::vector<index>>> jacobianColumns;
        std::vector<index> boundaryEntries;
        std::vector<int> boundaryZones;
        std::vector<int> forceZones;
        std::map<int, index> forceSlots;
        std::shared_ptr<Detail::BoundaryPointLayout> boundaryLayout;
        std::vector<index> gradientBoundaryPoints;
        bool hasInterfaces = false;
        std::vector<index> stencilFaces;
        std::vector<index> faceEntries;
        std::vector<index> auxiliaryCoreFaces, auxiliaryDependentFaces;
        // Unique faces required by each cell-output region. A face shared
        // across the regions may occur in both; each region remains callable
        // independently, without a hidden prior-core publication contract.
        std::vector<index> rhsCoreFaces, rhsDependentFaces;
        std::vector<index> auxiliaryCellOffsets, auxiliaryCellFaces;
        std::vector<real> auxiliaryFaceArea;
        std::optional<CFV::PolynomialSamplingPlan> positivitySamples;
        std::shared_ptr<const LimitedVRGeometry> limitedVRGeometry;
        std::vector<index> positivityOffsets;
        std::optional<ThermalBounds> thermalBounds;
        std::optional<WallDistanceSamples> wallDistance;
        Runtime::Array<real> ransCellLength;
        Runtime::TransferBinding BindHalo(CFV::FVField &field, const char *channel = "cauchy-field") const
        {
            return metrics.BindHalo(field, HaloTransferOptions(channel));
        }
        Algebra::DofHalo<> BindTrackedHalo(CFV::FVField &field, const char *channel = "cauchy-mean") const
        {
            return metrics.BindTrackedHalo(field, HaloTransferOptions(channel));
        }
        Runtime::TransferOptions HaloTransferOptions(const char *channel) const
        {
            Runtime::TransferOptions options;
            options.communicationWorker = communicationWorker;
            options.channel = channel;
            DNDS_check_throw_info(settings.execution.transferPreflight == "collective" || settings.execution.transferPreflight == "abortOnFailure", "invalid Cauchy transfer preflight policy");
            if (!settings.implicit.collectiveChecks || settings.execution.transferPreflight == "abortOnFailure")
                options.preflight = Runtime::TransferPreflight::AbortOnFailure;
            return options;
        }
        static CFV::FiniteVolumeMetricOptions MetricOptions(const DomainSettings &settings)
        {
            CFV::FiniteVolumeMetricOptions options;
            options.integrationOrder = settings.numerics.cfv.intOrder;
            options.faceIntegrationOrder = settings.numerics.RHSIntegrationOrder();
            options.collectivePreflight = settings.implicit.collectiveChecks;
            return options;
        }
        const ThermalBounds &PerfectGasBounds() const
        {
            DNDS_check_throw_info(thermalBounds, "flow-only thermal policy is unavailable for this equation");
            return *thermalBounds;
        }
        Data(ssp<Geom::UnstructuredMesh> source, DomainSettings controls, EquationSystem physics)
            : mesh(std::move(source)), settings(std::move(controls)), equation(std::move(physics)), motion(settings.motion, settings.mesh.dimension), flux(Riemann::RiemannFlux::Resolve(settings.numerics.flux)),
              fluxPolicy(FiniteVolumeDetail::ResolveFluxKernelPolicy(settings, flux)),
              boundary(settings.boundaries, equation), metrics(CFV::FiniteVolumePlan::ConstructMetrics(mesh, MetricOptions(settings))),
              reconstruction(CFV::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings.numerics.cfv, FiniteVolumeDetail::ReconstructionSupport(settings))),
              gradient(CFV::GradientPlan::Prepare(*mesh, metrics, settings.numerics.gradient == "greenGauss" ? CFV::GradientMethod::GreenGauss : settings.numerics.gradient == "ggmp" ? CFV::GradientMethod::GGMP
                                                                                                                                                                                      : CFV::GradientMethod::LeastSquares)),
              partition(metrics.CellWorkPartition())
        {
            if (settings.execution.communicationExecution == "worker")
                communicationWorker = std::make_shared<Runtime::CommunicationWorker>();
            if (!equation.HasMixture())
                thermalBounds = equation.PrepareThermalBounds(settings.numerics.densityFloor, settings.numerics.temperatureFloor, settings.numerics.pressureFloor);
        }
    };
    /// Private owned-quadrature PP composition. Physical coefficient fields
    /// keep their layouts; numeric policy, samples and scratch are independent
    /// of nonlinear/time ownership. No communication is hidden in Apply.
    struct FiniteVolume::PositivityWorkspace
    {
        struct State;
        std::shared_ptr<State> state;
        PositivityWorkspace(std::shared_ptr<const Data> owner, const EquationDofs &prototype,
                            const EquationDofs &coefficients, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers);
        void Apply(const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh);
        std::pair<index, real> LocalStatistics() const;
    };
}
