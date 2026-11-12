#pragma once

#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Discretization/BoundaryTraceGeometry.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
// Private shared preparation data. Public callers use FiniteVolume and explicit
// Workspace/AuxiliaryValues owners; this header is not a mesh or solver API.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Communication/CommunicationWorker.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Immutable point-flux dispatch resolved with the FV plan. Flow5 and the
    /// extensible equation adapter consume the same policy; numerical kernels
    /// retain their specialized storage and arithmetic.
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

}

namespace DNDS::Cauchy
{
    namespace Detail
    {
        struct BoundaryPointLayout
        {
            std::weak_ptr<Geom::UnstructuredMesh> mesh;
            Runtime::VersionStamp stamp;
            Runtime::Array<BoundaryPoint> points{Runtime::RowLayout::Dense(0, 1)};
            Runtime::Array<BoundaryPoint> faces{Runtime::RowLayout::Dense(0, 1)};
        };
    }
    struct FiniteVolume::Data
    {
        ssp<Geom::UnstructuredMesh> mesh;
        DomainSettings settings;
        std::shared_ptr<Runtime::CommunicationWorker> communicationWorker;
        EquationSystem equation;
        BoundaryPointOperator boundaryPoint;
        RigidMotion motion;
        Riemann::RiemannFlux flux;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        BoundarySystem boundary;
        CFV::FiniteVolumePlan metrics;
        /// Present only for the variational family. Constant/P0 and explicit
        /// physical-gradient P1 own no Taylor basis, VR matrices or modal plan.
        std::optional<CFV::VariationalReconstructionPlan> reconstruction;
        CFV::GradientPlan gradient;
        Geom::EntityWorkPartition partition;
        FiniteVolumeStencilStorage stencil;
        // Coarse exact dispatch avoids growing the frequently traversed Sample
        // structure. One rotated image retains the general Jacobian path for
        // the whole plan; ordinary/translated meshes skip identity products.
        bool hasRotatedFrames = false;
        /// One oriented mean-geometry face per cell incidence. Approximate
        /// Jacobians never traverse the RHS quadrature samples above.
        std::shared_ptr<const LinearPatternStorage> jacobianColumns;
        std::vector<int> boundaryZones;
        std::vector<int> forceZones;
        std::map<int, index> forceSlots;
        std::shared_ptr<Detail::BoundaryPointLayout> boundaryLayout;
        /// Mesh-sized inputs declared by the boundary operator; independent of
        /// the host setup vectors used by the still-pending RHS extraction.
        Runtime::Array<BoundaryTraceGeometry> boundaryTraceGeometry;
        Runtime::Array<index> gradientBoundaryRows;
        std::vector<BoundaryCondition> boundaryConditions;
        /// Domain-wide coupling participation. Every rank uses this to retain
        /// the collective interface publication protocol, including ranks
        /// with no locally owned interface quadrature.
        bool hasInterfaces = false;
        /// Whether this rank's RHS actually consumes an incoming interface
        /// row. A published zero-row packet is valid and must not be mistaken
        /// for a missing numeric payload on interior-only partitions.
        bool hasLocalInterfaceRows = false;
        // Unique faces required by each cell-output region. A face shared
        // across the regions may occur in both; each region remains callable
        // independently, without a hidden prior-core publication contract.
        std::optional<CFV::PolynomialSamplingPlan> positivitySamples;
        std::shared_ptr<const LimitedVRGeometry> limitedVRGeometry;
        /// Cell-center displacements for direct-P1 trace/volume PP checks.
        /// Shares positivityOffsets with polynomial sampling but owns no basis.
        /// Small fixed-flow leaves retain their five-component affine bounds;
        /// orchestration and complete equation checks use defaultAdmissibility.
        std::optional<ThermalBounds> fixedFlowThermalBounds;
        std::optional<AdmissibilityPolicy> defaultAdmissibility;
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
        const ThermalBounds &FixedFlowThermalBounds() const
        {
            DNDS_check_throw_info(
                fixedFlowThermalBounds,
                "fixed-flow thermal policy is unavailable for this equation");
            return *fixedFlowThermalBounds;
        }
        Data(ssp<Geom::UnstructuredMesh> source, DomainSettings controls, EquationSystem physics)
            : mesh(std::move(source)), settings(std::move(controls)), equation(std::move(physics)), boundaryPoint(equation), motion(settings.motion, settings.mesh.dimension), flux(Riemann::RiemannFlux::Resolve(settings.numerics.flux)),
              fluxPolicy(FiniteVolumeDetail::ResolveFluxKernelPolicy(settings, flux)),
              boundary(settings.boundaries, equation), metrics(CFV::FiniteVolumePlan::ConstructMetrics(mesh, MetricOptions(settings))),
              // Only the variational family owns Taylor bases, reconstruction
              // matrices and modal coefficient storage. Direct P0/P1 consume
              // means or physical gradients without a compatibility plan.
              reconstruction(settings.numerics.reconstruction == "variational"
                                 ? std::make_optional(CFV::VariationalReconstructionPlan::ConstructRecCoeff(
                                       metrics, settings.numerics.cfv,
                                       FiniteVolumeDetail::ReconstructionSupport(settings)))
                                 : std::nullopt),
              gradient(CFV::GradientPlan::Prepare(*mesh, metrics, settings.numerics.gradient == "greenGauss" ? CFV::GradientMethod::GreenGauss : settings.numerics.gradient == "ggmp" ? CFV::GradientMethod::GGMP
                                                                                                                                                                                      : CFV::GradientMethod::LeastSquares)),
              partition(metrics.CellWorkPartition())
        {
            if (settings.execution.communicationExecution == "worker")
                communicationWorker = std::make_shared<Runtime::CommunicationWorker>();
            if (!equation.HasMixture())
            {
                fixedFlowThermalBounds = equation.PrepareThermalBounds(
                    settings.numerics.densityFloor,
                    settings.numerics.temperatureFloor,
                    settings.numerics.pressureFloor);
                defaultAdmissibility.emplace(
                    equation, settings.numerics.densityFloor,
                    settings.numerics.temperatureFloor,
                    settings.numerics.sensibleEnergyFloor,
                    settings.numerics.pressureFloor,
                    equation.HasRANS()
                        ? RANSStateView{equation.RANS().Kind(),
                                        equation.RANS().ComponentNumber()}
                        : RANSStateView{});
            }
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
