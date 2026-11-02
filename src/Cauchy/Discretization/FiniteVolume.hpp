#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include <array>
#include <optional>
#include <atomic>
#include <map>

namespace DNDS::Cauchy
{
    /// Explicit result of physical-step selection. Alpha/count describe the PP
    /// bound before any independent growth cap or subsequent retry halving.
    struct PhysicalStepLimit
    {
        real dt = 0;
        index limitedCells = 0;
        real minimumAlpha = 1;
        bool growthLimited = false;
    };
    struct AuxiliaryRequest
    {
        bool meanThermodynamics = false;
        bool faceSpectralRadius = false;
        bool faceWaveJump = false;
        bool facePressureMinimum = false;
        bool cellSpectralSum = false;
        bool cellWaveJump = false;
        bool cellPressureMinimum = false;
    };
    /// Explicit, extensible output buffers. Every member is optional; absent
    /// outputs are neither allocated nor written. Face rows follow AuxiliaryFaces().
    /// Cell reductions write owned rows only; their halo exchange is a caller task.
    struct AuxiliaryValues
    {
        std::optional<Runtime::Array<real>> meanThermodynamics; // local owned+ghost rows: p,a,diffusivity
        std::optional<Runtime::Array<real>> faceSpectralRadius; // convection, diffusion (speed)
        std::optional<Runtime::Array<real>> faceWaveJump;
        std::optional<Runtime::Array<real>> facePressureMinimum;
        std::optional<CFV::FVField> cellSpectralSum; // sum A*(lambda_c+lambda_v)
        std::optional<CFV::FVField> cellWaveJump;
        std::optional<CFV::FVField> cellPressureMinimum;

    private:
        friend class FiniteVolume;
        struct Origin
        {
            std::shared_ptr<const void> owner;
            // Allocation identity alone does not identify a row view: distinct
            // ArrayStorage objects can share one allocation at different offsets.
            Runtime::ArrayStorage::Shared ownedStorage, ghostStorage;
            uint64_t ownedVersion, ghostVersion, ownedShape, ghostShape;
            real time;
            std::array<bool, 7> present{};
            std::array<Runtime::ArrayStorage::Shared, 7> storage{};
            std::array<Runtime::ArrayStorage::Shared, 3> ghosts{};
            std::array<uint64_t, 7> versions{}, shapes{};
        };
        std::optional<Origin> _origin;
    };
    using RHSFieldView = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSScalarView = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSOutputView = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    /// Packed read-only auxiliary inputs, independent of their producer. Current
    /// fields are mean [p,a,diffusivity], wave jumps and pressure minima. Extend
    /// this bundle for cached temperature/masks rather than adding positional
    /// arguments to every kernel. Optional absence is explicit even on empty ranks.
    struct RHSAuxiliaryViews
    {
        std::optional<RHSScalarView> meanThermodynamics;
        std::optional<RHSFieldView> cellWaveJump, cellPressureMinimum;
    };
    struct RHSReadViews
    {
        RHSFieldView mean, coefficients, primitiveGradient;
        RHSAuxiliaryViews auxiliary;
        /// Euler's physical-dt trial ignores viscosity while retaining the
        /// configured boundary/convective flux policy. Gradients may be absent
        /// when false; normal production residuals keep the default true.
        bool includeViscosity = true;
    };
    /// Explicit outputs. Diagnostics accumulate across core/dependent calls;
    /// caller initializes them before an evaluation and reduces them afterwards.
    struct RHSWriteViews
    {
        RHSOutputView residual;
        std::optional<RHSOutputView> lowOrder;
        std::atomic<index> *faceFallbacks = nullptr;
        std::map<int, std::array<std::atomic<real>, 3>> *boundaryForces = nullptr;
    };
    /// CPU cell-centered discretization. Preparation retains immutable geometry,
    /// incidence, quadrature and polynomial data. Each simultaneous residual
    /// requires a distinct Workspace. No solver clock/history/IO policy here.
    class FiniteVolume
    {
        struct Data;
        std::shared_ptr<const Data> _data;

    public:
        struct Workspace;
        struct MeanWorkspace;
        /// Copyable immutable read scope for RHS geometry/basis data. Apply is
        /// stateless: no MPI, reconstruction, cache update or hidden counters.
        /// Raw input/output views require retained/declared Array leases and
        /// nonoverlapping output storage supplied by the caller. Core input
        /// views need only owned data. The checked Evaluate wrapper does this.
        class RHSOperator
        {
            struct ReadData;
            std::shared_ptr<const ReadData> _read;
            explicit RHSOperator(std::shared_ptr<const ReadData> read) : _read(std::move(read)) {}
            friend class FiniteVolume;

        public:
            void Apply(const RHSReadViews &input, const RHSWriteViews &output,
                       CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
        };
        RHSOperator AcquireRHSOperator() const;
        /// Allocate a fixed cell-face-cell matrix stencil using the prototype's
        /// native row space. No payload is copied from the prototype.
        std::shared_ptr<LinearProblemContext> BuildLinearProblem(const CFV::FVField &prototype, DiagonalStorage diagonal) const;
        /// Fill explicit integrated approximate-Jacobian coefficients from
        /// frozen means and prepared face signals. No input mutation, MPI or
        /// inverse/factor preparation occurs here. inversePseudoTime is an
        /// owned-cell scalar table. Caller completes all halos before entry.
        void PrepareJacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                             const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                             JacobianPreparation controls = {}, real time = 0) const;
        FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation);
        const CFV::FiniteVolumePlan &Metrics() const;
        const Geom::UnstructuredMesh &Mesh() const;
        /// Additional channels are allocated once and fused into auxiliary
        /// preparation. Implicit consumers request face/cell spectral values;
        /// explicit consumers retain the minimal default packet. Collective:
        /// domain ranks must request the same resolved channels/workspace mode.
        std::shared_ptr<Workspace> BuildWorkspace(CFV::FVField &state, AuxiliaryRequest additional = {}) const;
        /// Separate typed scratch for Euler's zero-reconstruction physical-dt
        /// trial. It cannot be supplied to a nonlinear reconstruction solve.
        std::shared_ptr<MeanWorkspace> BuildMeanWorkspace(CFV::FVField &state) const;
        /// Check allocation capability before the first Evaluate. This does not
        /// certify prepared values or input provenance; PreparedAuxValues does.
        void RequireWorkspaceChannels(const Workspace &workspace, AuxiliaryRequest required) const;
        /// Allocate precisely the requested members once. Callers may instead
        /// supply compatible buffers themselves, including strided tables.
        AuxiliaryValues BuildAuxiliaryValues(AuxiliaryRequest request) const;
        const std::vector<index> &AuxiliaryFaces() const;
        /// Stateless fused CPU operator. Input halos must already be current and
        /// host-visible. Writes only the explicit optional outputs, performs no
        /// communication/global reduction, and never mutates state or FV caches.
        void PrepareAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time = 0) const;
        /// Reject changed inputs, geometry, boundary time, output selection or
        /// overwritten owned output buffers before a prepared result is reused.
        void ValidateAuxValues(const CFV::FVField &state, const AuxiliaryValues &values, real time = 0) const;
        real StableStep(const CFV::FVField &state, const AuxiliaryValues &values, real cfl, real time = 0) const;
        /// Clear the polynomial initial guess at a synchronous workspace boundary.
        void ResetReconstruction(Workspace &workspace) const;
        /// Fixed-sweep VR carries a numerical initial guess between stages.
        /// Retain it in a restart for exact continuation; only owned values are
        /// persistent, since Evaluate refreshes ghosts before consuming them.
        const CFV::FVField &ReconstructionGuess(const Workspace &workspace) const;
        /// Completed flux polynomial, after optional WBAP/PP. Filtering never
        /// overwrites ReconstructionGuess(), which remains the unlimited guess.
        /// With filtering disabled these references share the same coefficients.
        const CFV::FVField &FluxReconstruction(const Workspace &workspace) const;
        /// Interpretation of stored polynomial rows; independent of physical
        /// flow schema. Restart may restore means and cold-start incompatible
        /// coefficients rather than matching a C++ static row type.
        Json ReconstructionMetadata() const;
        /// Validate shape/finite values collectively, then replace the owned
        /// initial guess. Caller serializes workspace and publication accesses.
        void RestoreReconstructionGuess(Workspace &workspace, const Runtime::Array<real> &coefficients) const;
        /// Integrated d(V*U)/dt, owned rows only. Mean/rec/gradient exchanges are
        /// explicit inside this domain-level routine; numerical kernels use
        /// owned-only core inputs while ghost transfers are active. The caller
        /// serializes residuals on this workspace and coordinates domain errors.
        void Evaluate(CFV::FVField &state, CFV::FVField &residual, Workspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        /// Euler physical-dt trial: configured convective flux/BC/auxiliary
        /// policy, zero polynomial and viscosity explicitly ignored. No VR,
        /// WBAP or reconstruction PP. Uses only its independent typed scratch;
        /// accepted means, clock and the normal VR workspace are unchanged.
        void EvaluateMean(CFV::FVField &state, CFV::FVField &residual, MeanWorkspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        /// Collective Euler compress=1 bound from dt*scale*meanResidual/volume.
        /// Inputs are owned-only CPU flow5 fields; state ghost writes may be
        /// active. No input mutation, time advance, growth cap or communication
        /// of payloads. A zero returned dt means no positive admissible bound.
        /// All ranks enter with matching finite positive dt/scale, relax in
        /// (0,1], and positive compressionPower. Mean residual must be ready
        /// and nonaliased with state. Uses the equation's thermal energy bounds.
        PhysicalStepLimit LimitPhysicalStep(const CFV::FVField &state, const CFV::FVField &meanResidual,
                                            real dt, real relax, real scale, int compressionPower = 1) const;
        void Initialize(CFV::FVField &state, const FlowProfileSettings &profile, const FlowState &primitive) const;
        /// Volume-normalized L1/L2/Linf error in conservative cell averages.
        std::array<FlowState, 3> Error(const CFV::FVField &state, const FlowProfileSettings &profile, const FlowState &primitive, real time) const;
        real StableStep(const CFV::FVField &state, real cfl) const;
        /// Collective owned-only admissibility; never repairs a state silently.
        bool Admissible(const CFV::FVField &state) const;
        index LastFaceFallbacks(const Workspace &workspace) const;
        int LastReconstructionIterations(const Workspace &workspace) const;
        real LastReconstructionUpdate(const Workspace &workspace) const;
        const CFV::FVField &LowOrderResidual(const Workspace &workspace) const;
        /// Explicit mean/face/cell outputs of this workspace's last Evaluate.
        /// Only flux-required stencil channels are present. Cell ghosts are
        /// refreshed from owners using the field's canonical mapping.
        const AuxiliaryValues &PreparedAuxValues(const Workspace &workspace) const;
        const ThermalBounds &ThermalConstraints() const;
        /// Collective checked projection of a fresh copy of the unlimited
        /// guess into flux coefficients, retaining the unlimited guess. Reads only
        /// owned means and cached owned-cell points; a mean ghost write may be
        /// active. Coefficient transfers must be complete before publication.
        /// Explicit outputs are limited coefficients and per-call beta stats.
        /// Does not reconstruct, exchange, advance time or change cell means.
        void LimitReconstructionPP(const CFV::FVField &state, Workspace &workspace) const;
        index LastCharacteristicFallbacks(const Workspace &workspace) const;
        std::pair<index, real> PositivityReconstructionStats(const Workspace &workspace) const;
        Vector3 LastBoundaryForce(const Workspace &workspace, const std::vector<int> &zones) const;
        /// Native parallel VTKHDF cell means and rank-local surface quadrature
        /// samples. Does not gather mesh/state to a serial rank.
        void WriteVisualization(const CFV::FVField &state, const Workspace &workspace, const std::string &prefix, real time, int hdfDeflateLevel = 0) const;

    private:
        std::shared_ptr<Workspace> BuildWorkspaceImpl(CFV::FVField &state, AuxiliaryRequest request, bool meanOnly) const;
        void EvaluateImpl(CFV::FVField &state, CFV::FVField &residual, Workspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const;
        /// Evaluate preserves its preceding WBAP result; the public standalone
        /// projection always starts afresh from the unlimited guess.
        void LimitReconstructionPPImpl(const CFV::FVField &state, Workspace &workspace, bool refresh) const;
    };
}
