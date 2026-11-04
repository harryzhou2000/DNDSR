#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Geom
{
    class BoundaryPatch;
}
#include "Geom/Access/EntityWorkPartition.hpp"
#include <array>
#include <optional>
#include <atomic>
#include <map>

namespace DNDS::Cauchy
{
    /// Persistent buffers/plans for explicit mean-gradient preparation. One
    /// field per physical equation allocation, columns component*dim+direction.
    /// Gradient values use the cell's mesh frame; periodic tensors are rotated
    /// at stencil consumption. Copies share buffers and must not execute at
    /// the same time. Build for the exact state binding used by the caller.
    struct EquationMeanGradients
    {
        std::vector<CFV::FVField> fields;

    private:
        friend class FiniteVolume;
        struct Data;
        std::shared_ptr<Data> _data;
    };
    /// Optional prepared inputs to the stateless auxiliary property kernels.
    /// Two-equation RANS requires meanGradients. No implicit gradient solve or
    /// exchange is performed by BeginAuxValues/EquationAuxiliaryPreparation.
    struct EquationAuxiliaryInputs
    {
        const EquationMeanGradients *meanGradients = nullptr;
    };
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
        DerivedRequest derived;
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
        std::optional<DerivedFields> derivedOwned, derivedGhost;

    private:
        friend class FiniteVolume;
        bool _derivedThermodynamics = false;
        struct Origin
        {
            std::shared_ptr<const void> owner;
            // Allocation identity alone does not identify a row view: distinct
            // ArrayStorage objects can share one allocation at different offsets.
            Runtime::ArrayStorage::Shared ownedStorage, ghostStorage;
            uint64_t ownedVersion, ghostVersion, ownedShape, ghostShape;
            real time;
            std::optional<BoundaryInputs> boundary;
            std::array<bool, 7> present{};
            std::array<Runtime::ArrayStorage::Shared, 7> storage{};
            std::array<Runtime::ArrayStorage::Shared, 3> ghosts{};
            std::array<Runtime::ArrayStorage::Shared, 2> derived{};
            std::array<uint64_t, 7> versions{}, shapes{};
        };
        std::optional<Origin> _origin;
        struct EquationOrigin;
        std::shared_ptr<const EquationOrigin> _equationOrigin;
        // Only needed when a cell reduction is requested without its matching
        // public face output. Columns: spectral sum, wave jump, pressure min.
        std::optional<Runtime::Array<real>> _faceScratch;
    };
    using RHSFieldView = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSScalarView = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSOutputView = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    /// Borrowed [p,a,diffusivity] projection, possibly in a wider derived table.
    /// Preserve parent layout and retain the producer's read scopes.
    struct MeanThermodynamicView
    {
        RHSFieldView values;
        std::array<int, 3> columns{0, 1, 2};
        index width = 3;
        MeanThermodynamicView() = default;
        MeanThermodynamicView(RHSScalarView combined) : values{combined, {}}, width(combined.Width()) {}
        MeanThermodynamicView(RHSFieldView paired, std::array<int, 3> selected) : values(paired), columns(selected) {}
        static MeanThermodynamicView Derived(const DerivedFields::ReadScope &owned, const DerivedFields::ReadScope *ghost = nullptr)
        {
            const std::array<int, 3> selected{owned.Column(DerivedQuantity::Pressure), owned.Column(DerivedQuantity::SoundSpeed), owned.Column(DerivedQuantity::DiffusionBound)};
            if (ghost)
                DNDS_check_throw_info((selected == std::array<int, 3>{ghost->Column(DerivedQuantity::Pressure), ghost->Column(DerivedQuantity::SoundSpeed), ghost->Column(DerivedQuantity::DiffusionBound)}), "derived owned/ghost property columns differ");
            return {{owned.View(), ghost ? ghost->View() : RHSScalarView{}}, selected};
        }
        index RowCount() const { return values.RowCount(); }
        index Width() const { return width; }
        real operator()(index iRow, int column) const { return values(iRow, columns[column]); }
    };
    /// Packed read-only auxiliary inputs, independent of their producer. Current
    /// fields are mean [p,a,diffusivity], wave jumps and pressure minima. Extend
    /// this bundle for cached temperature/masks rather than adding positional
    /// arguments to every kernel. Optional absence is explicit even on empty ranks.
    struct RHSAuxiliaryViews
    {
        std::optional<MeanThermodynamicView> meanThermodynamics;
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
        /// Optional distinct polynomial for viscosity. Absence uses coefficients.
        /// Retain its read leases through execution, exactly as for coefficients.
        std::optional<RHSFieldView> viscousCoefficients;
        std::optional<RHSScalarView> boundary;
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
        struct LimitedVRGeometry;
        struct LimitedVRWorkspace;
        struct MatrixFreeJacobian;
        struct EquationJacobian;
        struct EquationPositivityWorkspace;
        std::shared_ptr<const Data> _data;
        void ValidateBoundaryLayout(const BoundaryInputs &input) const;

    public:
        struct Workspace;
        struct MeanWorkspace;
        struct EquationWorkspace;
        /// Full logical-state CPU workspace on the same geometry/CFV plans.
        /// Each physical field keeps its layout and shared row routing; no
        /// whole-state packing is performed. Matching schema/physical grouping
        /// is collective; payload layouts may differ across ranks. A mixture
        /// requires a matching explicit worker pool, shared in capacity only.
        /// Initial equation-state profile: Rusanov, constant/gradient/VR,
        /// componentwise WBAP, Euler quadrature PP, polynomial Newtonian/Fourier
        /// diffusion and optional mixture chemistry. Extrapolate, farfield,
        /// constant prescribed and adiabatic/isothermal wall producers use the
        /// actual EOS. Face fallback, characteristic WBAP, other flux preparations,
        /// interfaces and nonconstant analytic profiles are rejected here
        /// until their equation-state implementations are connected. Existing
        /// flow5 Workspace retains those established capabilities.
        std::shared_ptr<EquationWorkspace> BuildEquationWorkspace(EquationDofs &state,
                                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt) const;
        /// Local preflight of owner and bound state allocations. No numerical
        /// access, property refresh or communication occurs here.
        void ValidateWorkspace(const EquationDofs &state, const EquationWorkspace &workspace) const;
        /// Integrated residual for every logical equation block. CPU controller
        /// entry is synchronous; mean/coefficient transfers overlap owned-only
        /// core work when configured. Caller serializes workspace/state handles
        /// and coordinates collective call order. Failed scratch is discarded.
        /// This updates state ghosts and reconstruction scratch, never owned
        /// means. Each output physical allocation must be distinct from inputs.
        /// implicit.collectiveChecks=false delegates distributed failure
        /// coordination to the caller; entry/binding preflights still agree.
        void Evaluate(EquationDofs &state, EquationDofs &residual, EquationWorkspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        const EquationDofs &ReconstructionGuess(const EquationWorkspace &workspace) const;
        /// Collective transactional owned warm-guess restore: finite-value and
        /// lease preflight precedes every write. No halo transfer/mean mutation.
        /// Caller joins workspace work and supplies a distinct snapshot.
        void RestoreReconstructionGuess(EquationWorkspace &workspace, const EquationDofs &guess) const;
        /// Local reset; caller coordinates failures and joins workspace work.
        void ResetReconstruction(EquationWorkspace &workspace) const;
        const EquationDofs &FluxReconstruction(const EquationWorkspace &workspace) const;
        /// Latest complete equation evaluation's global coefficient-update
        /// norm. No extra reduction occurs when reading these diagnostics.
        /// Fixed sweeps without a tolerance leave the update norm unmeasured
        /// (infinity); constant/gradient reconstruction reports zero.
        int LastReconstructionIterations(const EquationWorkspace &workspace) const;
        real LastReconstructionUpdate(const EquationWorkspace &workspace) const;
        /// Owned-only polynomial projection. Standalone use starts from the
        /// unlimited guess; Evaluate preserves its preceding WBAP result.
        /// No halo transfer occurs here. Caller serializes workspace/state.
        void LimitReconstructionPP(const EquationDofs &state, EquationWorkspace &workspace) const;
        /// Collective count/min snapshot; no implicit refresh or mutation.
        std::pair<index, real> PositivityReconstructionStats(const EquationWorkspace &workspace) const;
        /// Whole-equation counterpart of AuxiliaryPreparation. Holds numeric
        /// input scopes and reserves optional output arrays before writing.
        /// Core reads owned inputs only; complete every input field's halo
        /// before HaloDependent. All runs those phases consecutively. No MPI
        /// or hidden state update occurs here. Finish publishes exact source,
        /// output and time provenance. Do not replace buffers during the scope.
        /// Keep FV/derived-bundle owners on the controller through worker joins;
        /// retained preparation plans may own MPI resources indirectly.
        class EquationAuxiliaryPreparation
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit EquationAuxiliaryPreparation(std::unique_ptr<State> state);
            friend class FiniteVolume;

        public:
            EquationAuxiliaryPreparation(EquationAuxiliaryPreparation &&) noexcept;
            EquationAuxiliaryPreparation &operator=(EquationAuxiliaryPreparation &&) noexcept;
            ~EquationAuxiliaryPreparation();
            void Apply(CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All);
            void Finish(AuxiliaryValues &output);
        };
        /// Synchronous CPU property preparation with explicit mixture workers.
        /// The caller owns collective order/error handling and halo completion;
        /// even an empty rank uses the same outer schedule. Boundary producers
        /// match EquationWorkspace. Two-equation closures require a current
        /// explicit mean-gradient publication; neither overload exchanges data.
        EquationAuxiliaryPreparation BeginAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time = 0,
                                                    std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                                    EquationAuxiliaryInputs inputs = {}) const;
        void PrepareAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time = 0,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                              EquationAuxiliaryInputs inputs = {}) const;
        /// Collective allocation/binding only, not a gradient evaluation. The
        /// returned workspace is tied to these physical allocations and row
        /// spaces; rebuild after rebinding/redistributing the equation state.
        EquationMeanGradients BuildMeanGradients(EquationDofs &state) const;
        /// Collective CPU orchestration: mean halo + gradient core, dependent
        /// gradients, then gradient halo + auxiliary core, and auxiliary rest.
        /// Uses persistent scratch and explicit workers. No VR coefficients are
        /// read or changed. Inline/worker and overlap/unsplit share the kernels.
        /// Release conflicting views first. All ranks enter with the same
        /// physical grouping; no simultaneous use of shared workspace copies.
        /// Returns with CPU-visible outputs. Mutating state or gradients makes
        /// a retained auxiliary publication stale and validation rejects it.
        void PrepareAuxValues(EquationDofs &state, AuxiliaryValues &output, EquationMeanGradients &gradients,
                              Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt) const;
        void ValidateAuxValues(const EquationDofs &state, const AuxiliaryValues &values, real time = 0) const;
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
        /// native row space. No payload is copied from the prototype. Evaluated
        /// mode omits off-diagonal storage; PrepareJacobian supplies its frozen
        /// physical snapshot. Both modes retain explicit diagonal storage.
        std::shared_ptr<LinearProblemContext> BuildLinearProblem(const CFV::FVField &prototype, DiagonalStorage diagonal,
                                                                 OffDiagonalStorage offDiagonal = OffDiagonalStorage::Assembled) const;
        std::shared_ptr<LinearProblemContext> BuildLinearProblem(const EquationDofs &prototype, DiagonalStorage diagonal,
                                                                 OffDiagonalStorage offDiagonal = OffDiagonalStorage::Assembled) const;
        /// Prepare integrated approximate-Jacobian coefficients from
        /// frozen means and prepared face signals. No input mutation, MPI or
        /// inverse/factor preparation occurs here. inversePseudoTime is an
        /// owned-cell scalar table. Caller completes all halos before entry.
        /// Semi-matrix-free mode copies frozen physical data, reuses immutable
        /// geometry, and regenerates blocks during action/SGS/ILU filling. It
        /// retains no live mean/auxiliary leases after preparation returns.
        void PrepareJacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                             const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                             JacobianPreparation controls = {}, real time = 0) const;
        /// Whole-equation CPU approximation with the same integrated/scalar
        /// dissipation convention. Uses actual conservative EOS derivatives,
        /// ALE/rotation terms, chemical source derivatives at the RHS volume
        /// quadrature points and scalar molecular/species diffusion bounds. This is not a
        /// derivative of the complete reconstructed viscous/source residual.
        /// Roe characteristics are not connected for this profile and reject.
        /// Frozen evaluated blocks retain numeric snapshots/geometry only;
        /// they perform no EOS calls, MPI or live-state/cache access.
        /// The caller completes halos, serializes matrix/input bindings and
        /// coordinates failures. No inverses/factors are updated implicitly.
        /// Output writers are reserved together; partial failure poisons them.
        /// Nonconstant chemical reconstruction requires sourceReconstruction
        /// from a successful Evaluate at this state/time. Values/coefficients
        /// must remain unchanged. Source derivatives freeze that reconstruction,
        /// as in Euler; constant reconstruction may omit this workspace.
        void PrepareJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary,
                             const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                             JacobianPreparation controls = {}, real time = 0,
                             std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                             const EquationWorkspace *sourceReconstruction = nullptr) const;
        FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation);
        const CFV::FiniteVolumePlan &Metrics() const;
        const EquationSystem &Equation() const;
        const CFV::VariationalReconstructionPlan &Reconstruction() const;
        const Geom::UnstructuredMesh &Mesh() const;
        const RigidMotion &Motion() const;
        const std::vector<BoundaryPoint> &BoundaryPoints() const;
        /// Direct owned-boundary selection for a physical coupling adapter.
        /// The result retains the reference mesh without exposing a mutable
        /// owner handle through the solver's read-only mesh facade.
        Geom::BoundaryPatch BuildBoundaryPatch(int zone) const;
        bool HasInterfaces() const;
        BoundaryInputs BuildBoundaryInputs() const;
        /// Synchronous local point operations, no MPI. Begin withdraws the
        /// publication; publish checks complete admissible interface rows only.
        void BeginBoundaryUpdate(BoundaryInputs &input) const;
        void PublishBoundaryInputs(BoundaryInputs &input, real time, StateEpoch epoch = {}) const;
        void ValidateBoundaryInputs(const BoundaryInputs &input, real time) const;
        /// Share a packet with another FV level only after proving identical
        /// boundary point order/geometry. No interpolation or payload copy.
        BoundaryInputs AliasBoundaryInputs(const BoundaryInputs &input) const;
        void SetBoundaryInputs(Workspace &workspace, const BoundaryInputs &input) const;
        void SetBoundaryInputs(MeanWorkspace &workspace, const BoundaryInputs &input) const;
        /// Additional channels are allocated once and fused into auxiliary
        /// preparation. Implicit consumers request face/cell spectral values;
        /// explicit consumers retain the minimal default packet. Collective:
        /// domain ranks must request the same resolved channels/workspace mode.
        std::shared_ptr<Workspace> BuildWorkspace(CFV::FVField &state, AuxiliaryRequest additional = {}) const;
        /// Semantic frontend. Validate the equation profile and backend before
        /// preparing persistent transfers. Initial kernels require one coupled
        /// physical field; split bindings fail without flattening or copying.
        std::shared_ptr<Workspace> BuildWorkspace(EquationDofs &state, AuxiliaryRequest additional = {}) const;
        /// Separate typed scratch for Euler's zero-reconstruction physical-dt
        /// trial. It cannot be supplied to a nonlinear reconstruction solve.
        std::shared_ptr<MeanWorkspace> BuildMeanWorkspace(CFV::FVField &state) const;
        /// Check allocation capability before the first Evaluate. This does not
        /// certify prepared values or input provenance; PreparedAuxValues does.
        void RequireWorkspaceChannels(const Workspace &workspace, AuxiliaryRequest required) const;
        /// Allocate the requested members once. Cell reductions without matching
        /// face outputs also need private face scratch. Callers may instead
        /// supply compatible buffers themselves, including strided tables;
        /// BeginAuxValues lazily allocates their missing private scratch once.
        AuxiliaryValues BuildAuxiliaryValues(AuxiliaryRequest request) const;
        const std::vector<index> &AuxiliaryFaces() const;
        /// Move-only CPU preparation transaction. Core reads owned means only;
        /// finish the mean halo before HaloDependent. All runs both phases.
        /// Face values are evaluated once, cell reductions preserve sorted-face
        /// order, and no output validates until Finish. The scope retains all
        /// allocations and leases; do not invoke its phases concurrently.
        /// This is a synchronous CPU phase API. If a caller submits a phase to
        /// an engine, retain the transaction through the stream join. Host-side
        /// Finish follows that join; a queued continuation may Finish directly
        /// after its synchronous dependent phase. Evaluate owns these joins.
        class AuxiliaryPreparation
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit AuxiliaryPreparation(std::unique_ptr<State> state);
            friend class FiniteVolume;

        public:
            AuxiliaryPreparation(const AuxiliaryPreparation &) = delete;
            AuxiliaryPreparation &operator=(const AuxiliaryPreparation &) = delete;
            AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept;
            AuxiliaryPreparation &operator=(AuxiliaryPreparation &&) noexcept;
            ~AuxiliaryPreparation();
            void Apply(CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All);
            /// Publish to the same output buffers (or an exact sharing copy).
            /// Reject incomplete work or replaced outputs. Releases all leases;
            /// abandoning a scope leaves the result without valid provenance.
            void Finish(AuxiliaryValues &output);
        };
        /// Begin without acquiring the input ghost lease. The caller must order
        /// its halo completion before the dependent phase. The input relation
        /// is Adj::Cell2CellFace, using the conservative cell-node-cell core.
        /// Requested outputs remain reserved until Finish/destruction. Unwinding
        /// after partial writes follows Runtime failure semantics: replace failed
        /// output buffers/workspace before retrying. No communication is posted.
        AuxiliaryPreparation BeginAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time = 0, const BoundaryInputs *boundary = nullptr) const;
        /// Stateless fused CPU operator. Input halos must already be current and
        /// host-visible. Writes only the explicit optional outputs, performs no
        /// communication/global reduction, and never mutates state or FV caches.
        /// Private face scratch belongs to output, not the immutable FV plan.
        void PrepareAuxValues(const CFV::FVField &state, AuxiliaryValues &output, real time = 0, const BoundaryInputs *boundary = nullptr) const;
        /// Reject changed inputs, geometry, boundary time, output selection,
        /// cell row identity or overwritten buffers before a result is reused.
        void ValidateAuxValues(const CFV::FVField &state, const AuxiliaryValues &values, real time = 0) const;
        real StableStep(const CFV::FVField &state, const AuxiliaryValues &values, real cfl, real time = 0) const;
        real StableStep(const EquationDofs &state, const AuxiliaryValues &values, real cfl, real time = 0) const;
        /// Clear owned polynomial and retained LVR gates at a synchronous
        /// workspace boundary. Local only; reserve all local destinations before
        /// writing, and let the caller coordinate rank-local errors. Evaluate
        /// refreshes coefficient ghosts before they are subsequently consumed.
        void ResetReconstruction(Workspace &workspace) const;
        /// Fixed-sweep VR carries a numerical initial guess between stages.
        /// Retain it in a restart for exact continuation; only owned values are
        /// persistent, since Evaluate refreshes ghosts before consuming them.
        const CFV::FVField &ReconstructionGuess(const Workspace &workspace) const;
        /// Completed flux polynomial, after optional WBAP/PP. Filtering never
        /// overwrites ReconstructionGuess(), which remains the unlimited guess.
        /// With filtering disabled these references share the same coefficients.
        const CFV::FVField &FluxReconstruction(const Workspace &workspace) const;
        /// Prepare the same boundary/VR/limiter polynomial as Evaluate, without
        /// computing a residual. Refreshes its canonical halos before returning.
        /// A coupling controller may sample it to refresh incoming traces.
        /// Interface packets must already be published at this stage time.
        void PrepareReconstruction(CFV::FVField &state, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &stream, real time) const;
        /// Refresh stage means and the retained unlimited polynomial for a
        /// predictor sampling round. Reuses the workspace's canonical bindings.
        void RefreshSamplingInputs(CFV::FVField &state, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &stream) const;
        /// Interpretation of stored polynomial rows; independent of physical
        /// flow schema. Restart may restore means and cold-start incompatible
        /// coefficients rather than matching a C++ static row type.
        Json ReconstructionMetadata() const;
        /// Validate shape/finite values collectively, then replace the owned
        /// initial guess. Caller serializes workspace and publication accesses.
        /// Optional LVR history is restored in the same write transaction.
        /// Omitting it explicitly cold-starts that history when LVR is active.
        void RestoreReconstructionGuess(Workspace &workspace, const Runtime::Array<real> &coefficients,
                                        const Runtime::Array<real> *limitedVRHistory = nullptr) const;
        /// Optional owned [shock,contact] retained LVR history. The returned
        /// pointer borrows the workspace; a copied Array retains its allocation.
        /// Release views before Evaluate.
        /// Domain rollback/observations/warm restart preserve this with the VR
        /// guess. RestoreReconstructionGuess restores both or cold-starts history.
        const Runtime::Array<real> *LimitedVRHistory(const Workspace &workspace) const;
        /// Most recent RHS's frozen alpha, not persistent restart state. It is
        /// overwritten by Evaluate and cleared by ResetReconstruction; a warm
        /// history restore does not prepare a new alpha. Same borrowing rules.
        const Runtime::Array<real> *LimitedVRAlpha(const Workspace &workspace) const;
        /// Integrated d(V*U)/dt, owned rows only. Mean/rec/gradient exchanges are
        /// explicit inside this domain-level routine; numerical kernels use
        /// owned-only core inputs while ghost transfers are active. The caller
        /// serializes residuals on this workspace and coordinates domain errors.
        /// Synchronous collective: invoke on an MPI-authorized controller thread.
        /// Queued work is observed on success and before borrowed scopes unwind
        /// on exceptions; failed scratch allocations require replacement.
        void Evaluate(CFV::FVField &state, CFV::FVField &residual, Workspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        /// Equation-independent entry; dispatch/semantic checks occur outside
        /// cell loops. The field overload is the trusted compressible-kernel
        /// adapter used by stage and multigrid scratch, not a model selector.
        void Evaluate(EquationDofs &state, CFV::FVField &residual, Workspace &workspace,
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
        /// Fill exact code-unit conservative values in equation-schema order.
        /// Applies the initial rigid-frame rotation to flow momentum only.
        /// Caller resolves/validates physical input with ResolvePhysicalState.
        void InitializeConservative(EquationDofs &state, const std::vector<real> &values) const;
        /// Collective error-coordinated constant initialization in ground axes.
        /// Uses actual EOS/full thermochemical energy, resolves named species
        /// once, and writes logical blocks through their native layout. Positive
        /// temperature overrides primitive p. No ghost exchange or reconstruction
        /// initialization; caller releases conflicting views/async work first.
        /// Mixtures require an explicit matching worker pool. RANS requires
        /// every named conservative scalar; all inputs are validated before
        /// writing any block. Does not infer turbulence values from flow.
        void InitializeConstant(EquationDofs &state, const FlowState &primitive, const std::map<std::string, real> &massFractions = {},
                                real temperature = 0, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                const std::map<std::string, real> &ransConservative = {}) const;
        /// Setup-only, collective error guard. Apply configured spatial
        /// overrides to the initialized constant background and conservatively
        /// average cell quadrature states. Writes owned values; no halo refresh.
        /// One serial expression/EOS context is retained for the whole pass.
        /// A point failure can leave partial setup output; discard it on error.
        void InitializeField(EquationDofs &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers = {}) const;
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
        /// Synchronous correction preparation when the time method already
        /// supplies a physical RHS cache. Refresh means and auxiliary values,
        /// without advancing reconstruction, limiter history or evaluating RHS.
        /// Input halo may be stale. Retain this bound workspace through return.
        void PrepareCorrectionAux(CFV::FVField &state, Workspace &workspace, Runtime::EagerEngine &engine,
                                  const Runtime::ExecutionStream &stream, real time) const;
        /// Full-state variant also refreshes sampled source state and RANS
        /// destruction from the retained polynomial, before publishing its
        /// Jacobian provenance. No chemical rates or physical RHS are evaluated.
        void PrepareCorrectionAux(EquationDofs &state, EquationWorkspace &workspace, AuxiliaryValues &auxiliary,
                                  Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const;
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
        /// Collective sum of the last completed equation RHS's wall momentum
        /// flux, in mesh axes, with the same sign as pressure*n - traction.
        /// Per-quadrature outputs include the actual numerical pressure, RANS
        /// stress and viscous traction; no separate EOS/flux recomputation.
        /// Workspace access is controller-serialized; do not read during Evaluate.
        Vector3 LastBoundaryForce(const EquationWorkspace &workspace, const std::vector<int> &zones) const;
        /// Native parallel VTKHDF cell means and rank-local surface quadrature
        /// samples. Does not gather mesh/state to a serial rank.
        void WriteVisualization(const CFV::FVField &state, const Workspace &workspace, const std::string &prefix, real time, int hdfDeflateLevel = 0) const;
        /// All conservative blocks plus mean thermodynamic/species fields and
        /// reconstructed boundary traces. Reads an observation workspace only;
        /// does not modify the production reconstruction or any state values.
        void WriteVisualization(const EquationDofs &state, const EquationWorkspace &workspace,
                                const std::string &prefix, real time, int hdfDeflateLevel,
                                const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const;

    private:
        void ValidateEquationSourceJacobian(const EquationDofs &state, const EquationWorkspace &workspace, real time) const;
        void AddEquationSourceJacobian(const EquationDofs &state, const EquationWorkspace &workspace,
                                       const Runtime::Array<real>::WriteLease &diagonal, DiagonalStorage storage,
                                       real scale, EquationPointEvaluator &evaluator) const;
        CFV::FVField &RequireState(EquationDofs &state) const;
        index PrepareFaceCharacteristics(const CFV::FVField &state, CFV::FaceCharacteristicTransform &transform) const;
        std::shared_ptr<Workspace> BuildWorkspaceImpl(CFV::FVField &state, AuxiliaryRequest request, bool meanOnly) const;
        void EvaluateImpl(CFV::FVField &state, CFV::FVField *residual, Workspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const;
        /// Evaluate preserves its preceding WBAP result; the public standalone
        /// projection always starts afresh from the unlimited guess.
        void LimitReconstructionPPImpl(const CFV::FVField &state, Workspace &workspace, bool refresh) const;
    };
}
