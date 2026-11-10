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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Cauchy
{
    namespace EquationFVDetail
    {
        struct MeanLinearizationOutput;
    }
    struct EquationThermalView;
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
        std::optional<std::reference_wrapper<const EquationMeanGradients>> meanGradients;
        std::optional<std::reference_wrapper<const BoundaryInputs>> boundary;
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
    /// One kernel region's plain, movable diagnostics. The cell loop updates
    /// only this local value and returns it for one final merge. A future OMP
    /// implementation can reduce the same value per thread; no atomic belongs
    /// in a face/quadrature hot loop.
    class RHSKernelDiagnostics
    {
        index _faceFallbacks = 0;
        std::vector<Vector3> _forces;
        friend class RHSLocalDiagnostics;

    public:
        explicit RHSKernelDiagnostics(index forceSlots = 0) : _forces(size_t(forceSlots), Vector3{}) {}
        void AddFaceFallback() { ++_faceFallbacks; }
        void AddBoundaryForce(index slot, int component, real value)
        {
            DNDS_assert_info(slot >= 0 && size_t(slot) < _forces.size() && component >= 0 && component < 3,
                             "invalid RHS boundary-force slot");
            _forces[size_t(slot)][component] += value;
        }
        index FaceFallbacks() const { return _faceFallbacks; }
    };
    /// Workspace-owned, rank-local aggregate for one complete RHS evaluation.
    /// Boundary zones are resolved to dense slots during plan setup; kernels
    /// neither search a map nor mutate this aggregate from their inner loop.
    class RHSLocalDiagnostics
    {
        std::vector<int> _zones;
        std::vector<Vector3> _forces;
        index _faceFallbacks = 0;

    public:
        RHSLocalDiagnostics() = default;
        explicit RHSLocalDiagnostics(std::vector<int> zones);
        void Reset();
        void Merge(const RHSKernelDiagnostics &input);
        index FaceFallbacks() const { return _faceFallbacks; }
        Vector3 BoundaryForce(const std::vector<int> &zones) const;
    };
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
        /// Direct FV physical conservative gradients, component*dim+direction.
        /// Required for direct P1; P0 face flux never reads this input.
        std::optional<RHSFieldView> conservativeGradient, viscousConservativeGradient;
        /// Explicit physical-dt/mean trial; no reconstructed point increment.
        bool meanOnly = false;
    };
    /// Explicit numerical outputs. Diagnostics are returned by the RHS kernel
    /// as a value and merged after each core/dependent operation completes.
    struct RHSWriteViews
    {
        RHSOutputView residual;
        std::optional<RHSOutputView> lowOrder;
        /// Optional integrated negative outward flux in AuxiliaryFaces() order,
        /// in the prepared anchor cell's axes. Selected faces are overwritten;
        /// core/dependent sets can overlap. No source term is included.
        std::optional<RHSOutputView> faceFlux;
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
        enum class RHSKernelPolicy
        {
            Auto,
            Generic,
            FixedFlow5
        };
        struct WorkspaceOptions
        {
            /// Resolve once during workspace construction. Generic is the
            /// layout/physics-general CPU oracle; FixedFlow5 requires one
            /// perfect-gas flow5 field and never changes the outer schedule.
            RHSKernelPolicy rhsKernel;
            constexpr explicit WorkspaceOptions(RHSKernelPolicy policy = RHSKernelPolicy::Auto)
                : rhsKernel(policy) {}
        };
        struct Workspace;
        struct MeanWorkspace;
        /// Reserved synchronous warm-guess publication. Construction validates
        /// every input and retains its reads, destination writes and workspace.
        /// Commit performs only prevalidated copies; abandonment cancels the
        /// untouched writers. Caller serializes workspace use until retirement.
        /// This permits several domains to reserve before any one publishes.
        class ReconstructionRestore
        {
            friend class FiniteVolume;
            std::function<void()> _commit;
            explicit ReconstructionRestore(std::function<void()> commit) : _commit(std::move(commit)) {}

        public:
            ReconstructionRestore(ReconstructionRestore &&other) noexcept { _commit.swap(other._commit); }
            ReconstructionRestore &operator=(ReconstructionRestore &&other) noexcept
            {
                if (this != &other)
                {
                    _commit = nullptr;
                    _commit.swap(other._commit);
                }
                return *this;
            }
            ReconstructionRestore(const ReconstructionRestore &) = delete;
            ReconstructionRestore &operator=(const ReconstructionRestore &) = delete;
            void Commit()
            {
                DNDS_check_throw_info(bool(_commit), "reconstruction restore was already committed or moved");
                auto commit = std::move(_commit);
                _commit = nullptr;
                commit();
            }
        };
        /// Reservation is memory-only. With collectiveChecks=false it performs
        /// no MPI; the caller must agree all reservations before any Commit.
        /// Simulation uses that mode for one cross-domain publication preflight.
        ReconstructionRestore ReserveReconstructionRestore(std::shared_ptr<Workspace> workspace, const EquationDofs &guess,
                                                           const Runtime::Array<real> *limitedVRHistory = nullptr,
                                                           bool collectiveChecks = true) const;
        /// Full logical-state CPU workspace on the same geometry/CFV plans.
        /// Each physical field keeps its layout and shared row routing; no
        /// whole-state packing is performed. Matching schema/physical grouping
        /// is collective; payload layouts may differ across ranks. A mixture
        /// requires a matching explicit worker pool, shared in capacity only.
        /// Initial equation-state profile: Rusanov, constant/gradient/VR,
        /// componentwise WBAP, Euler quadrature PP, polynomial Newtonian/Fourier
        /// diffusion and optional mixture chemistry. Extrapolate, farfield,
        /// constant prescribed and adiabatic/isothermal wall producers use the
        /// actual EOS. Operator-local fixed-flow leaves retain established
        /// flow5 arithmetic without a second workspace or execution loop.
        std::shared_ptr<Workspace> BuildWorkspace(EquationDofs &state,
                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                                                  WorkspaceOptions options = WorkspaceOptions()) const;
        /// Separate typed scratch for Euler's zero-reconstruction physical-dt
        /// trial. Shares geometry and kernels, never the ordinary warm guess.
        /// Includes configured volume/rotation sources, ignores viscous faces.
        /// RANS sources still require their actual molecular properties.
        std::shared_ptr<MeanWorkspace> BuildMeanWorkspace(EquationDofs &state,
                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt) const;
        /// Local preflight of owner and bound state allocations. No numerical
        /// access, property refresh or communication occurs here.
        void ValidateWorkspace(const EquationDofs &state, const Workspace &workspace) const;
        /// Whether the prepared Jacobian operator set contains the established
        /// fixed-flow Roe leaf. This is an operator capability query, not a
        /// selection between solver orchestration paths.
        bool HasRoeJacobianKernel(const Workspace &workspace) const;
        /// Integrated residual for every logical equation block. CPU controller
        /// entry is synchronous; mean/coefficient transfers overlap owned-only
        /// core work when configured. Caller serializes workspace/state handles
        /// and coordinates collective call order. Failed scratch is discarded.
        /// This updates state ghosts and reconstruction scratch, never owned
        /// means. Each output physical allocation must be distinct from inputs.
        /// implicit.collectiveChecks=false omits optional stage/launch safety
        /// agreements and aborts on local rejection; matching call order stays
        /// the caller's responsibility. Required setup protocols and numerical
        /// reductions remain. Access checking is an independent engine choice.
        void Evaluate(EquationDofs &state, EquationDofs &residual, Workspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        /// Prepare the requested auxiliary outputs during the mean exchange,
        /// before reconstruction/RHS. RHS reads this explicit publication;
        /// the caller may then consume its spectra for pseudo-time/Jacobian
        /// setup without recomputing and overwriting RHS-refined signals.
        /// Buffers must be distinct from state/residual and retained through
        /// this synchronous call. The no-aux overload retains its old scope.
        void Evaluate(EquationDofs &state, EquationDofs &residual, Workspace &workspace,
                      AuxiliaryValues &auxiliary, Runtime::EagerEngine &engine,
                      const Runtime::ExecutionStream &stream, real time = 0) const;
        /// The same mean/gradient/reconstruction/limiter preparation as Evaluate,
        /// including completed halos, but no flux/source evaluation or dummy
        /// residual allocation. Use before independent EvaluateFlux/AddSource.
        void PrepareReconstruction(EquationDofs &state, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &stream, real time = 0) const;
        void RefreshSamplingInputs(EquationDofs &state, Workspace &workspace, Runtime::EagerEngine &engine,
                                   const Runtime::ExecutionStream &stream) const;
        /// Same synchronous collective/access contract as Evaluate. Only this
        /// trial's private scratch and distinct residual are written; state
        /// owned values and ordinary reconstruction are unchanged.
        /// This zero-polynomial physical-dt probe retains parent quadrature,
        /// as in Euler; it is not a separately configured P0 FV operator.
        void EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time = 0) const;
        const EquationDofs &ReconstructionGuess(const Workspace &workspace) const;
        /// Collective transactional owned warm-guess restore: finite-value and
        /// lease preflight precedes every write. No halo transfer/mean mutation.
        /// Caller joins workspace work and supplies a distinct snapshot.
        void RestoreReconstructionGuess(Workspace &workspace, const EquationDofs &guess,
                                        const Runtime::Array<real> *limitedVRHistory = nullptr) const;
        std::optional<Runtime::Array<real>> LimitedVRHistory(const Workspace &workspace) const;
        std::optional<Runtime::Array<real>> LimitedVRAlpha(const Workspace &workspace) const;
        /// Local reset; caller coordinates failures and joins workspace work.
        void ResetReconstruction(Workspace &workspace) const;
        const EquationDofs &FluxReconstruction(const Workspace &workspace) const;
        /// Latest complete equation evaluation's global coefficient-update
        /// norm. No extra reduction occurs when reading these diagnostics.
        /// Fixed sweeps without a tolerance leave the update norm unmeasured
        /// (infinity); constant/gradient reconstruction reports zero.
        int LastReconstructionIterations(const Workspace &workspace) const;
        real LastReconstructionUpdate(const Workspace &workspace) const;
        /// Owned-only polynomial projection. Standalone use starts from the
        /// unlimited guess; Evaluate preserves its preceding WBAP result.
        /// No halo transfer occurs here. Caller serializes workspace/state.
        void LimitReconstructionPP(const EquationDofs &state, Workspace &workspace) const;
        /// Collective count/min snapshot; no implicit refresh or mutation.
        std::pair<index, real> PositivityReconstructionStats(const Workspace &workspace) const;
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
        /// match Workspace. Two-equation closures require a current
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
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                              EquationAuxiliaryInputs inputs = {}) const;
        void ValidateAuxValues(const EquationDofs &state, const AuxiliaryValues &values, real time = 0) const;
        /// Copyable immutable read scope for RHS geometry/basis data. Apply is
        /// stateless: no MPI, reconstruction, cache update or hidden counters.
        /// Raw input/output views require retained/declared Array leases and
        /// nonoverlapping output storage supplied by the caller. Core input
        /// views need only owned data. The checked Evaluate wrapper does this.
        class RHSOperator
        {
            struct ReadData;
            enum class FailurePolicy
            {
                Throw,
                Abort
            };
            std::shared_ptr<const ReadData> _read;
            explicit RHSOperator(std::shared_ptr<const ReadData> read) : _read(std::move(read)) {}
            template <FailurePolicy policy>
            RHSKernelDiagnostics ApplyFluxImpl(const RHSReadViews &input, const RHSWriteViews &output,
                                               CFV::ReconstructionRegion region, real time) const;
            template <FailurePolicy policy>
            void AddSourceImpl(const RHSReadViews &input, const RHSWriteViews &output,
                               CFV::ReconstructionRegion region, real time) const;
            friend class FiniteVolume;

        public:
            /// Checked synchronous borrowed-view entry point. The caller keeps
            /// every view lease and owner alive through return. Validation
            /// failures throw before or during this standalone kernel call.
            RHSKernelDiagnostics ApplyBorrowedChecked(const RHSReadViews &input, const RHSWriteViews &output,
                                                      CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Resolved production-kernel entry point. Its caller has already
            /// validated and retained the operation scope. A failed invariant
            /// terminates this rank directly; it never unwinds a collectively
            /// ordered worker path or calls MPI from that worker.
            RHSKernelDiagnostics ApplyBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                    CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Production flux-only leaf used by the unified FV orchestration;
            /// volume sources remain a separate equation operator. Kernel
            /// invariant and disabled-fallback failures abort directly.
            RHSKernelDiagnostics ApplyFluxBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                        CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Compatibility spelling for the checked borrowed-view entry.
            RHSKernelDiagnostics Apply(const RHSReadViews &input, const RHSWriteViews &output,
                                       CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Overwrite selected residual rows with integrated face flux only.
            /// Optional faceFlux follows the documented canonical face order.
            RHSKernelDiagnostics ApplyFlux(const RHSReadViews &input, const RHSWriteViews &output,
                                           CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
            /// Add owned-cell volume sources (flow5: rigid-frame source) to
            /// initialized output rows. No face, reconstruction or MPI work.
            void AddSource(const RHSReadViews &input, const RHSWriteViews &output,
                           CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0) const;
        };
        RHSOperator AcquireRHSOperator() const;
        /// Caller-owned local stencil-face table; no halo. Components follow
        /// logical equation order for whole-equation calls, flow5 for RHSOperator.
        Runtime::Array<real> BuildFaceFluxBuffer(index components) const;
        /// Reuse completed equation PrepareReconstruction/Evaluate publication.
        /// No halo exchange/reconstruction is hidden here. Input halos and all
        /// conflicting views must be ready/released. Flux overwrites selected
        /// owned rows; source adds to initialized owned rows. Output is distinct
        /// from state and workspace arrays. A face buffer is optional and follows
        /// AuxiliaryFaces() with the same orientation as RHSWriteViews::faceFlux.
        /// Caller serializes workspace use and coordinates distributed failures.
        void EvaluateFlux(const EquationDofs &state, EquationDofs &residual, Workspace &workspace,
                          CFV::ReconstructionRegion region = CFV::ReconstructionRegion::All, real time = 0,
                          Runtime::Array<real> *faceFlux = nullptr, const AuxiliaryValues *auxiliary = nullptr) const;
        void AddSource(const EquationDofs &state, EquationDofs &residual, Workspace &workspace, real time = 0) const;
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
        /// ALE/rotation terms, mean-state chemical source derivatives and scalar
        /// molecular/species diffusion bounds. This is not a
        /// derivative of the complete reconstructed viscous/source residual.
        /// Roe characteristics are not connected for this profile and reject.
        /// Frozen evaluated blocks retain numeric snapshots/geometry only;
        /// they perform no EOS calls, MPI or live-state/cache access.
        /// The caller completes halos, serializes matrix/input bindings and
        /// coordinates failures. No inverses/factors are updated implicitly.
        /// Output writers are reserved together; partial failure poisons them.
        /// RANS/source publication requires sourceReconstruction
        /// from a successful Evaluate at this state/time. Values/coefficients
        /// must remain unchanged. RANS derivatives use its current center
        /// gradients even on P0; chemical derivatives use the cell mean.
        void PrepareJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary,
                             const Runtime::Array<real> &inversePseudoTime, LinearProblemContext &output,
                             JacobianPreparation controls = {}, real time = 0,
                             std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                             const Workspace *sourceReconstruction = nullptr) const;
        FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation);
        bool SharesPlan(const FiniteVolume &other) const { return _data && _data == other._data; }
        const CFV::FiniteVolumePlan &Metrics() const;
        /// Bind through the metric space with this domain's explicit MPI
        /// launch policy. Setup remains collective in either policy.
        Runtime::TransferBinding BindHalo(CFV::FVField &field) const;
        const EquationSystem &Equation() const;
        /// Immutable resolved spatial controls, including source-free settings
        /// for a split Domain. Custom controllers copy this for their own
        /// nonlinear/hierarchy controls; changing a copy cannot mutate this plan.
        const DomainSettings &Settings() const;
        const CFV::VariationalReconstructionPlan &Reconstruction() const;
        const Geom::UnstructuredMesh &Mesh() const;
        const RigidMotion &Motion() const;
        const std::vector<BoundaryPoint> &BoundaryPoints() const;
        /// Maintained face-center rows, in faceFlow order; quadrature=-1.
        const std::vector<BoundaryPoint> &BoundaryFaces() const;
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
        /// Equation-orchestration entry. The initial interface implementation
        /// is available for the operator-local perfect-gas flow5 specialization;
        /// extended physical systems fail explicitly until their trace kernels
        /// are supplied.
        void SetBoundaryInputs(Workspace &workspace, const BoundaryInputs &input) const;
        void SetBoundaryInputs(MeanWorkspace &workspace, const BoundaryInputs &input) const;
        /// Allocate the requested members once. Cell reductions without matching
        /// face outputs also need private face scratch. Callers may instead
        /// supply compatible buffers themselves, including strided tables;
        /// BeginAuxValues lazily allocates their missing private scratch once.
        AuxiliaryValues BuildAuxiliaryValues(AuxiliaryRequest request) const;
        /// Add every auxiliary channel required by the configured RHS flux to
        /// a caller's explicit request. BuildAuxiliaryValues remains an exact
        /// allocator; RHS callers use this union so an independently requested
        /// spectral/dTau bundle cannot omit H-correction or pressure-stencil
        /// inputs consumed by the same numerical operator.
        AuxiliaryRequest CompleteRHSAuxiliaryRequest(AuxiliaryRequest request = {}) const;
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
        /// Interpretation of stored polynomial rows; independent of physical
        /// flow schema. Restart may restore means and cold-start incompatible
        /// coefficients rather than matching a C++ static row type.
        Json ReconstructionMetadata() const;
        /// Collective Euler compress=1 bound from dt*scale*meanResidual/volume.
        /// Inputs are owned-only CPU flow5 fields; state ghost writes may be
        /// active. No input mutation, time advance, growth cap or communication
        /// of payloads. A zero returned dt means no positive admissible bound.
        /// All ranks enter with matching finite positive dt/scale, relax in
        /// (0,1], and positive compressionPower. Mean residual must be ready
        /// and nonaliased with state. Uses the equation's thermal energy bounds.
        PhysicalStepLimit LimitPhysicalStep(const CFV::FVField &state, const CFV::FVField &meanResidual,
                                            real dt, real relax, real scale, int compressionPower = 1) const;
        /// Whole-equation compress=1 bound with logical, strided owned inputs.
        /// Caller supplies the matching model's frozen thermal policy and
        /// retains its coefficient buffers. Species move with flow along the
        /// thermal line; RANS positivity adds no separate dt ratio. Inputs must
        /// be ready, distinct and admissible. No EOS, mutation or payload MPI.
        PhysicalStepLimit LimitPhysicalStep(const EquationDofs &state, const EquationDofs &meanResidual,
                                            const EquationThermalView &thermal, real dt, real relax, real scale, int compressionPower = 1) const;
        /// Initialize the logical flow block from a cell-averaged analytic
        /// profile. The current producer is available for pure perfect-gas
        /// flow5 states; extended states use InitializeConstant/InitializeField.
        void Initialize(EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive) const;
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
        std::array<FlowState, 3> Error(const EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive, real time) const;
        std::array<FlowState, 3> Error(const CFV::FVField &state, const FlowProfileSettings &profile, const FlowState &primitive, real time) const;
        real StableStep(const CFV::FVField &state, real cfl) const;
        /// Collective owned-only admissibility; never repairs a state silently.
        bool Admissible(const EquationDofs &state) const;
        bool Admissible(const CFV::FVField &state) const;
        index LastFaceFallbacks(const Workspace &workspace) const;
        /// Full-state variant also refreshes sampled source inputs from the
        /// retained polynomial before publishing their Jacobian provenance.
        /// PrepareJacobian separately evaluates the mean-state source matrix;
        /// this call evaluates neither chemical rates nor the physical RHS.
        void PrepareCorrectionAux(EquationDofs &state, Workspace &workspace, AuxiliaryValues &auxiliary,
                                  Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const;
        const ThermalBounds &ThermalConstraints() const;
        /// Collective checked projection of a fresh copy of the unlimited
        /// guess into flux coefficients, retaining the unlimited guess. Reads only
        /// owned means and cached owned-cell points; a mean ghost write may be
        /// active. Coefficient transfers must be complete before publication.
        /// Explicit outputs are limited coefficients and per-call beta stats.
        /// Does not reconstruct, exchange, advance time or change cell means.
        index LastCharacteristicFallbacks(const Workspace &workspace) const;
        /// Flow5 explicit conservative-blend compatibility. This is an
        /// operator-local specialization; generic equations do not expose a
        /// separately constructed low-order residual yet.
        const EquationDofs &LowOrderResidual(const Workspace &workspace) const;
        /// Collective sum of the last completed equation RHS's wall momentum
        /// flux, in mesh axes, with the same sign as pressure*n - traction.
        /// Per-quadrature outputs include the actual numerical pressure, RANS
        /// stress and viscous traction; no separate EOS/flux recomputation.
        /// Workspace access is controller-serialized; do not read during Evaluate.
        Vector3 LastBoundaryForce(const Workspace &workspace, const std::vector<int> &zones) const;
        /// All conservative blocks plus mean thermodynamic/species fields and
        /// reconstructed boundary traces. Reads an observation workspace only;
        /// does not modify the production reconstruction or any state values.
        void WriteVisualization(const EquationDofs &state, const Workspace &workspace,
                                const std::string &prefix, real time, int hdfDeflateLevel,
                                const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const;

    private:
        ReconstructionRestore ReserveReconstructionRestoreImpl(Workspace &workspace, const EquationDofs &guess,
                                                               const Runtime::Array<real> *limitedVRHistory,
                                                               std::shared_ptr<Workspace> retained, bool collectiveChecks = true) const;
        std::shared_ptr<Workspace> BuildWorkspaceImpl(EquationDofs &state,
                                                      std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                      bool meanOnly, WorkspaceOptions options = WorkspaceOptions()) const;
        void EvaluateImpl(EquationDofs &state, EquationDofs *residual, Workspace &workspace,
                          Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time,
                          AuxiliaryValues *auxiliary = nullptr) const;
        void ValidateEquationSourceJacobian(const EquationDofs &state, const Workspace &workspace, real time) const;
        bool AddEquationSourceJacobian(const EquationDofs &state, const Workspace &workspace,
                                       const Runtime::Array<real>::WriteLease &diagonal, DiagonalStorage storage,
                                       real scale, EquationPointEvaluator &evaluator,
                                       const EquationFVDetail::MeanLinearizationOutput *linearization = nullptr) const;
        index PrepareFaceCharacteristics(const CFV::FVField &state, CFV::FaceCharacteristicTransform &transform,
                                         index firstComponent = 0) const;
    };
}
