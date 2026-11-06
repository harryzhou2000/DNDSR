#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/CoordinatedSchedule.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"

namespace DNDS::Cauchy
{
    struct ResolvedPhysicalInputs;
    /// One participating domain. Placement/context ownership belongs to the
    /// simulation; this object owns only local mesh, equation, fields and
    /// numerical workspace. Simulation coordinates its accepted clock;
    /// standalone Domain stepping is the coarse Python/debugging frontend.
    /// Public mutation/stepping is serialized by the caller. Future physical
    /// motion and coupled stage/implicit drivers use the same accepted/trial
    /// state separation and explicit domain-local residual boundary.
    class Domain
    {
        friend class Simulation;
        std::string _name;
        EquationSystem _equation;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> _workers;
        FiniteVolume _fv;
        EquationDofs _accepted, _trial;
        EquationDofs _residualState;
        EquationDofs _initial;
        std::shared_ptr<FiniteVolume::Workspace> _workspace;
        std::shared_ptr<FiniteVolume::EquationWorkspace> _equationWorkspace, _equationObservationWorkspace;
        std::optional<EquationDofs> _equationBackup, _observationState;
        std::optional<EquationThermalPolicy> _equationThermal;
        std::optional<RANSStateView> _ransState;
        std::shared_ptr<FiniteVolume::Workspace> _observationWorkspace;
        std::optional<CFV::FVField> _observationResidual;
        /// Logical-const diagnostics may enqueue work on the domain stream.
        mutable Runtime::EagerEngine _engine;
        Runtime::ExecutionStream _stream;
        std::unique_ptr<NonlinearSolver> _implicit;
        std::unique_ptr<EquationNonlinearSolver> _equationImplicit;
        std::unique_ptr<PolynomialMultigrid> _multigrid;
        std::unique_ptr<EquationTimeStepper> _timeStepper;
        std::vector<EquationNonlinearIterationReport> _lastImplicit;
        index _lastCoupledIterations = 0;
        real _lastCoupledTargetRatio = 0;
        std::vector<index> _ditrComponents;
        ImplicitSettings _implicitControls;
        real _coupledReconstructionTolerance = 0;
        struct CoupledStageOperator
        {
            EquationTimeStepper::Evaluate evaluate;
            std::function<EquationNonlinearIterationReport(const ODE::CoupledStageEquation<EquationDofs> &, const EquationDofs &, real)> correct;
            std::function<std::pair<int, real>()> reconstruction;
            std::function<void()> reset;
        };
        std::function<CoupledStageOperator(EquationDofs &)> _coupledFactory;
        std::optional<CoupledStageOperator> _coupledMiddle;
        Runtime::ArrayStorage::Shared _coupledMiddleOwner;
        std::optional<EquationDofs> _coupledResidual;
        std::optional<Runtime::Array<real>> _reconstructionBackup;
        std::optional<Runtime::Array<real>> _limitedVRBackup;
        bool _restoreReconstruction = false;
        mutable std::optional<AuxiliaryValues> _timeAuxiliary;
        mutable std::optional<EquationMeanGradients> _timeMeanGradients;
        mutable std::vector<Runtime::TransferBinding> _timeMeanHalo;
        PhysicalTimePositivitySettings _timePP;
        int _timeCompressionPower = 1;
        std::shared_ptr<FiniteVolume::MeanWorkspace> _timePPWorkspace;
        std::optional<CFV::FVField> _timePPResidual;
        std::shared_ptr<FiniteVolume::EquationMeanWorkspace> _timeEquationPPWorkspace;
        std::optional<EquationDofs> _timeEquationPPResidual;
        StateEpoch _epoch;
        StateEpoch _trialEpoch;
        uint64_t _nextTrial = 0;
        real _time = 0, _trialTime = 0, _trialDt = 0, _lastAcceptedDt = 0;
        bool _trialReady = false;
        index _lastFallbacks = 0;
        bool _positivity = false, _steady = false, _rejectUnconvergedStages = false;
        bool _sourceStrang = false;
        Physics::Chemistry::ReactorStepSettings _reactorControls;
        real _lastAlpha = 1, _lastBeta = 1;
        index _lastAlphaCells = 0, _lastBetaCells = 0;
        index _lastCharacteristicFallbacks = 0;
        void UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime);
        void ChemicalHalfStep(const EquationDofs &input, EquationDofs &output, real dt);
        void FinishChemicalStep();
        void PublishTrial(real time, StateEpoch epoch);
        void EvaluateObservation();
        void InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh);
        void ConfigureTimeStep(const TimeSettings &time, int compressionPower);
        void PrepareImplicitStep(real dt);
        void SetTimeStage(int stage);
        void PreserveReconstructionGuess();
        void EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::Workspace &workspace, real time, int stage);
        void EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::EquationWorkspace &workspace, real time, int stage);
        void NormalizeRate(EquationDofs &rate);
        bool RecordImplicitStage(NonlinearIterationReport report);
        bool RecordImplicitStage(EquationNonlinearIterationReport report);
        static EquationNonlinearIterationReport AsEquationReport(NonlinearIterationReport report);
        bool Admissible(const EquationDofs &state) const;
        static CFV::FVField &FlowField(EquationDofs &state);
        static const CFV::FVField &FlowField(const EquationDofs &state);
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time, EquationSetup setup);
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const ResolvedPhysicalInputs &inputs, const TimeSettings &time, EquationSetup setup);
        void BeginTrial(real dt);
        /// Simulation alone interleaves these serialized domain-local phases.
        /// Boundary exchanges happen between preparation and solve, outside
        /// every domain's inner nonlinear loop and MPI communicator.
        void BeginCoordinatedStep(real dt);
        real PrepareCoordinatedStage(int stage);
        void SolveCoordinatedStage();
        void CompleteCoordinatedStage();
        int CoordinatedStageNumber() const;
        std::optional<EquationTimeStepper::StageEquation> _stageEquation;
        int _coordinatedStage = 0;
        bool _coordinatedPrepared = false;
        std::optional<BoundaryInputs> _interfaceInputs;
        void BindInterfaceInputs(const BoundaryInputs &input);
        void CommitReserved(EquationDofs::WriteScope &destination, EquationDofs::ReadScope &source);
        /// Detached checkpoint candidate. Reads may use trial scratch but
        /// accepted means/history/guesses change only after all reservations.
        struct CheckpointRead
        {
            real time = 0, lastDt = 0;
            StateEpoch epoch;
            std::unique_ptr<EquationTimeStepper> stepper;
            std::optional<Runtime::Array<real>> guess, limitedVRHistory;
            std::optional<EquationDofs> equationGuess;
            std::optional<EquationDofs::ReadScope> source;
            std::optional<EquationDofs::WriteScope> destination;
            std::vector<EquationBlockArrayView<const real>> input;
            std::vector<EquationBlockArrayView<real>> output;
            std::optional<FiniteVolume::ReconstructionRestore> reconstruction;
            bool committed = false;
            ~CheckpointRead()
            {
                if (!committed && destination)
                    destination->CancelUnmodified();
            }
        };
        std::unique_ptr<CheckpointRead> PrepareCheckpointRead(const std::string &path, const Json &resolved);
        void ReserveCheckpointRead(CheckpointRead &checkpoint, bool collectiveChecks = true);
        void CommitCheckpointRead(CheckpointRead &checkpoint);

    public:
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time = {});
        const std::string &Name() const { return _name; }
        const EquationSystem &Equation() const { return _equation; }
        const FiniteVolume &Discretization() const { return _fv; }
        /// Shared worker capacity for standalone native operators. A copy
        /// retains the pool, never an acquired worker. Serialize operator and
        /// Domain use or provide sufficient capacity; no implicit pool exists.
        std::optional<Physics::Chemistry::ChemicalWorkerPool> ChemicalWorkers() const { return _workers; }
        const Algebra::DofBundle &State() const { return _accepted.Fields(); }
        const EquationDofs &Dofs() const { return _accepted; }
        /// Synchronous local point refresh using this domain's explicit worker
        /// capacity. No halo exchange; caller ensures ready ghosts when used.
        bool RefreshDerived(DerivedFields &fields, bool onlyIfStale = false) const;
        /// Coarse debugging/input boundary. Release writers before a trial and
        /// all views before commit; aliases intentionally block conflicting writes.
        /// Legacy/debug convenience for a dedicated physical flow5 field.
        /// Coupled extended tables use Dofs().ReadOwned().Block("flow") instead.
        CFV::FVField &Flow() { return FlowField(_accepted); }
        const CFV::FVField &Flow() const { return FlowField(_accepted); }
        real Time() const { return _time; }
        /// Actual last committed physical step, including retry halving. Zero
        /// for steady/fresh state or an older restart without this information.
        real PreviousTimeStep() const { return _lastAcceptedDt; }
        bool Steady() const { return _steady; }
        /// Actual domain policy: inert domains remain unsplit even when the
        /// simulation selects Strang for its chemical domains.
        bool UsesStrangSource() const { return _sourceStrang; }
        const StateEpoch &Epoch() const { return _epoch; }
        const std::vector<EquationNonlinearIterationReport> &LastImplicitStages() const { return _lastImplicit; }
        index LastCoupledIterations() const { return _lastCoupledIterations; }
        /// Shared nonlinear policy's selected-component ratio for I+H after
        /// the two DITR updates. Identical to its single solve report's ratio;
        /// <=1 passes the residual check. Zero for other time methods.
        real LastCoupledTargetRatio() const { return _lastCoupledTargetRatio; }
        index LastFaceFallbacks() const { return _lastFallbacks; }
        std::array<real, 5> LastPositivity() const { return {real(_lastBetaCells), _lastBeta, real(_lastAlphaCells), _lastAlpha, real(_lastCharacteristicFallbacks)}; }
        real StableStep(real cfl) const;
        /// Collective physical-step selection before PrepareStep/Advance.
        /// Uses separate zero-reconstruction scratch, then Euler's growth cap.
        /// Does not mutate means, accepted VR guess, clock or time history.
        /// All ranks supply the same finite positive requested step and release
        /// conflicting views first. Caller advances with the returned dt;
        /// Advance never silently changes its dt. Steady mode bypasses PP.
        PhysicalStepLimit LimitPhysicalStep(real requestedDt);
        /// Build the configured time-method candidate at this accepted time. Domain ranks enter
        /// together. Inadmissible stages fail collectively without changing the
        /// accepted field/clock. dt may be rejected by the caller and retried.
        /// Explicit SSPRK never compresses a per-cell flow5/species increment.
        /// Separate RANS exponential repair/mean bounds remain permitted. The
        /// flow5 path may use one conservative global flux blend; otherwise
        /// inadmissible stages reject for whole-step smaller-dt retry. Discard
        /// a failed candidate before retry; stages restart at accepted time.
        /// time.steady instead uses BDF1's infinite-dt limit: finite dt is ignored,
        /// and physical time is frozen. By default an iteration cap advances
        /// the final admissible iterate like Euler; LastImplicitStages retains
        /// its actual convergence result. implicit.rejectUnconvergedStages
        /// enables rejection on inner/FAS/coupled-stage caps for steady or transient.
        /// Physical Strang applies constant-volume species half-steps around
        /// the same transport driver, retaining VR and resetting method history
        /// across source jumps. LastImplicitStages reports transport solves.
        /// Only StepRejected certifies an agreed numerical retry. Unexpected
        /// backend/access/allocation exceptions must reach a fail-fast MPI
        /// boundary; catching and continuing to MPI/finalization is unsafe.
        void PrepareStep(real dt);
        void CommitStep();
        void DiscardStep() noexcept
        {
            _trialReady = false;
            _coordinatedPrepared = false;
            _stageEquation.reset();
            if (_timeStepper)
                _timeStepper->Discard();
        }
        void Advance(real dt)
        {
            PrepareStep(dt);
            CommitStep();
        }
        /// Recompute an owned integrated residual for inspection. The returned
        /// workspace is overwritten at the next residual/stage; share/copy
        /// explicitly if it must outlive that operation. Invalidates any
        /// prepared trial because the reconstruction/trial workspace is reused.
        /// Requires a dedicated physical flow5 field, like Flow(). Use
        /// ResidualState() for coupled extended-equation allocations.
        const CFV::FVField &Residual();
        /// Full logical integrated residual; shares the same overwriteable
        /// workspace and candidate-invalidating semantics as Residual().
        const EquationDofs &ResidualState();
        /// Collective diagnostics of the accepted state. Recomputes its RHS;
        /// no state/time update. Returns mass, energy, rho/p minima, residual L2,
        /// force xyz, VR sweeps/update, density L1/L2/Linf analytic errors, Tmin.
        std::array<real, 14> Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones);
        void WriteVisualization(const std::string &prefix, int hdfDeflateLevel = 0);
        /// Native parallel restart with persistent cell IDs and per-cell mesh
        /// validation. Default omits reconstruction; opt in for compatible
        /// fixed-sweep continuation. BDF history is stored when applicable.
        /// Reads may change rank distribution/order or polynomial degree. Means
        /// and compatible history are remapped by exact canonical cell IDs;
        /// missing/incompatible polynomial guesses cold-start at zero. Active
        /// LVR stores its retained shock/contact history with an opted-in warm
        /// guess; both restore together or cold-start together.
        /// Extended equation archives retain all physical fields and their
        /// logical block map, including species history. Native layouts are
        /// preserved on write and converted on read. Derived fields are omitted.
        void WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction = false) const;
        void ReadCheckpoint(const std::string &path, const Json &resolved);
    };
}
