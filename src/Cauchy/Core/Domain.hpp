#pragma once

#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/Core/DomainComponents.hpp"
#include "Cauchy/Core/CoordinatedSchedule.hpp"
#include "Cauchy/Core/DomainStateSnapshot.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/IO/CheckpointTransaction.hpp"

namespace DNDS::Cauchy
{
    struct ResolvedPhysicalInputs;
    class DomainController;
    /// One participating domain. Placement/context ownership belongs to the
    /// simulation; this object owns only local mesh, equation, fields and
    /// numerical workspace. Simulation coordinates its accepted clock;
    /// standalone Domain stepping is the coarse Python/debugging frontend.
    /// Public mutation/stepping is serialized by the caller. Future physical
    /// motion and coupled stage/implicit drivers use the same accepted/trial
    /// state separation and explicit domain-local residual boundary.
    class Domain
    {
        friend class DomainController;
        friend class DomainStateSnapshot;
        DomainProblem _problem;
        DomainState _state;
        DomainExecution _execution;
        std::vector<NonlinearIterationReport> _lastImplicit;
        std::function<void(const NonlinearProgress &)> _progressObserver;
        std::function<void(const NonlinearSolveState &)> _quiescentObserver;
        index _lastCoupledIterations = 0;
        real _lastCoupledTargetRatio = 0;
        std::vector<index> _ditrComponents;
        ImplicitSettings _implicitControls;
        real _coupledReconstructionTolerance = 0;
        std::optional<DomainCoupledStageOperator> _coupledMiddle;
        PhysicalTimePositivitySettings _timePP;
        int _timeCompressionPower = 1;
        index _lastFallbacks = 0;
        bool _positivity = false, _steady = false, _rejectUnconvergedStages = false;
        bool _sourceStrang = false;
        Physics::Chemistry::ReactorStepSettings _reactorControls;
        real _lastAlpha = 1, _lastBeta = 1;
        index _lastAlphaCells = 0, _lastBetaCells = 0;
        index _lastCharacteristicFallbacks = 0;
        StepResult UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime);
        void ChemicalHalfStep(const EquationDofs &input, EquationDofs &output, real dt);
        void FinishChemicalStep();
        void PublishTrial(real time, StateEpoch epoch);
        void EvaluateObservation(const DomainStateSnapshot &snapshot);
        void InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh);
        void ConfigureTimeStep(const TimeSettings &time, int compressionPower);
        StepResult PrepareImplicitStep(real dt);
        StepResult ResumeImplicitStep();
        void PrepareCoupledImplicit(EquationTimeStepper::DITR &method,
                                    bool resume);
        bool SolveCoupledImplicit(EquationTimeStepper::DITR &method, EquationDofs &endpoint, bool resume);
        NonlinearSolveControl ProgressControl() const;
        void SetTimeStage(int stage);
        void PreserveReconstructionGuess();
        void EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::Workspace &workspace, real time, int stage);
        void NormalizeRate(EquationDofs &rate);
        bool RecordImplicitStage(NonlinearIterationReport report);
        Json UpdatePolicyMetadata() const;
        bool Admissible(const EquationDofs &state) const;
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time, EquationSetup setup);
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const ResolvedPhysicalInputs &inputs, const TimeSettings &time, EquationSetup setup);
        void BeginTrial(real dt);
        /// Simulation alone interleaves these serialized domain-local phases.
        /// Boundary exchanges happen between preparation and solve, outside
        /// every domain's inner nonlinear loop and MPI communicator.
        void BeginCoordinatedStep(real dt);
        real PrepareCoordinatedStage(int stage);
        StepResult SolveCoordinatedStage();
        StepResult CompleteCoordinatedStage();
        int CoordinatedStageNumber() const;
        bool CoordinatedStepIsCoupled() const;
        real CoupledStageTime(int coupledStage) const;
        void PrepareCoupledInterface(int coupledStage, bool reconstructed);
        FiniteVolume::InterfaceDonorPublication PublishCoupledInterfaceDonor(
            int coupledStage, bool reconstructed);
        void BindCoupledInterfaceInputs(int coupledStage,
                                        const BoundaryInputs &input);
        void ActivateCoupledInterface(int coupledStage);
        StepResult SolveCoordinatedCoupledStep();
        void BindInterfaceInputs(const BoundaryInputs &input);
        void CommitReserved(EquationDofs::WriteScope &destination, EquationDofs::ReadScope &source);
        std::unique_ptr<IO::DomainCheckpointRead> PrepareCheckpointRead(const std::string &path, const Json &resolved);
        void ReserveCheckpointRead(IO::DomainCheckpointRead &checkpoint, bool collectiveChecks = true);
        void CommitCheckpointRead(IO::DomainCheckpointRead &checkpoint);
        std::unique_ptr<IO::DomainInternalCheckpointRead> PrepareInternalCheckpointRead(const std::string &path, const Json &resolved);
        void CommitInternalCheckpointRead(IO::DomainInternalCheckpointRead &checkpoint);

    public:
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time = {});
        /// Construct a short-lived controller view over this domain. The
        /// controller owns no numerical resources and must not outlive Domain;
        /// it is the only orchestration API allowed to reserve or publish a
        /// coordinated trial candidate.
        DomainController Controller();
        /// Pin one accepted generation for synchronous or asynchronous output.
        /// The returned zero-copy snapshot blocks accepted-state publication
        /// until all copies are released.
        DomainStateSnapshot CaptureStateSnapshot(
            bool includeReconstruction = false) const;
        const std::string &Name() const { return _problem.name; }
        const EquationSystem &Equation() const { return _problem.equation; }
        const FiniteVolume &Discretization() const { return _problem.finiteVolume; }
        /// Shared worker capacity for standalone native operators. A copy
        /// retains the pool, never an acquired worker. Serialize operator and
        /// Domain use or provide sufficient capacity; no implicit pool exists.
        std::optional<Physics::Chemistry::ChemicalWorkerPool> ChemicalWorkers() const
        {
            return _execution.ChemicalWorkers();
        }
        EquationDofs &Dofs() { return _state.accepted; }
        const EquationDofs &Dofs() const { return _state.accepted; }
        /// Synchronous local point refresh using this domain's explicit worker
        /// capacity. No halo exchange; caller ensures ready ghosts when used.
        bool RefreshDerived(DerivedFields &fields, bool onlyIfStale = false) const;
        real Time() const { return _state.time; }
        real InternalObservationTime() const { return _state.progressTime; }
        /// Actual last committed physical step, including retry halving. Zero
        /// for steady/fresh state or an older restart without this information.
        real PreviousTimeStep() const { return _state.lastAcceptedDt; }
        bool Steady() const { return _steady; }
        /// Actual domain policy: inert domains remain unsplit even when the
        /// simulation selects Strang for its chemical domains.
        bool UsesStrangSource() const { return _sourceStrang; }
        const StateEpoch &Epoch() const { return _state.epoch; }
        const std::vector<NonlinearIterationReport> &LastImplicitStages() const { return _lastImplicit; }
        /// Synchronous sink-neutral nonlinear progress callback. The callback
        /// may buffer/report locally but must not enter unrelated collectives.
        void SetProgressObserver(std::function<void(const NonlinearProgress &)> observer)
        {
            _progressObserver = std::move(observer);
        }
        void SetQuiescentObserver(std::function<void(const NonlinearSolveState &)> observer)
        {
            _quiescentObserver = std::move(observer);
        }
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
        /// A rejected StepResult certifies an agreed numerical retry. Unexpected
        /// backend/access/allocation exceptions must reach a fail-fast MPI
        /// boundary; catching and continuing to MPI/finalization is unsafe.
        [[nodiscard]] StepResult PrepareStep(real dt);
        void CommitStep();
        void DiscardStep() noexcept
        {
            _state.trialReady = false;
            _state.internalResume = false;
            _state.coordinatedPrepared = false;
            _state.stageEquation.reset();
            if (_state.timeStepper)
                _state.timeStepper->Discard();
        }
        [[nodiscard]] StepResult Advance(real dt)
        {
            auto result = PrepareStep(dt);
            if (result)
                CommitStep();
            else
                DiscardStep();
            return result;
        }
        /// Recompute an owned integrated residual for inspection. The returned
        /// workspace is overwritten at the next residual/stage; share/copy
        /// explicitly if it must outlive that operation. Invalidates any
        /// prepared trial because the reconstruction/trial workspace is reused.
        /// Full logical integrated residual; shares overwriteable workspace
        /// and invalidates any prepared candidate.
        const EquationDofs &Residual();
        /// Collective diagnostics of the accepted state. Recomputes its RHS;
        /// no state/time update. Returns mass, energy, rho/p minima, residual L2,
        /// force xyz, VR sweeps/update, density L1/L2/Linf analytic errors, Tmin.
        std::array<real, 14> Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones);
        void WriteVisualization(const std::string &prefix, int hdfDeflateLevel = 0);
        /// Observe the current nonlinear trial at a synchronous progress
        /// boundary. This uses the stage's already-finalized reconstruction and
        /// never rebuilds or mutates the accepted observation workspace.
        void WriteInternalVisualization(const std::string &prefix, int hdfDeflateLevel = 0);
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
        void WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction = false,
                             const Json &outputController = Json()) const;
        /// Exact nonlinear-iteration snapshot. The caller invokes this only at
        /// the quiescent callback published by the nonlinear controller.
        void WriteInternalCheckpoint(const std::string &path, const Json &resolved,
                                     const NonlinearSolveState &state,
                                     const Json &outputController = Json()) const;
        void ReadCheckpoint(const std::string &path, const Json &resolved);
    };
}
