#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/CoordinatedSchedule.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Solver/FieldTimeStepper.hpp"

namespace DNDS::Cauchy
{
    /// One participating domain. Placement/context ownership belongs to the
    /// simulation; this object owns only local mesh, equation, fields and
    /// numerical workspace. Simulation coordinates its accepted clock;
    /// standalone Domain stepping is the coarse Python/debugging frontend.
    /// Public mutation/stepping is serialized by the caller. Future physical
    /// motion and coupled stage/implicit drivers use the same accepted/trial
    /// state separation and explicit domain-local residual boundary.
    class Domain
    {
        std::string _name;
        EquationSystem _equation;
        FiniteVolume _fv;
        Algebra::DofBundle _accepted, _trial;
        CFV::FVField _residual, _initial;
        std::shared_ptr<FiniteVolume::Workspace> _workspace;
        std::shared_ptr<FiniteVolume::Workspace> _observationWorkspace;
        std::optional<CFV::FVField> _observationResidual;
        Runtime::EagerEngine _engine;
        Runtime::ExecutionStream _stream;
        std::unique_ptr<NonlinearSolver> _implicit;
        std::unique_ptr<PolynomialMultigrid> _multigrid;
        std::unique_ptr<ODE::FieldTimeStepper> _timeStepper;
        std::vector<NonlinearIterationReport> _lastImplicit;
        std::optional<Runtime::Array<real>> _reconstructionBackup;
        bool _restoreReconstruction = false;
        mutable std::optional<AuxiliaryValues> _timeAuxiliary;
        mutable std::optional<Runtime::TransferBinding> _timeMeanHalo;
        StateEpoch _epoch;
        StateEpoch _trialEpoch;
        uint64_t _nextTrial = 0;
        real _time = 0, _trialTime = 0;
        bool _trialReady = false;
        index _lastFallbacks = 0;
        bool _positivity = false;
        real _lastAlpha = 1, _lastBeta = 1;
        index _lastAlphaCells = 0, _lastBetaCells = 0;
        index _lastCharacteristicFallbacks = 0;
        void UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime);
        void PublishTrial(real time, StateEpoch epoch, const Runtime::Array<real> *reconstructionGuess = nullptr);
        void EvaluateObservation();
        void InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh);
        void PrepareImplicitStep(real dt);
        void PreserveReconstructionGuess();

    public:
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time = {});
        const std::string &Name() const { return _name; }
        const EquationSystem &Equation() const { return _equation; }
        const FiniteVolume &Discretization() const { return _fv; }
        const Algebra::DofBundle &State() const { return _accepted; }
        /// Coarse debugging/input boundary. Release writers before a trial and
        /// all views before commit; aliases intentionally block conflicting writes.
        CFV::FVField &Flow() { return _accepted.Block("flow"); }
        const CFV::FVField &Flow() const { return _accepted.Block("flow"); }
        real Time() const { return _time; }
        const StateEpoch &Epoch() const { return _epoch; }
        const std::vector<NonlinearIterationReport> &LastImplicitStages() const { return _lastImplicit; }
        index LastFaceFallbacks() const { return _lastFallbacks; }
        std::array<real, 5> LastPositivity() const { return {real(_lastBetaCells), _lastBeta, real(_lastAlphaCells), _lastAlpha, real(_lastCharacteristicFallbacks)}; }
        real StableStep(real cfl) const;
        /// Build the configured time-method candidate at this accepted time. Domain ranks enter
        /// together. Inadmissible stages fail collectively without changing the
        /// accepted field/clock. dt may be rejected by the caller and retried.
        void PrepareStep(real dt);
        void CommitStep();
        void DiscardStep() noexcept
        {
            _trialReady = false;
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
        const CFV::FVField &Residual();
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
        /// missing/incompatible polynomial guesses cold-start at zero.
        void WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction = false) const;
        void ReadCheckpoint(const std::string &path, const Json &resolved);
    };
}
