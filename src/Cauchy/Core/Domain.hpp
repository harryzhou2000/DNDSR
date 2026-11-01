#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/CoordinatedSchedule.hpp"

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
        Runtime::EagerEngine _engine;
        Runtime::ExecutionStream _stream;
        StateEpoch _epoch;
        StateEpoch _trialEpoch;
        uint64_t _nextTrial = 0;
        real _time = 0, _trialTime = 0;
        bool _trialReady = false;
        index _lastFallbacks = 0;
        void UpdateStage(real dt, real historyWeight, real stageWeight);
        void PublishTrial(real time, StateEpoch epoch);

    public:
        Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings);
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
        index LastFaceFallbacks() const { return _lastFallbacks; }
        real StableStep(real cfl) const { return _fv.StableStep(Flow(), cfl); }
        /// Build an SSPRK3 candidate at this accepted time. Domain ranks enter
        /// together. Inadmissible stages fail collectively without changing the
        /// accepted field/clock. dt may be rejected by the caller and retried.
        void PrepareStep(real dt);
        void CommitStep();
        void DiscardStep() noexcept { _trialReady = false; }
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
        /// Initial restart envelope: exact config, partition, persistent cell
        /// ordering, primary cell geometry/topology and boundary zones. Native
        /// Array I/O writes the flow allocation directly, preserving layouts.
        /// Per-cell mesh metadata is recorded separately for exact validation.
        /// Changed partitions/restart remapping belong to a later IO package.
        void WriteCheckpoint(const std::string &path, const Json &resolved) const;
        void ReadCheckpoint(const std::string &path, const Json &resolved);
    };
}
