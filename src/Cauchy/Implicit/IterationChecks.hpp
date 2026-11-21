#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    /// Ordered first numerical failure in a speculative nonlinear update.
    /// Device and CPU producers write on the domain compute lane. No atomics
    /// are needed between phases. The controller reads only at a stop gate.
    struct IterationFailure
    {
        NumericalStatus status{};
        index iteration = 0;
        int stage = 0, phase = 0;
    };
    enum class IterationPhase
    {
        PseudoTime,
        Jacobian,
        Inverse,
        Linear,
        Correction,
        Candidate,
        Flux,
        Source,
        Defect
    };
    class IterationChecks
    {
        Runtime::Array<IterationFailure> _first;
        Runtime::ExecutionStream _stream;
        index _iteration = 0;
        int _stage = 0;
        bool _pending = false;
        std::vector<std::function<void()>> _publications;

    public:
        explicit IterationChecks(Runtime::ExecutionStream);
        void Begin(index iteration, int stage);
        /// Latch the first failing local operation. The scalar producer can
        /// immediately reuse its own scratch on this lane.
        void Capture(IterationPhase, const Runtime::Array<NumericalStatus> &);
        void CaptureHost(IterationPhase, std::function<NumericalStatus()>);
        void CaptureFinite(IterationPhase, const Runtime::Array<real> &);
        void CaptureInvalidRow(IterationPhase, const Runtime::Array<index> &);
        void CopyIfValid(const Runtime::Array<real> &, Runtime::Array<real> &);
        /// Retire every submitted check before deciding whether to use a trial.
        void PublishAfterObservation(std::function<void()> publish) { _publications.push_back(std::move(publish)); }
        void Observe();
    };
    namespace CUDA
    {
        using FailureView = Runtime::ArrayView<IterationFailure>;
        void LatchIterationFailure(Runtime::ExecutionContext, FailureView, IterationFailure, Runtime::ArrayView<const NumericalStatus>, Runtime::ArrayView<const real>, Runtime::ArrayView<const index> invalidRow = {});
        void CopyValidIterationRows(Runtime::ExecutionContext, Runtime::ArrayView<const IterationFailure>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>);
    }
}
