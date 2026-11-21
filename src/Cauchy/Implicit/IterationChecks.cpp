#include "DNDS/Core/OMP.hpp"
#include "Cauchy/Implicit/IterationChecks.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include <fmt/format.h>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    IterationChecks::IterationChecks(R::ExecutionStream lane)
        : _first(R::RowLayout::Dense(1, 1), lane.Context().device), _stream(std::move(lane))
    {
        _first.SetName("nonlinear/checks/firstFailure", DNDS_ALLOCATION_SITE());
    }
    bool IterationChecks::CheckHost(OptionalRef<IterationChecks> checks, IterationPhase phase, NumericalStatus status, const char *operation)
    {
        if (status)
            return true;
        if (checks)
            checks->CaptureHost(phase, [status]
                                { return status; });
        else
            RequireNumericalSuccess(status, operation);
        return false;
    }
    void IterationChecks::Begin(index iteration, int stage)
    {
        _stage = stage;
        _iteration = iteration;
        // Runtime allocations are initially zero (NumericalError::None).
        // A successful observation leaves the success sentinel intact.
        _pending = true;
    }
    void IterationChecks::Capture(IterationPhase phase, const R::Array<NumericalStatus> &status)
    {
        auto engine = _stream.LockEngine();
        IterationFailure tag{{}, _iteration, _stage, int(phase)};
        R::SubmitArrays(engine, _stream, [tag](auto where, auto src, auto dst) mutable
                        {
                            if (where.device == R::Device::Host())
                            {
                                tag.status = src(0, 0);
                                if (dst(0, 0).status && !tag.status)
                                    dst(0, 0) = tag;
                            }
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::LatchIterationFailure(where, dst.DynamicView(), tag, src.DynamicView(), {});
#endif
                        },
                        R::Read(status), R::ReadWrite(_first));
    }
    void IterationChecks::CaptureHost(IterationPhase phase, std::function<NumericalStatus()> read)
    {
        DNDS_check_throw_info(_first.GetDevice() == R::Device::Host(), "host numerical status on a device lane");
        auto engine = _stream.LockEngine();
        IterationFailure tag{{}, _iteration, _stage, int(phase)};
        R::SubmitArrays(engine, _stream, [tag, read = std::move(read)](auto, auto dst) mutable
                        {
            tag.status = read();
            if (dst(0,0).status && !tag.status) dst(0,0) = tag; }, R::ReadWrite(_first));
    }
    void IterationChecks::CaptureFinite(IterationPhase phase, const R::Array<real> &finite)
    {
        auto engine = _stream.LockEngine();
        IterationFailure tag{{NumericalError::InvalidState}, _iteration, _stage, int(phase)};
        R::SubmitArrays(engine, _stream, [tag](auto where, auto src, auto dst)
                        {
                            if (where.device == R::Device::Host())
                            {
                                if (dst(0, 0).status && src(0, 0) != 1)
                                    dst(0, 0) = tag;
                            }
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::LatchIterationFailure(where, dst.DynamicView(), tag, {}, src.DynamicView());
#endif
                        },
                        R::Read(finite), R::ReadWrite(_first));
    }
    void IterationChecks::CaptureInvalidRow(IterationPhase phase, const R::Array<index> &invalidRow)
    {
        auto engine = _stream.LockEngine();
        IterationFailure tag{{NumericalError::InvalidState, 1}, _iteration, _stage, int(phase)};
        R::SubmitArrays(engine, _stream, [tag](auto where, auto src, auto dst) mutable
                        {
                            if (where.device == R::Device::Host())
                            {
                                tag.status.row = src(0, 0);
                                if (dst(0, 0).status && tag.status.row >= 0)
                                    dst(0, 0) = tag;
                            }
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::LatchIterationFailure(where, dst.DynamicView(), tag, {}, {}, src.DynamicView());
#endif
                        },
                        R::Read(invalidRow), R::ReadWrite(_first));
    }
    void IterationChecks::CopyIfValid(const R::Array<real> &from, R::Array<real> &to)
    {
        DNDS_check_throw_info(from.GetDevice() == _first.GetDevice() && to.GetDevice() == from.GetDevice() &&
                                  from.Layout() == to.Layout() && !from.Storage().Overlaps(to.Storage()),
                              "candidate publication requires distinct compatible arrays on the compute lane");
        auto engine = _stream.LockEngine();
        R::SubmitArrays(engine, _stream, [](auto where, auto valid, auto src, auto dst)
                        {
                            if (where.device == R::Device::Host())
                            {
                                if (valid(0, 0).status)
                                    OMP::ForRows(0, src.RowCount(), [&](index i)
                                                 { for (index j = 0; j < src.RowSize(i); ++j) dst(i,j) = src(i,j); });
                            }
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::CopyValidIterationRows(where, valid.DynamicView(), src.DynamicView(), dst.DynamicView());
#endif
                        },
                        R::Read(_first), R::Read(from), R::ReadWrite(to));
    }
    void IterationChecks::Observe()
    {
        if (!_pending)
            return;
        _stream.Record().WaitHost();
        auto host = _first.GetDevice() == R::Device::Host() ? _first : _first.To(R::Device::Host());
        const auto failure = host.Read().View()(0, 0);
        _pending = false;
        auto publications = std::exchange(_publications, {});
        if (!failure.status)
        {
            constexpr const char *names[]{"pseudo-time", "Jacobian", "diagonal inverse", "linear solve", "correction", "candidate", "flux", "source", "defect"};
            auto message = fmt::format("nonlinear update {} stage {} {}", failure.iteration, failure.stage, names[failure.phase]);
            RequireNumericalSuccess(failure.status, message.c_str());
        }
        for (auto &publish : publications)
            publish();
    }
}
