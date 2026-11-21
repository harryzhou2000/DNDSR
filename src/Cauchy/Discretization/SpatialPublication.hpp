#pragma once

#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/Failure.hpp"

namespace DNDS::Cauchy
{
    /// Controller-owned tail of a submitted FV evaluation. Its owning solve
    /// retains the operands and workspace. Consume before reusing either. The
    /// event covers the compute lane and imported transfer publications only.
    class SpatialPublication
    {
        Runtime::StreamEvent _ready;
        std::function<void()> _publish;

    public:
        SpatialPublication() = default;
        SpatialPublication(const SpatialPublication &) = delete;
        SpatialPublication &operator=(const SpatialPublication &) = delete;
        template <class Publish>
        void Set(Runtime::StreamEvent ready, Publish publish)
        {
            DNDS_check_throw_info(!_publish, "spatial publication is already pending");
            auto retained = std::make_shared<Publish>(std::move(publish));
            _publish = [retained]
            { (*retained)(); };
            _ready = std::move(ready);
        }
        void Observe()
        {
            auto publish = std::exchange(_publish, {});
            if (publish)
            {
                _ready.WaitHost();
                publish();
            }
        }
        ~SpatialPublication()
        {
            // A discarded solve still owes retirement before releasing its
            // borrowed workspace. Only the explicit observer publishes values.
            if (_publish)
                try
                {
                    _ready.WaitHost();
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("discarded FV prefix failed during retirement");
                }
        }
    };
}
