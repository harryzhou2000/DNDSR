#pragma once

#include <string>
#include <utility>

namespace DNDS::Cauchy
{
    /// Normal, collectively agreed numerical outcome of a candidate step.
    /// A rejected candidate leaves accepted state/history unchanged. The caller
    /// may discard it and retry at smaller dt. An interrupted step has already
    /// saved its restart cursor and must exit instead of retrying. Backend,
    /// access and allocation failures must terminate the MPI run.
    struct StepResult
    {
        bool accepted = true;
        std::string reason;
        bool interrupted = false;
        static StepResult Rejected(std::string reason)
        {
            return {false, std::move(reason)};
        }
        static StepResult Interrupted() { return {false, {}, true}; }
        explicit operator bool() const { return accepted; }
    };
}
