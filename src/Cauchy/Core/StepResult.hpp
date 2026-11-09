#pragma once

#include <string>
#include <utility>

namespace DNDS::Cauchy
{
    /// Normal, collectively agreed numerical outcome of a candidate step.
    /// A rejected candidate leaves accepted state/history unchanged. The caller
    /// may discard it and retry at smaller dt. Backend, access and allocation
    /// failures are not numerical rejection and must terminate the MPI run.
    struct StepResult
    {
        bool accepted = true;
        std::string reason;
        static StepResult Rejected(std::string reason)
        {
            return {false, std::move(reason)};
        }
        explicit operator bool() const { return accepted; }
    };
}
