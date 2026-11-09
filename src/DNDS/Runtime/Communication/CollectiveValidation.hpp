#pragma once

#include "DNDS/MPI/MPI.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <exception>
#include <utility>

namespace DNDS::Runtime::CommunicationDetail
{
    /// Mandatory agreement at a coordinated preparation boundary. The actual
    /// selected policy must agree, even when several valid contexts exist.
    inline void AgreePolicy(MPI_Comm comm, bool enabled)
    {
        MPI::CheckPreflight(comm != MPI_COMM_NULL, "preflight policy agreement needs a communicator");
        int policy[]{int(enabled), -int(enabled)};
        MPI_Allreduce(MPI_IN_PLACE, policy, 2, MPI_INT, MPI_MIN, comm);
        MPI::CheckPreflight(policy[0] == -policy[1], "collective preflight policy differs between ranks");
    }
    /// Required all-rank agreement. Protocol/schema choices use this regardless
    /// of the optional safety policy; do not substitute it for routing data.
    inline void RequireAll(bool valid, MPI_Comm comm, const char *message)
    {
        if (!valid)
            MPI::AbortIfSolverFailure(message);
        int all = valid;
        MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, comm);
        MPI::CheckPreflight(all, message);
    }

    /// Optional safety agreement, not a numerical or routing collective.
    /// Without agreement, a local failure cannot be recovered rank-locally:
    /// terminate the communicator before peers can hang or publish partial data.
    /// Successful unchecked starts perform no replacement handshake.
    inline void Preflight(bool valid, MPI_Comm comm, bool enabled, const char *message)
    {
        if (enabled)
            RequireAll(valid, comm, message);
        else if (!valid)
        {
            MPI::AbortIfSolverFailure(message);
            MPI::Abort(comm, message);
        }
    }

    /// Local preparation only. Agreement preserves the originating exception;
    /// disabled agreement aborts the communicator on failure before peers can
    /// enter unmatched payload work. Required protocol collectives stay outside.
    template <class F>
    void LocalStage(MPI_Comm comm, bool enabled, F &&run, const char *message)
    {
        std::exception_ptr failure;
        try
        {
            std::forward<F>(run)();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure(message);
            failure = std::current_exception();
        }
        try
        {
            Preflight(!failure, comm, enabled, message);
        }
        catch (...)
        {
            if (failure)
                std::rethrow_exception(failure);
            throw;
        }
    }
}
