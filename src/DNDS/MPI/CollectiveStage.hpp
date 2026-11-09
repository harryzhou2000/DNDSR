#pragma once

#include "DNDS/MPI/Failure.hpp"
#include <exception>

namespace DNDS::MPI
{
    /// Run local preparation, then agree before entering another collective.
    /// The action must be rank-local: no communication, collective I/O, domain
    /// solve or callback which may enter MPI. A catch followed by Allreduce
    /// cannot recover a failure from inside an MPI-containing operation: peers
    /// may still be in an earlier collective. Such errors must propagate to a
    /// fail-fast application boundary (MPI_Abort, without another agreement).
    /// Local communicator queries and local datatype construction are allowed.
    /// All ranks participate,
    /// including empty owners. The originating exception is preserved locally;
    /// peers throw the supplied diagnostic. This is not distributed rollback.
    template <class F>
    void LocalStage(const MPIInfo &mpi, F &&action,
                    const char *peerError = "local preparation failed on another rank before collective work")
    {
        int valid = 1;
        std::exception_ptr failure;
        try
        {
            action();
        }
        catch (...)
        {
            AbortIfSolverFailure("solver local preflight failed");
            valid = 0;
            failure = std::current_exception();
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (failure)
            std::rethrow_exception(failure);
        CheckPreflight(valid, peerError);
    }

    /// Compatibility convenience for an absent (-1) or bounded candidate ID.
    /// The bound and collective order must already agree across ranks.
    inline int AgreeCandidate(const MPIInfo &mpi, int candidate, size_t count)
    {
        DNDS_check_throw_info(count <= size_t(INT_MAX - 2), "candidate agreement range exceeds MPI range");
        const auto local = candidate >= 0 ? std::optional<int>(candidate) : std::nullopt;
        return AgreeOptionalInteger(mpi, local, int(count)).value_or(-1);
    }
}
