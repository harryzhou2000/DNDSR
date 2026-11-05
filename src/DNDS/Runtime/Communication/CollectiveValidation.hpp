#pragma once

#include "DNDS/MPI/MPI.hpp"
#include <cstdio>

namespace DNDS::Runtime::CommunicationDetail
{
    /// Required all-rank agreement. Protocol/schema choices use this regardless
    /// of the optional safety policy; do not substitute it for routing data.
    inline void RequireAll(bool valid, MPI_Comm comm, const char *message)
    {
        int all = valid;
        MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, comm);
        DNDS_check_throw_info(all, message);
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
            int rank = 0;
            MPI_Comm_rank(comm, &rank);
            std::fprintf(stderr, "Runtime rank %d: %s (collective preflight disabled; aborting)\n", rank, message);
            std::fflush(stderr);
            MPI_Abort(comm, 1);
            DNDS_check_throw_info(false, message); // An MPI error handler must not permit publication.
        }
    }
}
