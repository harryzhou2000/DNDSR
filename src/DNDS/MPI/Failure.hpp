#pragma once

#include "DNDS/MPI/MPI.hpp"

namespace DNDS::MPI
{
    /// Fatal application boundary for an unexpected distributed exception.
    /// No agreement, barrier, request draining or MPI finalization: peers may
    /// still be inside a different collective. Call on an MPI-permitted thread
    /// with a live communicator. Recover only explicitly agreed rejections.
    [[noreturn]] inline void Abort(MPI_Comm comm, const std::string &message)
    {
        int rank = -1;
        MPI_Comm_rank(comm, &rank);
        std::cerr << "DNDS fatal MPI error on rank " << rank << ": " << message << std::endl;
        MPI_Abort(comm, 1);
        std::abort();
    }
}
