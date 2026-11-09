#pragma once

#include "DNDS/MPI/MPI.hpp"
#include <cstdlib>
#include <exception>
#include <string_view>

namespace DNDS::MPI
{
    /// Fatal solver boundary for a catch around collectively ordered work,
    /// including matched point-to-point MPI, collective I/O and callbacks.
    /// Checks may throw; a production catch must not recover. A coordinated
    /// unit test may recover when no peer can be stranded.
    /// No agreement, barrier, request draining or MPI finalization: peers may
    /// still be inside a different collective. Call on an MPI-permitted thread
    /// with a live communicator. Before initialization/after finalization only
    /// process termination is available. This function never returns.
    [[noreturn]] inline void Abort(MPI_Comm comm, std::string_view message, const char *cause = nullptr)
    {
        int rank = -1;
        int initialized = 0, finalized = 0;
        if (comm != MPI_COMM_NULL)
            MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        int mainThread = 0, support = MPI_THREAD_SINGLE;
        if (initialized && !finalized)
        {
            MPI_Is_thread_main(&mainThread);
            MPI_Query_thread(&support);
        }
        const bool live = initialized && !finalized && comm != MPI_COMM_NULL &&
                          (mainThread || support == MPI_THREAD_MULTIPLE);
        if (live)
            MPI_Comm_rank(comm, &rank);
        std::cerr << "DNDS fatal MPI error on rank " << rank << ": " << message;
        if (cause)
            std::cerr << ": " << cause;
        std::cerr << std::endl;
        if (live)
            MPI_Abort(comm, 1);
        std::abort();
    }

    namespace FailureDetail
    {
        inline thread_local MPI_Comm solverCommunicator = MPI_COMM_NULL;
    }

    /// Opt the production controller into fatal rejection independently of
    /// preflight agreement. This is controller-thread scoped; local APIs and
    /// unit tests do not inherit it merely because MPI is initialized.
    class SolverFailureScope
    {
        MPI_Comm _previous = FailureDetail::solverCommunicator;

    public:
        explicit SolverFailureScope(MPI_Comm control) noexcept { FailureDetail::solverCommunicator = control; }
        SolverFailureScope(const SolverFailureScope &) = delete;
        SolverFailureScope &operator=(const SolverFailureScope &) = delete;
        ~SolverFailureScope() { FailureDetail::solverCommunicator = _previous; }
    };

    inline bool SolverFailureScopeActive() noexcept
    {
        return FailureDetail::solverCommunicator != MPI_COMM_NULL;
    }

    /// Null communicator deliberately avoids all MPI calls, for instance on a
    /// compute worker when only the controller thread is allowed to call MPI.
    [[noreturn]] inline void AbortCaught(MPI_Comm comm, std::string_view message)
    {
        if (auto error = std::current_exception())
            try
            {
                std::rethrow_exception(error);
            }
            catch (const std::exception &exception)
            {
                Abort(comm, message, exception.what());
            }
            catch (...)
            {
                Abort(comm, message);
            }
        Abort(comm, message);
    }

    /// Inside a production catch, report the current exception and terminate
    /// on the solver communicator before any cleanup that could wait on MPI.
    /// With no solver scope the caller keeps its local validation contract.
    inline void AbortIfSolverFailure(std::string_view message)
    {
        const auto comm = FailureDetail::solverCommunicator;
        if (comm != MPI_COMM_NULL)
            AbortCaught(comm, message);
    }

    /// Assert a preflight result. A checked library caller can observe a
    /// rejection; the production controller cannot recover from it.
    inline void CheckPreflight(bool valid, const char *message)
    {
        if (!valid)
        {
            AbortIfSolverFailure(message);
            DNDS_check_throw_info(valid, message);
        }
    }
}
