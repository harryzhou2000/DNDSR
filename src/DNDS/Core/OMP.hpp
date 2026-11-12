#pragma once
/// @file OMP.hpp
/// @brief Helpers for OpenMP thread-count configuration at solver entry points.

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include "Eigen/src/Core/products/Parallelizer.h"
#include <algorithm>
#include <exception>
#include <mutex>

namespace DNDS::OMP
{
    /// Explicit team size for distributed kernels. Querying this on a CPU
    /// worker gives the same policy as on the controller; OpenMP's per-thread
    /// inherited max-team setting is deliberately not the policy source.
    inline int DistributedThreads() noexcept
    {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
        if (omp_in_parallel())
            return 1;
        return std::max(1, get_env_DNDS_DIST_OMP_NUM_THREADS());
#else
        return 1;
#endif
    }

    /** Independent row arithmetic; no lease, MPI, allocation or exception work.
     * @pre body writes only its row or thread-local state. Read-only metadata
     * is already published, and outputs cannot alias another row's input.
     * Ordered SGS/SOR scans must use subpartition policies instead.
     * One thread follows the original serial order with no OpenMP region.
     */
    template <class Body>
    void ForRows(index begin, index end, Body &&body) noexcept
    {
        const int threads = DistributedThreads();
        if (threads == 1 || end - begin <= 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                body(iRow);
            return;
        }
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (index iRow = begin; iRow < end; ++iRow)
            body(iRow);
    }

    /** Independent row arithmetic with explicit numerical failure status.
     * Successful rows have no shared-counter/atomic update. Each worker keeps
     * its first failure; only failing workers enter the final error merge.
     * The lowest logical row wins, independently of OpenMP scheduling. As with
     * every failed kernel, partial output must not be published by the caller.
     * body must not throw or perform MPI, leases, or mutable shared work.
     */
    template <class Body>
    NumericalStatus ForRowsStatus(index begin, index end, Body &&body) noexcept
    {
        const int threads = DistributedThreads();
        if (threads == 1 || end - begin <= 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                if (const auto status = body(iRow); !status)
                    return status;
            return {};
        }
        NumericalStatus result;
        index failedRow = end;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel num_threads(threads)
#endif
        {
            NumericalStatus local;
            index localRow = end;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp for schedule(static)
#endif
            for (index iRow = begin; iRow < end; ++iRow)
                if (local)
                    if (const auto status = body(iRow); !status)
                    {
                        local = status;
                        localRow = iRow;
                    }
            if (!local)
            {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical(DNDS_OMP_ROW_FAILURE)
#endif
                {
                    if (localRow < failedRow)
                    {
                        failedRow = localRow;
                        result = local;
                    }
                }
            }
        }
        return result;
    }

    /** Setup-only parallel work, allowing allocating/validating library calls.
     * Exceptions are retained until every worker has joined, then rethrown on
     * the controller (whose collective stage applies its usual failure policy).
     * No work item may call MPI. No exception crosses an OpenMP team boundary.
     * Do not use this helper for hot numerical kernels or numerical fallback.
     */
    template <class Body>
    void ForSetupRows(index begin, index end, Body &&body)
    {
        if (DistributedThreads() == 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                body(iRow);
            return;
        }
        std::mutex failureMutex;
        std::exception_ptr failure;
        index failedRow = end;
        ForRows(begin, end, [&](index iRow) noexcept
                {
            try
            {
                body(iRow);
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(failureMutex);
                if (iRow < failedRow)
                {
                    failedRow = iRow;
                    failure = std::current_exception();
                }
            } });
        if (failure)
            std::rethrow_exception(failure);
    }

    /**
     * @brief Configure OpenMP and Eigen threading for the "distributed OMP" pattern.
     *
     * @details Sets the OpenMP thread count from the @ref DNDS_DIST_OMP_NUM_THREADS
     * environment variable (falling back to 1) and disables Eigen's internal
     * parallelism so it does not oversubscribe the MPI+OpenMP layout.
     *
     * Intended to be called once near `main()` when @ref DNDS_USE_OMP is on.
     */
    inline void set_full_parallel_OMP()
    {
#ifdef DNDS_USE_OMP
        // omp_set_num_threads( // note that the meaning is like "omp_set_max_threads()"
        //     DNDS::MPIWorldSize() == -1
        //         ? std::min(omp_get_num_procs(), omp_get_max_threads())
        //         : (get_env_DNDS_DIST_OMP_NUM_THREADS() == 0 ? 1 : DNDS::get_env_DNDS_DIST_OMP_NUM_THREADS()));
        omp_set_num_threads(
            (get_env_DNDS_DIST_OMP_NUM_THREADS() == 0 ? 1 : DNDS::get_env_DNDS_DIST_OMP_NUM_THREADS()));
#endif
        Eigen::setNbThreads(1); // ! sets Eigen's multithreading!
    }
}
