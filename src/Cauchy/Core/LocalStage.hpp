#pragma once

#include "DNDS/MPI/CollectiveStage.hpp"
#include <utility>

namespace DNDS::Cauchy
{
    /// A local calculation between collective phases, never an MPI action.
    /// In unchecked library mode the caller must coordinate any rejection.
    /// Production always aborts in the catch before a later MPI operation.
    template <class F>
    void RunLocalStage(const MPIInfo &mpi, bool agree, F &&run, const char *message)
    {
        if (agree)
            MPI::LocalStage(mpi, std::forward<F>(run), message);
        else
            try
            {
                std::forward<F>(run)();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure(message);
                throw;
            }
    }
}
