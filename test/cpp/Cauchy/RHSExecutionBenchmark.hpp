#pragma once

#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/MPI.hpp"
#include <nlohmann/json.hpp>

namespace Plan53
{
    /// Benchmark-only owner. Link wrappers substitute public halo bindings,
    /// never numerical kernels. Caller/compute-worker keep MPI on the caller;
    /// mpi-worker sends hot control calls and all payload progression to one
    /// worker. Every variant reserves the same two physical cores per rank.
    class RHSExecution
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        RHSExecution(const DNDS::MPIInfo &, std::string mode, std::string transport);
        ~RHSExecution();
        RHSExecution(const RHSExecution &) = delete;
        RHSExecution &operator=(const RHSExecution &) = delete;
        int HostThreads() const;
        void PinCompute(DNDS::Runtime::EagerEngine &, const DNDS::Runtime::ExecutionStream &);
        /// Disable only the benchmark payload substitution while constructing
        /// the independent original Direct/inline reference workspace.
        void OriginalBindings(bool);
        void BeginSample(bool instrumented);
        nlohmann::ordered_json EndSample();
        nlohmann::ordered_json Placement() const;
    };
}
