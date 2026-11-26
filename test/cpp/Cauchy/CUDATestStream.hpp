#pragma once

#include "DNDS/MPI/MPI.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

#include <memory>

namespace CauchyTest
{
    inline DNDS::MPIInfo WorldMPI()
    {
        DNDS::MPIInfo world;
        world.setWorld();
        return world;
    }

    /// RAII non-default CUDA stream backed by a real CUDALauncher.
    /// Avoids legacy default null-stream serialization that masks stream-ordering races.
    struct NonDefaultCUDAStream
    {
        DNDS::Runtime::EagerEngine engine;
        DNDS::Runtime::ExecutionStream stream;

        explicit NonDefaultCUDAStream(int device = 0)
            : engine(0), stream(engine.CreateStream(std::make_shared<DNDS::Runtime::CUDALauncher>(device)))
        {
        }

        DNDS::Runtime::ExecutionContext Context() const
        {
            return stream.Context();
        }

        operator DNDS::Runtime::ExecutionContext() const
        {
            return stream.Context();
        }

        void Synchronize() const
        {
            DNDS::Runtime::Completion::Record(stream.Context()).Wait();
        }
    };
}
