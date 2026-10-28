#include "DofAlgebra.hpp"
#include <array>
#include <climits>

namespace DNDS::Algebra::Detail
{
    std::vector<double> Aggregate(const DofSpace &space, const std::function<GlobalSchema()> &describe,
                                  const std::function<std::vector<double>()> &local)
    {
        const auto retained = space;
        const auto comm = retained.Communicator();
        int thread = MPI_THREAD_SINGLE, main = 0;
        MPI_Query_thread(&thread);
        MPI_Is_thread_main(&main);
        DNDS_check_throw_info(main || thread == MPI_THREAD_MULTIPLE, "global algebra reduction called from an unsupported MPI thread");
        GlobalSchema description{};
        std::exception_ptr failure;
        try
        {
            description = describe();
        }
        catch (...)
        {
            failure = std::current_exception();
        }
        int valid = !failure;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        if (failure)
            std::rethrow_exception(failure);
        DNDS_check_throw_info(valid, "algebra global operand validation failed on a rank");
        const auto width = description.width;
        const auto operation = description.operation;
        std::array<uint64_t, 5> schema{uint64_t(width), uint64_t(description.fieldWidth), uint64_t(description.scalarBytes), uint64_t(operation), uint64_t(description.protocol)};
        auto minimum = schema, maximum = schema;
        MPI_Allreduce(MPI_IN_PLACE, minimum.data(), int(minimum.size()), MPI_UINT64_T, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, maximum.data(), int(maximum.size()), MPI_UINT64_T, MPI_MAX, comm);
        DNDS_check_throw_info(minimum == maximum && width && width <= INT_MAX && operation >= 0 && operation <= 2,
                              "global algebra reduction schema differs between ranks or exceeds MPI range");
        std::vector<double> values;
        try
        {
            values = local();
            DNDS_check_throw_info(values.size() == width, "local algebra reduction result width differs");
        }
        catch (...)
        {
            failure = std::current_exception();
        }
        valid = !failure;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        if (failure)
            std::rethrow_exception(failure);
        DNDS_check_throw_info(valid, "local algebra reduction failed on a rank before MPI aggregation");
        MPI_Allreduce(MPI_IN_PLACE, values.data(), int(width), MPI_DOUBLE,
                      operation == 0 ? MPI_SUM : operation == 1 ? MPI_MIN
                                                                : MPI_MAX,
                      comm);
        return values;
    }

#ifndef DNDS_USE_CUDA
    void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<float>, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, double, double)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
    void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<double>, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, double, double)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
    void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, Runtime::ArrayView<double>)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
    void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, Runtime::ArrayView<double>)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
#endif
}
