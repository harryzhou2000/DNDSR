#include "DNDS/Algebra/DofAlgebra.hpp"
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
        const bool checkPreflight = retained.ChecksCollectivePreflight();
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
        if (checkPreflight)
        {
            // Combine operand-validation status and schema min/max agreement into one collective.
            const int64_t w = int64_t(description.width);
            const int64_t fw = int64_t(description.fieldWidth);
            const int64_t sb = int64_t(description.scalarBytes);
            const int64_t op = int64_t(description.operation);
            const int64_t proto = int64_t(description.protocol);
            std::array<int64_t, 11> check{!failure ? 1 : 0, w, fw, sb, op, proto, -w, -fw, -sb, -op, -proto};
            MPI_Allreduce(MPI_IN_PLACE, check.data(), int(check.size()), MPI_INT64_T, MPI_MIN, comm);
            if (failure)
                std::rethrow_exception(failure);
            DNDS_check_throw_info(check[0] == 1, "algebra global operand validation failed on a rank");
            const bool schemaAgreed = (check[1] == -check[6]) && (check[2] == -check[7]) &&
                                      (check[3] == -check[8]) && (check[4] == -check[9]) && (check[5] == -check[10]);
            DNDS_check_throw_info(schemaAgreed && description.width && description.width <= INT_MAX &&
                                      description.operation >= 0 && description.operation <= 2,
                                  "global algebra reduction schema differs between ranks or exceeds MPI range");
        }
        else
        {
            if (failure)
                std::rethrow_exception(failure);
            DNDS_check_throw_info(description.width && description.width <= INT_MAX &&
                                      description.operation >= 0 && description.operation <= 2,
                                  "global algebra reduction schema exceeds MPI range");
        }
        const auto width = description.width;
        const auto operation = description.operation;
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
        if (checkPreflight)
        {
            int valid = !failure;
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
            if (failure)
                std::rethrow_exception(failure);
            DNDS_check_throw_info(valid, "local algebra reduction failed on a rank before MPI aggregation");
        }
        else if (failure)
            std::rethrow_exception(failure);
        MPI_Allreduce(MPI_IN_PLACE, values.data(), int(width), MPI_DOUBLE,
                      operation == 0 ? MPI_SUM : operation == 1 ? MPI_MIN
                                                                : MPI_MAX,
                      comm);
        return values;
    }

#ifndef DNDS_USE_CUDA
    size_t ReduceCUDABlockCount(Reduction, size_t, size_t) { return 1; }
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
    void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, Runtime::ArrayView<double>, Runtime::ArrayView<double>, Runtime::ArrayView<unsigned int>)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
    void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, Runtime::ArrayView<double>, Runtime::ArrayView<double>, Runtime::ArrayView<unsigned int>)
    {
        DNDS_check_throw_info(false, "CUDA algebra is unavailable in this build");
    }
#endif
}
