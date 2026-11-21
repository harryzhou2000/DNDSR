#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/SYCLQueue.hpp"

namespace DNDS::Algebra::Detail
{
    template <class T>
    void ApplySYCL(Runtime::ExecutionContext context, Pointwise op, Runtime::ArrayView<T> out,
                   Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y, double a, double b)
    {
        const auto rows = size_t(out.RowCount());
        if (!rows)
            return;
        Runtime::SYCLQueue(context).parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id)
                                                 {
            const auto i = index(id[0]);
            for (index j = 0; j < out.RowSize(i); ++j)
                switch (op)
                {
                case Pointwise::Fill: out(i,j) = T(a); break;
                case Pointwise::Copy: out(i,j) = x(i,j); break;
                case Pointwise::Scale: out(i,j) *= T(a); break;
                case Pointwise::Axpy: out(i,j) += T(a) * x(i,j); break;
                case Pointwise::Combination: out(i,j) = T(a) * x(i,j) + T(b) * y(i,j); break;
                } });
    }

    template <class T>
    void ReduceSYCL(Runtime::ExecutionContext context, Reduction op, Runtime::ArrayView<const T> x,
                    Runtime::ArrayView<const T> y, Runtime::ArrayView<double> out)
    {
        // One independent component per work item. Preserve serial arithmetic
        // order, including empty partitions and the scalar all-component dot.
        Runtime::SYCLQueue(context).parallel_for(sycl::range<1>(size_t(out.width)), [=](sycl::id<1> id)
                                                 {
            const auto component = index(id[0]);
            double value = op == Reduction::Min ? std::numeric_limits<double>::infinity()
                         : op == Reduction::Max ? -std::numeric_limits<double>::infinity() : 0.;
            for (index i = 0; i < x.RowCount(); ++i)
                if (op == Reduction::Dot)
                    for (index j = 0; j < x.width; ++j)
                        value += double(x(i,j)) * double(y(i,j));
                else
                {
                    const double v = x(i,component);
                    switch (op)
                    {
                    case Reduction::Sum: value += v; break;
                    case Reduction::Squares: value += v * v; break;
                    case Reduction::Min: value = sycl::fmin(value,v); break;
                    case Reduction::Max: value = sycl::fmax(value,v); break;
                    default: break;
                    }
                }
            out(0,component) = value; });
    }

#define DNDS_INSTANTIATE_SYCL_ALGEBRA(T)                                                                                                                               \
    template void ApplySYCL<T>(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<T>, Runtime::ArrayView<const T>, Runtime::ArrayView<const T>, double, double); \
    template void ReduceSYCL<T>(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const T>, Runtime::ArrayView<const T>, Runtime::ArrayView<double>)
    DNDS_INSTANTIATE_SYCL_ALGEBRA(float);
    DNDS_INSTANTIATE_SYCL_ALGEBRA(double);
#undef DNDS_INSTANTIATE_SYCL_ALGEBRA
}
