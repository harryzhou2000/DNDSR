#include "DNDS/Runtime/Array.hpp"
#include <cuda_runtime.h>

namespace R = DNDS::Runtime;
using State = R::Array<double, R::FixedRows<5>>;

__global__ void ScaleSharedState(R::ArrayView<double> values, double factor)
{
    for (size_t i = blockIdx.x; i < values.Rows(); i += gridDim.x)
        for (size_t j = threadIdx.x; j < values.RowSize(i); j += blockDim.x)
            values(i, j) *= factor;
}

void BridgeScaleCUDA(State &array, double factor)
{
    auto context = R::ExecutionContext{array.GetDevice(), nullptr};
    auto access = array.Write(context);
    ScaleSharedState<<<3, 32>>>(access.View(), factor);
    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "bridge kernel launch failed");
    access.Complete(R::Completion::Record(context));
    access.Wait();
}
