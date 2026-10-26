#include "plain.cuh"
__device__ int add_from_a(int value) { return value + 3; }
extern "C" cudaError_t launch_a(int *out)
{
    shared_kernel<int><<<1, 1>>>(out);
    return cudaGetLastError();
}
