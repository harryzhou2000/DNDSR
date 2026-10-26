#include "plain.cuh"
#include <cstdio>
extern "C" cudaError_t launch_a(int *);
int main()
{
    int *out = nullptr, a = 0, b = 0;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&out, sizeof(int)) != cudaSuccess)
        return 10;
    if (launch_a(out) != cudaSuccess || cudaMemcpy(&a, out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 11;
    shared_kernel<int><<<1, 1>>>(out);
    if (cudaGetLastError() != cudaSuccess || cudaMemcpy(&b, out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 12;
    if (cudaFree(out) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return 13;
    std::printf("cross-TU device function: a=%d b=%d\n", a, b);
    return a == 7 && b == 7 ? 0 : 14;
}
