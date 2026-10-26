#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
extern "C" int from_a();
int main()
{
    if (cudaSetDevice(0) != cudaSuccess)
        return 10;
    thrust::device_vector<uint8_t> bytes(16, uint8_t(5));
    thrust::device_vector<double> doubles(8, 3.0);
    const int a = from_a(), b = int(bytes[0]) + int(doubles[0]);
    const auto status = cudaDeviceSynchronize();
    std::printf("a=%d b=%d cuda=%s\n", a, b, cudaGetErrorString(status));
    return a == 9 && b == 8 && status == cudaSuccess ? 0 : 11;
}
