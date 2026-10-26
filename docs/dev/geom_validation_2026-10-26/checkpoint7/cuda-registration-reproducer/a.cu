#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cstdint>
extern "C" int from_a()
{
    thrust::device_vector<uint8_t> bytes(16, uint8_t(7));
    thrust::device_vector<double> doubles(8, 2.0);
    return int(bytes[0]) + int(doubles[0]);
}
