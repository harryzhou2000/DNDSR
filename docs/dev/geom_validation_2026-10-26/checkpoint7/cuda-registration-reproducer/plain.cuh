#include <cuda_runtime.h>
extern __device__ int add_from_a(int);
template <class T>
__global__ void shared_kernel(T *out) { *out = T(add_from_a(4)); }
