#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void Check(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}

__global__ void Value(int *result) { *result = 37; }

int main(int argc, char **argv)
{
    const bool before = argc > 1 && std::strcmp(argv[1], "before") == 0;
    if (before)
    {
        Check(cudaSetDevice(0), "cudaSetDevice before MPI");
        Check(cudaFree(nullptr), "context before MPI");
    }
    MPI_Init(&argc, &argv);
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int *device = nullptr, value = 0, total = 0;
    Check(cudaMalloc(&device, sizeof(int)), "cudaMalloc");
    Value<<<1, 1>>>(device);
    Check(cudaGetLastError(), "Value launch");
    Check(cudaDeviceSynchronize(), "Value synchronize");
    Check(cudaMemcpy(&value, device, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    Check(cudaFree(device), "cudaFree");
    MPI_Allreduce(&value, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::printf("rank=%d size=%d init=%s device=%s value=%d total=%d expected=%d\n", rank, size,
                before ? "CUDA-before-MPI" : "MPI-before-CUDA", properties.name, value, total, 37 * size);
    std::fflush(stdout);
    const int result = value == 37 && total == 37 * size ? 0 : 3;
    MPI_Finalize();
    std::puts("MPI_Finalize returned");
    return result;
}
