// Dependency-only control: no DNDS headers, allocations, arrays or operators.
#include <mpi.h>
#include <cuda_runtime_api.h>
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto result = cudaFree(nullptr); // Initialize the CUDA runtime context.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return result == cudaSuccess ? 0 : 2;
}
