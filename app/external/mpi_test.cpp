#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int a, b;
    a = 1;
    MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "MPI through"<<std::endl;
    std::cout << b << std::endl;
    MPI_Finalize();
    return 0;
}