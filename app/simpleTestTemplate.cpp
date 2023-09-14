#include "Geom/Quadrature.hpp"

#include <cstdlib>

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    // MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    // MPI_Finalize();

    return 0;
}