#include "Euler/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    int errc = DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS>(argc, argv);
    if (errc)
        MPI_Abort(MPI_COMM_WORLD, errc);
    MPI_Finalize();
}