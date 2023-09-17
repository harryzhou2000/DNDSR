

import sys
import pybind11_basicTest


mpi_init_ret = pybind11_basicTest.MPI_Init(sys.argv)
mpi = pybind11_basicTest.MPIInfo()
mpi.setWorld()

if mpi.rank == 0 :
    print("Symbols in ext: " + str(dir(pybind11_basicTest)))

print(mpi)

pybind11_basicTest.MPI_Finalize()




