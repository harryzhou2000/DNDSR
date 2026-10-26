from DNDSR import DNDS, Geom
import os, sys
import mpi4py.MPI as MPI
import numpy as np


def get_mesh_2D(meshFile: str, mpi: DNDS.MPIInfo):

    from DNDSR.Geom.utils import read_mesh, prepare_mesh

    assert os.path.isfile(meshFile)
    result = read_mesh(meshFile, mpi, 2, partition_options={"metisUfactor": 5})
    prepare_mesh(result.mesh, result.reader, build_serial_out=False)
    return result.mesh, result.reader, result.name_to_id


def get_mpi4py_comm_from_MPIInfo(mpi: DNDS.MPIInfo):
    return MPI.Comm.fromhandle(mpi.comm())  # handling raw MPI_Comm handle (pointer)

t_travelling_cell_pack = tuple[Geom.Elem.ElemType, int, int, list[int], np.ndarray]

def pack_travelling_cell(
    cellType: Geom.Elem.ElemType,
    cellZone: int,
    iCell: int,
    cell2nodeRow: list[int],
    coords: np.ndarray,
):
    return (cellType, cellZone, iCell, cell2nodeRow, coords)
