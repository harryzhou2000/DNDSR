from DNDSR import DNDS
from DNDSR.Geom.utils import read_mesh, prepare_mesh
import os

def get_mesh_2D(meshFile: str):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    result = read_mesh(meshFile, mpi, 2, partition_options={"metisUfactor": 5})
    print(result.name_to_id.n2id_map)
    prepare_mesh(result.mesh, result.reader, build_serial_out=False)
    result.mesh.AssertOnFaces()

    return result.mesh, result.reader


if __name__ == "__main__":
    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    )
    m, msr = get_mesh_2D(meshFile)
