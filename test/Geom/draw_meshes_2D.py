from DNDSR import DNDS, Geom
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_meshes_2D():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    mesh = Geom.UnstructuredMesh(mpi, 2)
    meshReader = Geom.UnstructuredMeshSerialRW(mesh, 0)
    name2ID = meshReader.ReadFromCGNSSerial(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
        )
    )
    print(name2ID.n2id_map)
    meshReader.Deduplicate1to1Periodic(1e-9)
    meshReader.BuildCell2Cell()
    meshReader.MeshPartitionCell2Cell({"metisUfactor": 5})
    meshReader.PartitionReorderToMeshCell2Cell()

    mesh.RecoverNode2CellAndNode2Bnd()
    mesh.RecoverCell2CellAndBnd2Cell()
    mesh.BuildGhostPrimary()
    mesh.AdjGlobal2LocalPrimary()
    mesh.InterpolateFace()
    mesh.AssertOnFaces()

    fig, ax = plt.subplots(figsize=(16, 16), dpi=320)
    xymaxs = np.array([-1e100, -1e100], dtype=np.double)
    xymins = np.array([1e100, 1e100], dtype=np.double)
    with mesh.acquire_read() as read:
        owned, _, combined = read.counts["cell"]
        for iCell in range(combined):
            # Physical element coordinates preserve periodic incidence frames.
            vertices = read.cell_coords(iCell)[:, :2]
            xymaxs = np.maximum(vertices.max(axis=0), xymaxs)
            xymins = np.minimum(vertices.min(axis=0), xymins)
            polygon = patches.Polygon(
                vertices,
                closed=True,
                edgecolor="black",
                facecolor="blue" if iCell < owned else "red",
                lw=1,
            )
            ax.add_patch(polygon)

    # Maintain aspect ratio
    ax.set_aspect("equal")
    ax.set_xlim(xymins[0], xymaxs[0])
    ax.set_ylim(xymins[1], xymaxs[1])
    ax.set_title(f"part_{mpi.rank}")

    # Show the plot
    # plt.show(block=False)
    plt.figure(fig)
    plt.savefig(f"test_print_part_{mpi.rank}.png")


if __name__ == "__main__":
    draw_meshes_2D()
