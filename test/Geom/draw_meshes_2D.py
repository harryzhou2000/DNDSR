from DNDSR import DNDS
from DNDSR.Geom.utils import read_mesh, prepare_mesh
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_meshes_2D(mesh_file=None, output_file=None):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    if mesh_file is None:
        mesh_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
        )
    result = read_mesh(mesh_file, mpi, 2, partition_options={"metisUfactor": 5})
    print(result.name_to_id.n2id_map)
    mesh = result.mesh
    prepare_mesh(mesh, result.reader, build_serial_out=False)
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
    if combined:
        ax.set_xlim(xymins[0], xymaxs[0])
        ax.set_ylim(xymins[1], xymaxs[1])
    ax.set_title(f"part_{mpi.rank}")

    # Show the plot
    # plt.show(block=False)
    fig.savefig(output_file or f"test_print_part_{mpi.rank}.png")
    return fig


if __name__ == "__main__":
    draw_meshes_2D()
