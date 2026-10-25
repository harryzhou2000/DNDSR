"""Native periodic CGNS identification uses explicit transforms and retains images."""

from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom


MESH = Path(__file__).resolve().parents[2] / "data/mesh/IV10_10.cgns"
FRAMES = {"translation1": [10., 0., 0.], "translation2": [0., 10., 0.]}


@pytest.mark.mpi
def test_native_periodic_reader_exact_physical_cells_and_collective_retry():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    # Errors on only one rank agree before file/payload communication.
    bad = dict(FRAMES)
    if mpi.rank == 0:
        bad["unknown_periodic_option"] = [0., 0., 0.]
    with pytest.raises(RuntimeError):
        Geom.read_native_cgns(mpi, str(MESH), 2, periodic_geometry=bad)
    with pytest.raises(RuntimeError):
        Geom.read_native_cgns(mpi, str(MESH), 2, periodic_tolerance=1e-8)
    bad = dict(FRAMES)
    if mpi.rank == 0:
        bad["translation1"] = [11., 0., 0.]
    with pytest.raises(RuntimeError):
        Geom.read_native_cgns(mpi, str(MESH), 2, periodic_geometry=bad)
    source, names = Geom.read_native_cgns_with_names(
        mpi, str(MESH), 2, reader_rank=mpi.size - 1,
        periodic_geometry=FRAMES, periodic_tolerance=1e-8)
    assert len({tuple(sorted(part.items())) for part in MPI.COMM_WORLD.allgather(names.n2id_map)}) == 1
    assert source.has_native_state
    assert sum(MPI.COMM_WORLD.allgather(source.NumNode())) == 100
    assert sum(MPI.COMM_WORLD.allgather(source.NumCell())) == 100
    with source.acquire_read() as retained:
        before = retained.cell_coords(0).copy() if source.NumCell() else None
        completed = Geom.with_native_topology(source)
        if source.NumCell():
            np.testing.assert_array_equal(retained.cell_coords(0), before)
    assert sum(MPI.COMM_WORLD.allgather(completed.NumFace())) == 200
    with completed.acquire_read(faces=True, cell_face_neighbors=True) as read:
        assert read.is_periodic
        for cell in range(completed.NumCell()):
            points = read.cell_coords(cell)
            # The original10x10 physical quads remain unit squares, including
            # those spanning canonical seam nodes.
            np.testing.assert_allclose(np.ptp(points[:, :2], axis=0), [1., 1.], atol=1e-12)
            area = .5 * sum(points[i, 0] * points[(i + 1) % 4, 1] - points[(i + 1) % 4, 0] * points[i, 1] for i in range(4))
            assert area == pytest.approx(1.)
