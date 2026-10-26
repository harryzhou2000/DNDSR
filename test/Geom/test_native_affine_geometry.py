"""Collective Python input failures must leave native geometry unpublished."""

from pathlib import Path

import numpy as np
import pytest

from DNDSR import DNDS
from DNDSR.Geom.utils import read_mesh, prepare_native_mesh


def test_affine_python_shape_errors_and_retained_read_rollback():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    source = Path(__file__).resolve().parents[2] / "data/mesh/UniformSquare_10.cgns"
    mesh = prepare_native_mesh(read_mesh(str(source), mpi, 2, native=True).mesh)
    old = mesh.snapshot(faces=True)
    with old.acquire_read() as read:
        before = read.coordinates()
    linear = np.array([[2., .4, 0.], [0., .5, 0.], [0., 0., 1.]])
    offset = np.array([.3, -.2, .7])
    for matrix, shift in (
        (np.eye(2) if mpi.rank == 0 else linear, offset),
        (linear, np.zeros(2) if mpi.rank == 0 else offset),
        (np.full((3, 3), np.nan) if mpi.rank == 0 else linear, offset),
        (-np.eye(3) if mpi.rank == 0 else linear, offset),
    ):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            mesh.apply_affine_geometry_transform(matrix, shift)
        assert old.source_current
        with mesh.acquire_read() as read:
            np.testing.assert_array_equal(read.coordinates(), before)
    held = mesh.acquire_read() if mpi.rank == 0 else None
    try:
        with pytest.raises(RuntimeError):
            mesh.apply_affine_geometry_transform(linear, offset)
        assert old.source_current
    finally:
        if held is not None:
            held.close()
    mesh.apply_affine_geometry_transform(linear, offset)
    assert not old.source_current
    with mesh.acquire_read() as read:
        np.testing.assert_allclose(read.coordinates(), before @ linear.T + offset,
                                   rtol=1e-12, atol=1e-12)
    with old.acquire_read() as read:
        np.testing.assert_array_equal(read.coordinates(), before)
