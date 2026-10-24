"""Managed host mesh queries; requires rebuilt and installed Geom bindings."""

import gc
from pathlib import Path

import numpy as np
import pytest

from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import read_mesh, prepare_mesh


def _build_mesh(periodic=False):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    source = Path(__file__).resolve().parents[2] / "data/mesh" / (
        "IV10_10.cgns" if periodic else "UniformSquare_10.cgns"
    )
    assert source.is_file()
    options = {"periodic_geometry": {"translation1": [10., 0., 0.], "translation2": [0., 10., 0.]}} if periodic else {}
    result = read_mesh(str(source), mpi=mpi, dim=2, **options)
    prepare_mesh(result.mesh, result.reader, build_serial_out=False)
    return result.mesh


@pytest.fixture
def facade_mesh():
    return _build_mesh()


def test_readonly_queries_and_checked_indices(facade_mesh):
    mesh = facade_mesh
    with mesh.acquire_read(faces=True) as read:
        assert not read.closed and read.valid
        assert read.dimension == 2 and not read.is_periodic
        assert read.counts["cell"] == (
            mesh.NumCell(), mesh.NumCellGhost(), mesh.NumCellProc()
        )
        nodes = read.cell_nodes(0)
        faces = read.cell_faces(0)
        assert nodes.dtype == np.dtype(np.int64)
        assert len(nodes) in (3, 4)
        np.testing.assert_array_equal(nodes, read.relation(Geom.MeshRelation.CellNodes, 0))
        points = read.cell_coords(0)
        np.testing.assert_allclose(points, np.stack([read.coord(int(n)) for n in nodes]))
        assert points.shape == (len(nodes), 3)
        for slot, face in enumerate(faces):
            adjacent = read.face_cells(int(face))
            assert 0 in adjacent
            assert read.cell_face_other(0, int(face), slot) in adjacent
            assert read.cell_is_face_back(0, int(face), slot) == (adjacent[0] == 0)
        for array in (nodes, faces, points, read.coord(0), read.coordinates()):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
            if array.size:
                with pytest.raises(ValueError):
                    array.flat[0] = 99
        info = read.cell_info(0)
        expected_zone = info.zone
        info.zone = 123456
        assert read.cell_info(0).zone == expected_zone
        with pytest.raises(IndexError):
            read.cell_nodes(-1)
        with pytest.raises(IndexError):
            read.coord(read.counts["node"][2])
        with pytest.raises(IndexError):
            read.cell_face_other(0, int(faces[0]), len(faces))
        with pytest.raises(ValueError):
            read.cell_node_periodic_bits(0)
        with pytest.raises(ValueError):
            read.wall_distance(0)
    assert read.closed and not read.valid
    with pytest.raises(ValueError):
        read.cell_nodes(0)
    read.close()  # Idempotent, including after __exit__.


def test_escaped_numpy_slices_retain_scope(facade_mesh):
    mesh = facade_mesh
    read = mesh.acquire_read(faces=True)
    row = read.cell_nodes(0)
    expected = row.copy()
    escaped = row[::-1]
    converted = np.asarray(escaped)
    read.close()
    del row, escaped
    gc.collect()
    np.testing.assert_array_equal(converted, expected[::-1])
    with pytest.raises(ValueError):
        converted.setflags(write=True)
    with pytest.raises(RuntimeError):
        mesh.ReorderLocalCells(2, 1)
    del converted
    gc.collect()
    mesh.ReorderLocalCells(2, 1)
    with mesh.acquire_read(faces=True) as current:
        assert current.counts["cell"][0] == mesh.NumCell()


def test_exception_closes_context_and_snapshots_do_not_pin_mesh(facade_mesh):
    mesh = facade_mesh
    with pytest.raises(LookupError):
        with mesh.acquire_read() as read:
            snapshot = read.coordinates()
            with pytest.raises((ValueError, IndexError)):
                read.face_nodes(0)
            raise LookupError("deliberate caller failure")
    assert read.closed
    mesh.ReorderLocalCells(2, 1)
    assert snapshot.shape[1] == 3 and not snapshot.flags.writeable
    with pytest.raises(ValueError):
        read.__enter__()


def test_parent_deletion_keeps_row_and_metadata():
    mesh = _build_mesh()
    read = mesh.acquire_read()
    row = read.cell_nodes(0)
    coordinate = read.coord(int(row[0]))
    expected = coordinate.copy()
    read.close()
    del mesh, read, row
    gc.collect()
    np.testing.assert_array_equal(coordinate[:], expected)
    assert not coordinate.flags.writeable


def test_detached_snapshot_and_escaped_rows(facade_mesh):
    mesh = facade_mesh
    snapshot = mesh.snapshot(faces=True)
    assert snapshot.device == "cpu" and snapshot.source_current
    with snapshot.acquire_read() as read:
        row = read.cell_nodes(0)
        original_nodes = row.copy()
        original_points = read.coordinates()
    mesh.transform_coordinates(lambda point: point + [0.5, 0.25, 0.0])
    assert not snapshot.source_current
    with snapshot.acquire_read() as read:
        np.testing.assert_array_equal(read.coordinates(), original_points)
    del snapshot
    gc.collect()
    np.testing.assert_array_equal(row, original_nodes)
    assert not row.flags.writeable


def test_detached_cuda_snapshot_debug_copy(facade_mesh):
    from DNDSR.DNDS import Runtime

    if not Runtime.cuda_enabled:
        pytest.skip("CPU-only build; run the CUDA binding profile for this case")
    snapshot = facade_mesh.snapshot(faces=True)
    with snapshot.acquire_read() as read:
        expected = read.coordinates()
    device = snapshot.to_device(0)
    assert device.device == "cuda:0"
    with pytest.raises(RuntimeError):
        device.acquire_read()
    host = device.to_host()
    del snapshot, device
    gc.collect()
    with host.acquire_read() as read:
        np.testing.assert_array_equal(read.coordinates(), expected)


def test_periodic_incidence_queries():
    from mpi4py import MPI

    mesh = _build_mesh(periodic=True)
    transformed = 0
    with mesh.acquire_read(faces=True) as read:
        assert read.is_periodic
        for cell in range(read.counts["cell"][0]):
            nodes = read.cell_nodes(cell)
            bits = read.cell_node_periodic_bits(cell)
            assert bits.dtype == np.dtype(np.uint8)
            assert len(bits) == len(nodes)
            assert not bits.flags.writeable
            with pytest.raises(ValueError):
                bits.setflags(write=True)
            raw = np.stack([read.coord(int(node)) for node in nodes])
            physical = read.cell_coords(cell)
            transformed += int(np.any(np.abs(physical - raw) > 1e-10))
    assert MPI.COMM_WORLD.allreduce(transformed) > 0


def test_managed_coordinate_transform_rejects_escaped_rows(facade_mesh):
    mesh = facade_mesh
    with mesh.acquire_read() as read:
        before = read.coordinates()
        escaped = read.coord(0)[::2]
    called = []

    def transform(point):
        called.append(point)
        return point + np.array([0.125, -0.25, 0.5])

    with pytest.raises(RuntimeError):
        mesh.transform_coordinates(transform)
    assert not called
    del escaped
    gc.collect()
    mesh.transform_coordinates(transform)
    with mesh.acquire_read() as current:
        after = current.coordinates()
    np.testing.assert_allclose(after, before + [0.125, -0.25, 0.5])
    assert called
    called[0][:] = 999  # Callback inputs are detached; this cannot edit the mesh.
    with mesh.acquire_read() as current:
        np.testing.assert_array_equal(current.coordinates(), after)

    def in_place(point):
        point[2] += 0.25

    mesh.transform_coordinates(in_place)
    with mesh.acquire_read() as current:
        np.testing.assert_allclose(current.coordinates(), after + [0, 0, 0.25])


def test_coordinate_callback_failure_marks_mesh_failed():
    mesh = _build_mesh()

    def fail(_point):
        raise LookupError("deliberate coordinate callback failure")

    with pytest.raises(LookupError):
        mesh.transform_coordinates(fail)
    with pytest.raises(RuntimeError):
        mesh.acquire_read()


@pytest.mark.parametrize("bad_point", [[], [float("nan"), 0, 0]])
def test_coordinate_callback_shape_and_finite_checks(bad_point):
    mesh = _build_mesh()
    with pytest.raises(ValueError):
        mesh.transform_coordinates(lambda _point: bad_point)
    with pytest.raises(RuntimeError):
        mesh.acquire_read()
