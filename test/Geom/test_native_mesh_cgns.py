"""Native import, topology completion and restart through the Python facade."""

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom


def _fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("native_cgns_fixture", root / "scripts/generate_multiblock_cgns.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path, error = None, None
    if MPI.COMM_WORLD.rank == 0:
        try:
            # One independent quad; no external mesh files or parser-generated
            # expected topology. More ranks than cells exercises empty owners.
            path = str(module.generate(1, tmp_path, root))
        except Exception as exception:
            error = str(exception)
    path, error = MPI.COMM_WORLD.bcast((path, error), root=0)
    assert error is None, error
    return path


def _check(mesh):
    nodes = mesh.entities(Geom.EntityKind.Node)
    cells = mesh.entities(Geom.EntityKind.Cell)
    faces = mesh.entities(Geom.EntityKind.Face)
    assert mesh.has_native_state
    assert mesh.NumCellGlobal() == 1
    assert mesh.NumNodeGlobal() == 4
    assert mesh.NumFaceGlobal() == 4
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    expected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    payload = {}
    with mesh.acquire_read(inverse=True, faces=True) as read:
        owned_faces = {
            faces.local_to_global(row): tuple(nodes.local_to_global(int(node)) for node in read.face_nodes(row))
            for row in range(faces.owned)
        }
        face_catalog = {}
        for part in MPI.COMM_WORLD.allgather(owned_faces):
            face_catalog.update(part)
        assert len(face_catalog) == 4
        for row in range(nodes.local):
            global_row = nodes.local_to_global(row)
            np.testing.assert_array_equal(read.coord(row), coordinates[global_row])
            assert mesh.original_entity_id(Geom.EntityKind.Node, row) == global_row
        for row in range(cells.local):
            assert [nodes.local_to_global(int(node)) for node in read.cell_nodes(row)] == [0, 1, 2, 3]
            assert mesh.original_entity_id(Geom.EntityKind.Cell, row) == 0
            cone = read.cell_faces(row)
            assert len(cone) == 4
            # Outer ghost rows may retain unavailable global references with
            # negative encoding. Resolve those against the owned-row oracle.
            actual = [face_catalog[-1 - int(face) if face < 0 else faces.local_to_global(int(face))] for face in cone]
            # The cell's element face slots preserve the oriented perimeter.
            assert actual == expected_edges
        for row in range(faces.local):
            cone = tuple(nodes.local_to_global(int(node)) for node in read.face_nodes(row))
            adjacent = read.face_cells(row)
            assert len(adjacent) == 2
            assert cells.local_to_global(int(adjacent[0])) == 0
            assert adjacent[1] == np.iinfo(np.int64).min
            assert cone in expected_edges
            assert mesh.original_entity_id(Geom.EntityKind.Face, row) == faces.local_to_global(row)
            payload[mesh.original_entity_id(Geom.EntityKind.Face, row)] = cone
    return payload


@pytest.mark.mpi
def test_native_cgns_topology_archive_and_retained_source(tmp_path):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = _fixture(tmp_path)
    primary = Geom.read_native_cgns(mpi, path, 2, reader_rank=mpi.size - 1, complete_node_star=True)
    assert primary.NumCell() == int(mpi.rank == 0)
    read = primary.acquire_read(inverse=True)
    escaped = read.coord(0)[::-1].copy() if primary.NumNodeProc() else None
    # Keep an actual escaped read row across a new topology publication.
    retained = read.coord(0)[::-1] if primary.NumNodeProc() else None
    read.close()
    mesh = Geom.with_native_topology(primary)
    expected = _check(mesh)
    del primary, read
    gc.collect()
    if retained is not None:
        np.testing.assert_array_equal(retained, escaped)
        with pytest.raises(ValueError):
            retained.setflags(write=True)
    filename = tmp_path / f"native-cgns-r{mpi.rank}.json"
    serializer = DNDS.Serializer.SerializerJSON()
    serializer.OpenFile(str(filename), False)
    mesh.write_native_archive(serializer, "mesh")
    serializer.CloseFile()
    serializer.OpenFile(str(filename), True)
    restored = Geom.UnstructuredMesh.read_native_archive(mpi, serializer, "mesh")
    serializer.CloseFile()
    assert _check(restored) == expected
    # Repeating an already completed family is an agreed setup error.
    with pytest.raises(RuntimeError):
        Geom.with_native_topology(restored)
    # Reader-only IO failure must not poison the next collective operation.
    with pytest.raises(RuntimeError):
        Geom.read_native_cgns(mpi, path + ".missing", 2, reader_rank=mpi.size - 1)
    retry = Geom.read_native_cgns(mpi, path, 2, reader_rank=mpi.size - 1)
    assert retry.NumCellGlobal() == 1
