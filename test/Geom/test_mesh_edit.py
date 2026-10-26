"""Staged owned mesh edits; rebuild/install bindings before execution."""

import gc
from pathlib import Path

import numpy as np
import pytest

from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import read_mesh, prepare_native_mesh


def _build_mesh():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    source = Path(__file__).resolve().parents[2] / "data/mesh/UniformSquare_10.cgns"
    mesh = prepare_native_mesh(read_mesh(str(source), mpi, 2, native=True).mesh)
    assert mesh.has_native_state
    return mesh


def test_current_authority_geometry_and_escaped_copies():
    mesh = _build_mesh()
    with mesh.acquire_read() as read:
        before = read.coordinates()
    snapshot = mesh.snapshot()
    with mesh.edit() as edit:
        escaped = edit.coord(0)
        for node in range(edit.num_nodes):
            edit.set_coord(node, edit.coord(node) + [0.25, 0.5, 0])
        edit.commit()
        assert edit.closed
    escaped[:] = [-1, -2, -3]
    committed = mesh.snapshot()
    # Supported native geometry entry points publish tracked replacements.
    # Keep the unsafe whole-array legacy callback explicitly rejected.
    with pytest.raises(RuntimeError):
        mesh.TransformCoords(lambda points: points.fill(99))
    assert committed.source_current
    for operation in (
        lambda: mesh.transform_coordinates(lambda p: p + 99),
        lambda: mesh.apply_geometry_transform(np.eye(3), 2),
        lambda: mesh.rectify_boundary_coordinates(lambda zone: 1),
        lambda: mesh.SetPeriodicGeometry(),
    ):
        with mesh.acquire_read():
            with pytest.raises(RuntimeError):
                operation()
        assert committed.source_current
    with pytest.raises(RuntimeError):
        mesh.SetPeriodicGeometry()  # Native frame setup keeps its legacy-only contract.
    assert committed.source_current
    expected = before + [0.25, 0.5, 0]
    with mesh.acquire_read() as read:
        np.testing.assert_allclose(read.coordinates(), expected)
    entities = mesh.entities(Geom.EntityKind.Node)
    mesh.transform_coordinates(lambda p: p + [1.0, 0.0, 0.0])
    expected = expected + [1.0, 0.0, 0.0]
    assert mesh.has_native_state and not committed.source_current
    assert entities.valid
    mesh.apply_geometry_transform(np.eye(3), 2)
    expected *= 2
    # This source has z=0; boundary rectification is an exact no-op.
    mesh.rectify_boundary_coordinates(lambda zone: 3)
    with mesh.acquire_read() as read:
        np.testing.assert_allclose(read.coordinates(), expected)
    with committed.acquire_read() as read:
        np.testing.assert_allclose(read.coordinates(), before + [0.25, 0.5, 0])
    with snapshot.acquire_read() as read:
        np.testing.assert_array_equal(read.coordinates(), before)
    with pytest.raises(RuntimeError):
        edit.coord(0)


def test_explicit_commit_metadata_validation_and_abort():
    mesh = _build_mesh()
    with mesh.acquire_read() as read:
        before = read.coordinates()
        original_type = read.cell_info(0).getElemType()
    with mesh.edit() as edit:
        edit.set_coord(0, [9, 8, 7])
        leaked = edit.coord(0)
    leaked[:] = 0
    with mesh.acquire_read() as read:
        np.testing.assert_array_equal(read.coordinates(), before)
    with mesh.edit() as edit:
        invalid = edit.cell_info(0)
        invalid.setElemType(Geom.Elem.ElemType.UnknownElem)
        edit.set_cell_info(0, invalid)
        with pytest.raises(RuntimeError):
            edit.commit()
        invalid.setElemType(original_type)
        edit.set_cell_info(0, invalid)
        with pytest.raises(ValueError):
            edit.set_coord(0, [1, 2])
        with pytest.raises(RuntimeError):
            edit.set_coord(edit.num_nodes, [1, 2, 3])
        edit.set_coord(0, [np.nan, 0, 0])
        with pytest.raises(RuntimeError):
            edit.commit()
        edit.set_coord(0, before[0])
        info = edit.cell_info(0)
        info.zone = 12345
        edit.set_cell_info(0, info)
        edit.commit()
    info.zone = -999
    with mesh.acquire_read() as read:
        assert read.cell_info(0).zone == 12345
        assert read.cell_info(0).getElemType() == original_type


def test_neighbor_order_validation_and_reader_conflict():
    mesh = _build_mesh()
    with mesh.edit() as edit:
        for row in range(edit.num_cells):
            neighbors = edit.cell_neighbors(row)
            for slot, value in enumerate(neighbors[::-1]):
                edit.set_cell_neighbor(row, slot, int(value))
        edit.commit()
    with mesh.edit() as edit:
        for row in range(edit.num_cells):
            if edit.cell_neighbors(row).size:
                edit.set_cell_neighbor(row, 0, -123456789)
                break
        with pytest.raises(RuntimeError):
            edit.commit()
    edit = mesh.edit()
    read = mesh.acquire_read()
    escaped = read.coord(0)
    read.close()
    with pytest.raises(RuntimeError):
        edit.commit()
    assert edit.closed
    del escaped
    gc.collect()
    with mesh.acquire_read() as current:
        assert current.counts["cell"][0] == mesh.NumCell()
