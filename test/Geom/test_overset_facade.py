"""Headless geometry/identity checks for the overset demo's query helpers."""

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import prepare_mesh, read_mesh


def _cell_field(mpi, cells, width=1):
    row_size = 1 if width == 1 else "D"
    field = DNDS.ArrayEigenVectorPair(row_size)
    field.father = DNDS.ArrayEigenVector(row_size, init_args=(mpi,))
    field.son = DNDS.ArrayEigenVector(row_size, init_args=(mpi,))
    if width == 1:
        field.father.Resize(cells.owned)
    else:
        field.father.Resize(cells.owned, width)
    field.TransAttach()
    return field


def _fill_and_pull(field, cells, width=1):
    for row in range(cells.owned):
        np.asarray(field[row])[:] = cells.local_to_global(row) + np.arange(width) / 8 + 0.25
    field.trans.createMPITypes()
    field.trans.pullOnce()
    values = []
    for row in range(cells.local):
        expected = cells.local_to_global(row) + np.arange(width) / 8 + 0.25
        np.testing.assert_array_equal(field[row], expected)
        values.append(np.array(field[row], copy=True))
    return values


@pytest.mark.parametrize("periodic", [False, True])
def test_overset_travelling_geometry_uses_current_query_authority(mpi, monkeypatch, periodic):
    pytest.importorskip("scipy")
    pytest.importorskip("numba")
    demo = Path(__file__).parent / "OversetCart"
    monkeypatch.syspath_prepend(str(demo))
    # The demo file and its containing package share a name. Load the script
    # explicitly, preserving the names used by its existing sibling imports.
    for name, filename in (("utils", "utils.py"), ("OversetCart", "OversetCart.py")):
        spec = importlib.util.spec_from_file_location(name, demo / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    parts = sys.modules["OversetCart"]
    monkeypatch.delitem(sys.modules, "OversetCartUtil", raising=False)
    helpers = importlib.import_module("OversetCartUtil")
    source = Path(__file__).resolve().parents[2] / "data/mesh" / (
        "IV10_10.cgns" if periodic else "UniformSquare_10.cgns"
    )
    options = {"periodic_geometry": {"translation1": [10., 0., 0.], "translation2": [0., 10., 0.]}} if periodic else {}
    result = read_mesh(str(source), mpi=mpi, dim=2, **options)
    mesh = result.mesh
    prepare_mesh(mesh, result.reader, build_serial_out=False)
    part = parts.OversetPart2D(mpi, (np.eye(3), np.array([2., -3., 0.])))
    part._mesh = mesh

    def sample():
        with mesh.acquire_read(faces=True) as read:
            cells = mesh.entities(Geom.EntityKind.Cell)
            nodes = mesh.entities(Geom.EntityKind.Node)
            packed = [part._travelling_cell_from_read(read, cell) for cell in range(cells.owned)]
            for cell, item in enumerate(packed):
                assert item[2] == cells.local_to_global(cell)
                assert item[3] == [nodes.local_to_global(int(node)) for node in read.cell_nodes(cell)]
                np.testing.assert_allclose(item[4], part.coord_mesh_to_phy(read.cell_coords(cell).T))
            for boundary in range(read.counts["boundary"][0]):
                coords = read.boundary_coords(boundary)
                assert not coords.flags.writeable
                with pytest.raises(ValueError):
                    coords.setflags(write=True)
                face = int(read.boundary_faces(boundary)[0])
                if face >= 0:
                    # Boundary and face orientations can differ. Periodic
                    # physical copies may differ by a whole domain translation.
                    face_coords = read.face_coords(face)
                    delta = coords[:, None, :] - face_coords[None, :, :]
                    if periodic:
                        delta[:, :, :2] -= 10 * np.round(delta[:, :, :2] / 10)
                    np.testing.assert_allclose(np.linalg.norm(delta, axis=2).min(axis=1), 0, atol=1e-12)
                if not periodic:
                    np.testing.assert_allclose(coords, np.stack([read.coord(int(node)) for node in read.boundary_nodes(boundary)]))
                info = read.boundary_info(boundary)
                zone = info.zone
                info.zone = 999
                assert read.boundary_info(boundary).zone == zone
        boundaries = helpers.get_mesh_bnd_elems(part, is_phy=True)
        return packed, boundaries

    before = sample()
    legacy_cells = mesh.entities(Geom.EntityKind.Cell)
    legacy_field = _cell_field(mpi, legacy_cells)
    helpers._borrow_cell_field_indexing(legacy_field, mesh)
    legacy_values = _fill_and_pull(legacy_field, legacy_cells)
    legacy_coords = mesh.coords
    legacy_cell_nodes = mesh.cell2node
    legacy_boundaries = mesh.bnd2node
    legacy_boundary_info = mesh.bndElemInfo
    mesh.publish_native(faces=True)
    # These are intentional raw compatibility edits, not managed mutations.
    for node in range(mesh.NumNodeProc()):
        np.asarray(legacy_coords[node])[:] = 73
    for cell in range(mesh.NumCellProc()):
        np.asarray(legacy_cell_nodes[cell])[:] = DNDS.UnInitIndex
    for boundary in range(mesh.NumBndProc()):
        np.asarray(legacy_boundaries[boundary])[:] = DNDS.UnInitIndex
        legacy_boundary_info[boundary, 0].zone = 999
    after = sample()
    for expected_items, actual_items in zip(before, after):
        assert len(actual_items) == len(expected_items)
        for expected, actual in zip(expected_items, actual_items):
            assert actual[:4] == expected[:4]
            np.testing.assert_allclose(actual[4], expected[4])
    native_cells = mesh.entities(Geom.EntityKind.Cell)
    legacy_cell_nodes.trans.LGhostMapping = None
    legacy_cell_nodes.trans.LGlobalMapping = None
    field = _cell_field(mpi, native_cells)
    helpers._borrow_cell_field_indexing(field, mesh)
    native_values = _fill_and_pull(field, native_cells)
    for expected, actual in zip(legacy_values, native_values):
        np.testing.assert_array_equal(actual, expected)
    dynamic = _cell_field(mpi, native_cells, width=2)
    Geom.borrow_legacy_field_indexing(dynamic.trans, native_cells)
    _fill_and_pull(dynamic, native_cells, width=2)
    # Cloned map export never exposes the canonical map objects.
    assert field.trans.LGhostMapping is not dynamic.trans.LGhostMapping
    assert field.trans.LGlobalMapping is not dynamic.trans.LGlobalMapping
    assert native_cells.same_order(mesh.entities(Geom.EntityKind.Cell))
    with pytest.raises(RuntimeError, match="stale"):
        Geom.borrow_legacy_field_indexing(field.trans, legacy_cells)
    wrong_size = _cell_field(mpi, native_cells)
    wrong_size.father.Resize(native_cells.owned + 1)
    with pytest.raises(RuntimeError, match="owned rows"):
        Geom.borrow_legacy_field_indexing(wrong_size.trans, native_cells)
    if mpi.size > 1:
        from mpi4py import MPI
        other_comm = DNDS.MPIInfo(MPI.COMM_SELF.handle)
        wrong_comm = _cell_field(other_comm, native_cells)
        with pytest.raises(RuntimeError, match="communicator"):
            Geom.borrow_legacy_field_indexing(wrong_comm.trans, native_cells)
    assert native_cells.same_order(mesh.entities(Geom.EntityKind.Cell))
