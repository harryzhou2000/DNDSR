"""Independent child-cone and durable ancestry checks for native refinement."""

import gc
import json

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R


NODE, CELL, BOUNDARY = Geom.EntityKind.Node, Geom.EntityKind.Cell, Geom.EntityKind.Boundary
BASE = 10**12


def _array(rows, dtype=np.int64):
    result = R.Array(R.RowLayout.ragged([len(row) for row in rows]), np.dtype(dtype))
    with result.write() as write:
        for row, values in enumerate(rows):
            write.assign_row(row, np.asarray(values, dtype=dtype))
    return result


def _column(values):
    result = R.Array(R.RowLayout.dense(len(values), 1), np.dtype(np.int64))
    result.assign(np.asarray(values, dtype=np.int64).reshape(-1, 1))
    return result


def _info(kind, zone, count):
    result = R.Array(R.RowLayout.dense(count, 1), Geom.ElemInfo)
    with result.write() as write:
        for row in range(count):
            value = Geom.ElemInfo()
            value.setElemType(kind)
            value.zone = zone
            write[row, 0] = value
    return result


def _source(quad, empty):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    active = not empty or mpi.rank == 0
    corners = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]] if quad else
                       [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    width = len(corners)
    coordinates = R.Array(R.RowLayout.dense(width if active else 0, 3), np.dtype(np.float64))
    points = corners.copy() if active else np.empty((0, 3))
    points[:, 0] += 3 * mpi.rank
    coordinates.assign(points)
    next_rank = 0 if empty else (mpi.rank + 1) % mpi.size
    cone = [width * next_rank + slot for slot in range(width)]
    mesh = Geom.build_native_mesh(
        mpi, 2, coordinates, _array([cone] if active else []),
        _info(Geom.Elem.ElemType.Quad4 if quad else Geom.Elem.ElemType.Tri3, 31, int(active)),
        _array([cone[:2]] if active else []), _info(Geom.Elem.ElemType.Line2, 24, int(active)),
        complete_node_star=True,
        node_ids=_column([BASE + 1000 + 17 * (width * mpi.rank + slot) for slot in range(width)] if active else []),
        cell_ids=_column([BASE + 100 + mpi.rank] if active else []),
        boundary_ids=_column([BASE + 200 + mpi.rank] if active else []),
    )
    return mpi, mesh, coordinates, corners, cone, active


def _save(mesh, mpi, path):
    serializer = DNDS.Serializer.SerializerJSON()
    serializer.OpenFile(str(path), False)
    mesh.write_native_archive(serializer, "mesh")
    serializer.CloseFile()
    return json.loads(json.loads(path.read_text())["mesh"]["metadata_json"])


def _catalog(mesh, provenance):
    nodes = mesh.entities(NODE)
    parents = provenance["source_ids"][NODE]
    assert parents.layout.row_count == nodes.owned
    local = {}
    with mesh.acquire_read(inverse=True) as read, parents.read() as ancestry:
        for row in range(nodes.owned):
            local[mesh.original_entity_id(NODE, row)] = (tuple(int(value) for value in ancestry.row(row)), read.coord(row).copy())
    result = {}
    for part in MPI.COMM_WORLD.allgather(local):
        result.update(part)
    return result


@pytest.mark.mpi
@pytest.mark.parametrize("quad,empty", [(False, False), (True, False), (False, True)])
def test_native_elevation_bisection_exact_cones_ancestry_and_restart(tmp_path, quad, empty):
    mpi, source, coordinates, corners, cone, active = _source(quad, empty)
    assert source.IsO1() and not source.IsO2()
    assert Geom.native_mesh_provenance(source) is None
    source_metadata = _save(source, mpi, tmp_path / f"source-r{mpi.rank}.json")
    with pytest.raises(RuntimeError):
        Geom.bisect_native_mesh(source)
    elevated = Geom.elevate_native_mesh(source, complete_node_star=True)
    assert not elevated.IsO1() and elevated.IsO2()
    elevated_metadata = _save(elevated, mpi, tmp_path / f"elevated-r{mpi.rank}.json")
    assert elevated_metadata["namespace"] != source_metadata["namespace"]
    provenance = Geom.native_mesh_provenance(elevated)
    assert provenance["source_namespace"] == source_metadata["namespace"]
    assert provenance["source_kinds"] == {NODE: NODE, CELL: CELL, BOUNDARY: BOUNDARY}
    assert all(isinstance(value, R.Array) for value in provenance["source_ids"].values())
    catalog = _catalog(elevated, provenance)
    for parents, point in catalog.values():
        expected = []
        for identifier in parents:
            global_node = (identifier - BASE - 1000) // 17
            owner, slot = divmod(global_node, len(corners))
            expected.append(corners[slot] + [3 * owner, 0, 0])
        np.testing.assert_array_equal(point, np.mean(expected, axis=0))

    elevated_ids = []
    cells = elevated.entities(CELL)
    with elevated.acquire_read(inverse=True) as read:
        assert cells.owned == int(active)
        if active:
            elevated_ids = [elevated.original_entity_id(NODE, int(node)) for node in read.cell_nodes(0)]
            source_ids = [BASE + 1000 + 17 * node for node in cone]
            spans = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1, 2, 3)] if quad else [(0, 1), (1, 2), (2, 0)]
            expected = [(identifier,) for identifier in source_ids]
            expected += [tuple(source_ids[slot] for slot in span) for span in spans]
            assert [catalog[identifier][0] for identifier in elevated_ids] == expected
            assert provenance["source_ids"][CELL][0, 0] == BASE + 100 + mpi.rank
            assert provenance["source_ids"][BOUNDARY][0, 0] == BASE + 200 + mpi.rank
            assert read.cell_info(0).zone == 31
            assert read.boundary_info(0).zone == 24
    with pytest.raises(RuntimeError):
        Geom.elevate_native_mesh(elevated)
    parent_cell = elevated.original_entity_id(CELL, 0) if active else None
    parent_boundary = elevated.original_entity_id(BOUNDARY, 0) if active else None
    bisected = Geom.bisect_native_mesh(elevated, complete_node_star=True)
    assert bisected.IsO1() and not bisected.IsO2()
    child_provenance = Geom.native_mesh_provenance(bisected)
    assert child_provenance["source_namespace"] == elevated_metadata["namespace"]
    child_catalog = _catalog(bisected, child_provenance)
    for parents, point in child_catalog.values():
        assert len(parents) == 1
        np.testing.assert_array_equal(point, catalog[parents[0]][1])
    splits = [[0, 4, 8, 7], [4, 1, 5, 8], [7, 8, 6, 3], [8, 5, 2, 6]] if quad else \
             [[0, 3, 5], [3, 1, 4], [5, 3, 4], [5, 4, 2]]
    assert bisected.NumCell() == 4 * int(active)
    assert bisected.NumBnd() == 2 * int(active)
    with bisected.acquire_read(inverse=True) as read:
        for row, slots in enumerate(splits if active else []):
            actual = [child_catalog[bisected.original_entity_id(NODE, int(node))][0][0] for node in read.cell_nodes(row)]
            assert actual == [elevated_ids[slot] for slot in slots]
            assert child_provenance["source_ids"][CELL][row, 0] == parent_cell
            assert read.cell_info(row).zone == 31
        for row, slots in enumerate([[0, len(corners)], [len(corners), 1]] if active else []):
            actual = [child_catalog[bisected.original_entity_id(NODE, int(node))][0][0] for node in read.boundary_nodes(row)]
            assert actual == [elevated_ids[slot] for slot in slots]
            assert child_provenance["source_ids"][BOUNDARY][row, 0] == parent_boundary
            assert read.boundary_info(row).zone == 24

    filename = tmp_path / f"bisected-r{mpi.rank}.json"
    saved = _save(bisected, mpi, filename)
    assert saved["namespace"] not in [source_metadata["namespace"], elevated_metadata["namespace"]]
    serializer = DNDS.Serializer.SerializerJSON()
    serializer.OpenFile(str(filename), True)
    restored = Geom.UnstructuredMesh.read_native_archive(mpi, serializer, "mesh")
    serializer.CloseFile()
    replay = Geom.native_mesh_provenance(restored)
    assert restored.IsO1() and not restored.IsO2()
    assert replay["source_namespace"] == child_provenance["source_namespace"]
    assert replay["source_kinds"] == child_provenance["source_kinds"]
    for kind, array in child_provenance["source_ids"].items():
        with array.read() as before, replay["source_ids"][kind].read() as after:
            for row in range(array.layout.row_count):
                np.testing.assert_array_equal(before.row(row), after.row(row))
    del source, elevated, bisected, restored, provenance, replay, child_provenance
    gc.collect()
    # Refinement allocates new coordinates; original builder storage is released.
    with coordinates.write() as write:
        if active:
            write[0, 0] = -7


@pytest.mark.mpi
def test_provenance_array_independently_retains_shared_owner_after_mesh_release():
    mpi, source, coordinates, _, _, _ = _source(False, False)
    elevated = Geom.elevate_native_mesh(source)
    mesh = Geom.bisect_native_mesh(elevated)
    metadata = Geom.native_mesh_provenance(mesh)
    array = metadata["source_ids"][CELL]
    second = Geom.native_mesh_provenance(mesh)["source_ids"][CELL]
    with array.read() as first_read, second.read() as second_read:
        assert first_read.unsafe_address() == second_read.unsafe_address()
        expected = [first_read.row(row) for row in range(array.layout.row_count)]
    del metadata, second, mesh, elevated, source
    gc.collect()
    with array.read() as read:
        for row, values in enumerate(expected):
            np.testing.assert_array_equal(read.row(row), values)
    with pytest.raises(RuntimeError):
        array.write()
    del array
    gc.collect()
    with coordinates.write() as write:
        write[0, 0] = -7
