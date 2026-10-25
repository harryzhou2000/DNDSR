"""Native boundary facade, exact node compaction and durable parent identity."""

import gc
import json

import numpy as np
import pytest

from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R


NODE, CELL, BOUNDARY = Geom.EntityKind.Node, Geom.EntityKind.Cell, Geom.EntityKind.Boundary


def _rows(values):
    result = R.Array(R.RowLayout.ragged([len(row) for row in values]), np.dtype(np.int64))
    with result.write() as write:
        for row, value in enumerate(values):
            write.assign_row(row, np.asarray(value, dtype=np.int64))
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


def _source(empty):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    active = not empty or mpi.rank == 0
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [8., 8., 8.]])
    points[:, 0] += 3 * mpi.rank
    coords = R.Array(R.RowLayout.dense(4 if active else 0, 3), np.dtype(np.float64))
    coords.assign(points if active else np.empty((0, 3)))
    owner = 0 if empty else (mpi.rank + 1) % mpi.size
    start = 4 * owner
    mesh = Geom.build_native_mesh(
        mpi, 2, coords, _rows([[start, start + 1, start + 2]] if active else []),
        _info(Geom.Elem.ElemType.Tri3, -1, int(active)),
        _rows([[start + 1, start]] if active else []),
        _info(Geom.Elem.ElemType.Line2, 12, int(active)),
        node_ids=_column([1000 + 4 * mpi.rank + slot for slot in range(4)] if active else []),
        boundary_ids=_column([2000 + mpi.rank] if active else []),
    )
    return mpi, mesh, owner, active


@pytest.mark.mpi
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("faces", [False, True])
def test_extract_native_boundary_keeps_source_reads_and_durable_ancestry(tmp_path, empty, faces):
    mpi, source, owner, active = _source(empty)
    if faces:
        source = Geom.with_native_topology(source)
    with source.acquire_read() as retained:
        before = retained.cell_nodes(0).copy() if active else None
        result = Geom.extract_native_boundary(source)
        if active:
            np.testing.assert_array_equal(retained.cell_nodes(0), before)
        destination = Geom.UnstructuredMesh(mpi, 1)
        source.ConstructBndMesh(destination)
        assert destination.has_native_state
        if active:
            np.testing.assert_array_equal(retained.cell_nodes(0), before)
    assert source.NumNode() == (4 if active else 0)
    assert result.NumNode() == (2 if active else 0)
    assert result.NumCell() == int(active)
    assert result.NumBnd() == 0
    with pytest.raises(RuntimeError):
        Geom.with_native_topology(result)
    metadata = Geom.native_mesh_provenance(result)
    assert metadata["source_kinds"] == {NODE: NODE, CELL: BOUNDARY}
    parents = metadata["source_ids"][NODE]
    with parents.read() as read:
        assert [int(read.row(row)[0]) for row in range(result.NumNode())] == ([1000 + 4 * mpi.rank, 1001 + 4 * mpi.rank] if active else [])
    cell_parents = metadata["source_ids"][CELL]
    del source
    gc.collect()
    with result.acquire_read() as read, cell_parents.read() as parent_read:
        if active:
            assert int(parent_read.row(0)[0]) == 2000 + mpi.rank
            expected = [0., 1.] if faces else [1., 0.]
            np.testing.assert_allclose([read.coord(int(node))[0] for node in read.cell_nodes(0)], np.asarray(expected) + 3 * owner)
            assert read.cell_info(0).zone == 12
    serializer = DNDS.Serializer.SerializerJSON()
    path = tmp_path / f"extracted-{mpi.rank}.json"
    serializer.OpenFile(str(path), False)
    result.write_native_archive(serializer, "mesh")
    serializer.CloseFile()
    archive = json.loads(json.loads(path.read_text())["mesh"]["metadata_json"])
    assert archive["dimension"] == 1
    assert archive["namespace"] != metadata["source_namespace"]
    if result.NumNode():
        with pytest.raises(RuntimeError):
            parents[0, 0] = -1
