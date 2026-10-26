"""Native compatibility views retain allocation, shape metadata and read leases."""

import gc
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R
from DNDSR.Geom.utils import prepare_native_mesh, read_mesh


def _new_mesh():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    source = Path(__file__).resolve().parents[2] / "data/mesh/UniformSquare_10.cgns"
    result = read_mesh(str(source), mpi, 2, native=True, complete_node_star=True)
    return prepare_native_mesh(result.mesh)


def _tet_mesh():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    def rows(values):
        array = R.Array(R.RowLayout.ragged([len(row) for row in values]), np.dtype(np.int64))
        with array.write() as write:
            for row, value in enumerate(values):
                write.assign_row(row, np.array(value, dtype=np.int64))
        return array

    def info(count, kind):
        array = R.Array(R.RowLayout.dense(count, 1), Geom.ElemInfo)
        with array.write() as write:
            for row in range(count):
                value = Geom.ElemInfo()
                value.setElemType(kind)
                write[row, 0] = value
        return array

    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [0., 0., -1.]]) if mpi.rank == 0 else np.empty((0, 3))
    coords = R.Array(R.RowLayout.dense(len(points), 3), np.dtype(np.float64))
    coords.assign(points)
    cells = [[0, 1, 2, 3]] if mpi.rank == 0 else []
    if mpi.rank == min(1, mpi.size - 1):
        cells.append([0, 2, 1, 4])
    source = Geom.build_native_mesh(mpi, 3, coords, rows(cells),
        info(len(cells), Geom.Elem.ElemType.Tet4), rows([]),
        info(0, Geom.Elem.ElemType.Tri3), complete_node_star=True)
    return prepare_native_mesh(source, edges=True)


@pytest.fixture
def mesh():
    return _new_mesh()


@pytest.mark.parametrize("dimension", [2, 3])
def test_all_incidence_aliases_and_owned_ghost_boundaries(dimension):
    mesh = _new_mesh() if dimension == 2 else _tet_mesh()
    aliases = {
        "cell2node": "CellNodes", "bnd2node": "BoundaryNodes",
        "bnd2cell": "BoundaryCells", "cell2cell": "CellCells",
        "cell2face": "CellFaces", "face2cell": "FaceCells",
        "face2node": "FaceNodes", "face2bnd": "FaceBoundaries",
        "bnd2face": "BoundaryFaces", "node2cell": "NodeCells",
        "node2bnd": "NodeBoundaries", "cell2edge": "CellEdges",
        "edge2cell": "EdgeCells", "edge2node": "EdgeNodes",
        "cell2cellFace": "CellFaceCells",
    }
    if dimension == 2:
        for alias in ("cell2edge", "edge2cell", "edge2node"):
            aliases.pop(alias)
    with mesh.acquire_read(faces=True, inverse=True, edges=dimension == 3, cell_face_neighbors=True) as read:
        for name, relation in aliases.items():
            pair = getattr(mesh, name)
            assert isinstance(pair, Geom.NativeMeshPairRead)
            assert pair.Size() == pair.father.Size() + pair.son.Size()
            with pytest.raises(IndexError):
                pair[-1]
            with pytest.raises(IndexError):
                pair[pair.Size()]
            offset = 0
            for part in (pair.father, pair.son):
                for row in range(part.Size()):
                    expected = read.relation(getattr(Geom.MeshRelation, relation), row + offset)
                    actual = part[row]
                    np.testing.assert_array_equal(actual, expected)
                    if len(actual):
                        assert np.shares_memory(actual, expected)
                    assert part.RowSize(row) == len(actual)
                    assert not actual.flags.writeable
                    for slot in range(len(actual)):
                        assert part[row, slot] == pair(row + offset, slot) == expected[slot]
                    with pytest.raises(IndexError):
                        part[row, len(actual)]
                offset += part.Size()


def test_coordinate_and_id_buffers_match_current_numbering(mesh):
    with mesh.acquire_read() as read:
        for part, start in ((mesh.coords.father, 0), (mesh.coords.son, mesh.NumNode())):
            values = part.data().reshape((-1, 3))
            assert not values.flags.writeable
            for row in range(part.Size()):
                np.testing.assert_array_equal(values[row], read.coord(row + start))
                assert np.shares_memory(values[row], part[row])
                assert np.shares_memory(values[row], read.coord(row + start))
        for name, kind in (("cell2cellOrig", Geom.EntityKind.Cell), ("node2nodeOrig", Geom.EntityKind.Node)):
            pair = getattr(mesh, name)
            all_ids = pair.data()
            assert all_ids.dtype == np.dtype(np.int64)
            for row in range(pair.Size()):
                assert all_ids[row] == pair[row, 0] == mesh.original_entity_id(kind, row)
            np.testing.assert_array_equal(all_ids[:pair.father.Size()], pair.father.data())
            np.testing.assert_array_equal(all_ids[pair.father.Size():], pair.son.data())
        if mesh.NumNodeGhost():
            with pytest.raises(ValueError, match="separate allocations"):
                mesh.coords.data()
    with pytest.raises(ValueError, match="wall distance"):
        mesh.nodeWallDist
    with pytest.raises(ValueError, match="periodic"):
        mesh.cell2nodePbi


def test_escaped_rows_and_whole_buffers_block_mutation_and_survive_mesh_owner():
    mesh = _new_mesh()
    whole = mesh.coords.father.data().reshape((-1, 3))
    escaped = np.asarray(whole[::-1, :2])
    expected = escaped.copy()
    del whole
    gc.collect()
    with pytest.raises(ValueError):
        escaped.setflags(write=True)
    with pytest.raises(RuntimeError):
        mesh.ReorderLocalCells(2, 1)
    del escaped
    gc.collect()
    mesh.ReorderLocalCells(2, 1)
    escaped = mesh.coords.father.data().reshape((-1, 3))[::-1, :2]
    # Cell-only reordering preserves node order and retained coordinate values.
    np.testing.assert_array_equal(escaped, expected)
    row = mesh.cell2node[0][::-1]
    row_expected = row.copy()
    del mesh
    gc.collect()
    np.testing.assert_array_equal(escaped, expected)
    np.testing.assert_array_equal(row, row_expected)


def test_metadata_is_readonly_retained_and_edit_uses_detached_values(mesh):
    info = mesh.cellElemInfo[0]
    zone = info.zone
    element = info.getElemType()
    with pytest.raises(AttributeError):
        info.zone = zone + 1
    with pytest.raises(ValueError, match="POD"):
        mesh.cellElemInfo.father.data()
    detached = info.snapshot()
    detached.zone = zone + 17
    assert info.zone == zone
    assert info.getElemType() == element
    with mesh.edit() as edit:
        edit.set_cell_info(0, detached)
        with pytest.raises(RuntimeError):
            edit.commit()
    del info
    gc.collect()
    with mesh.edit() as edit:
        edit.set_cell_info(0, detached)
        edit.commit()
    detached.zone = -999
    assert mesh.cellElemInfo[0].zone == zone + 17


def test_rank_local_escaped_id_rejects_collective_edit_and_allows_retry(mesh):
    before = mesh.original_entity_id(Geom.EntityKind.Cell, 0)
    held = mesh.cell2cellOrig.data()[::2] if MPI.COMM_WORLD.rank == 0 else None
    with mesh.edit() as edit:
        with pytest.raises(RuntimeError):
            edit.commit()
    del held
    gc.collect()
    with mesh.edit() as edit:
        edit.commit()
    assert mesh.original_entity_id(Geom.EntityKind.Cell, 0) == before


def test_empty_parts_ragged_buffers_and_metadata_after_mesh_deletion():
    mesh = _tet_mesh()
    for part in (mesh.coords.father, mesh.coords.son):
        assert part.data().size == 3 * part.Size()
        assert not part.data().flags.writeable
    # All physical boundary arrays are present but empty, including ghost rows.
    assert mesh.bnd2node.data().size == 0
    assert mesh.bnd2node.father.data().dtype == np.dtype(np.int64)
    pair = mesh.node2cell.father
    widths = [pair.RowSize(row) for row in range(pair.Size())]
    if len(set(widths)) > 1:
        with pytest.raises(ValueError, match="uniform contiguous"):
            pair.data()
    info = mesh.cellElemInfo[0] if mesh.NumCellProc() else None
    expected = None if info is None else (info.zone, info.getElemType())
    del pair, part, mesh
    gc.collect()
    if info is not None:
        assert (info.zone, info.getElemType()) == expected


def test_periodic_image_aliases_and_retained_wall_vectors(mesh):
    source = Path(__file__).resolve().parents[2] / "data/mesh/IV10_10.cgns"
    result = read_mesh(str(source), mesh.getMPI(), 2, native=True,
        periodic_geometry={"translation1": [10., 0., 0.], "translation2": [0., 10., 0.]})
    periodic = prepare_native_mesh(result.mesh)
    with periodic.acquire_read(faces=True) as read:
        for name, method in (("cell2nodePbi", read.cell_node_periodic_bits),
                             ("face2nodePbi", read.face_node_periodic_bits)):
            pair = getattr(periodic, name)
            for row in range(pair.Size()):
                np.testing.assert_array_equal(pair[row], method(row))
                if pair.RowSize(row):
                    assert np.shares_memory(pair[row], method(row))
                assert pair[row].dtype == np.dtype(np.uint8)
                assert not pair[row].flags.writeable
        for row in range(periodic.cell2facePbi.Size()):
            assert periodic.cell2facePbi.RowSize(row) == periodic.cell2face.RowSize(row)
        assert periodic.bnd2nodePbi.Size() == periodic.bnd2node.Size()
    # A native setup operator produces wall vectors; zero-copy aliases must
    # continue reading the old snapshot after the mesh Python owner disappears.
    walls = prepare_native_mesh(mesh, wall_dist_predicate=lambda zone: True,
        wall_dist_options={"method": 1, "subdivide_quad": 2, "minWallDist": 0.0})
    retained = walls.nodeWallDist.father.data()
    assert retained.size == 3 * walls.NumNode()
    assert np.isfinite(retained).all() and not retained.flags.writeable
    expected = retained.copy()
    del walls
    gc.collect()
    np.testing.assert_array_equal(retained, expected)
