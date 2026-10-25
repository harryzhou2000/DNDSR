"""Native Python construction; requires rebuilt and installed DNDS/Geom modules."""

import gc

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R


def _world():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return mpi


def _rows(rows, dtype=np.int64):
    array = R.Array(R.RowLayout.ragged([len(row) for row in rows]), np.dtype(dtype))
    with array.write() as write:
        for index, row in enumerate(rows):
            write.assign_row(index, np.asarray(row, dtype=dtype))
    return array


def _info(element_type, zone, count):
    array = R.Array(R.RowLayout.dense(count, 1), Geom.ElemInfo)
    with array.write() as write:
        for row in range(count):
            value = Geom.ElemInfo()
            value.setElemType(element_type)
            value.zone = zone
            write[row, 0] = value
    return array


def _input(boundaries=False, empty_ranks=False):
    mpi = _world()
    owned = not empty_ranks or mpi.rank == 0
    count = 3 if owned else 0
    coordinates = R.Array(R.RowLayout.dense(count, 3, 4), np.dtype(np.float64))
    values = np.zeros((count, 3), dtype=np.float64)
    if owned:
        first = 0 if empty_ranks else 3 * mpi.rank
        values[:, 0] = np.arange(first, first + 3)
        values[2, 1] = 1
    coordinates.assign(values)
    next_rank = 0 if empty_ranks else (mpi.rank + 1) % mpi.size
    cell_nodes = _rows([[3 * next_rank, 3 * next_rank + 1, 3 * next_rank + 2]] if owned else [])
    boundary_nodes = _rows([[3 * mpi.rank, 3 * mpi.rank + 1]] if owned and boundaries else [])
    return dict(
        mpi=mpi,
        dimension=2,
        coordinates=coordinates,
        cell_nodes=cell_nodes,
        cell_info=_info(Geom.Elem.ElemType.Tri3, -1, int(owned)),
        boundary_nodes=boundary_nodes,
        boundary_info=_info(Geom.Elem.ElemType.Line2, 2 + mpi.rank, int(owned and boundaries)),
    )


def _ids(count, first, family):
    values = R.Array(R.RowLayout.dense(count, 1), np.dtype(np.int64))
    values.assign((10**12 + 17 * np.arange(first, first + count) + family).astype(np.int64).reshape(count, 1))
    return values


def _variable_uniform(source, width, padded):
    count = source.layout.rows
    layout = R.RowLayout.padded([width] * count, width + 2) if padded else R.RowLayout.ragged([width] * count)
    target = R.Array(layout, source.element_type or source.dtype)
    with target.write() as write:
        for row in range(count):
            for column in range(width):
                write[row, column] = source[row, column]
    return target


@pytest.mark.mpi
@pytest.mark.parametrize("complete", [False, True])
def test_remote_node_star_and_owned_global_csr(complete):
    data = _input()
    mpi = data["mpi"]
    previous = (mpi.rank + mpi.size - 1) % mpi.size
    mesh = Geom.build_native_mesh(**data, complete_node_star=complete)
    assert mesh.has_native_state
    assert mesh.NumCell() == 1
    assert mesh.NumCellGhost() == int(complete and mpi.size > 1)
    assert mesh.NumNodeGhost() == (3 if mpi.size > 1 else 0)
    cells = mesh.entities(Geom.EntityKind.Cell)
    nodes = mesh.entities(Geom.EntityKind.Node)
    with mesh.acquire_read(inverse=True) as read:
        assert read.cell_nodes(0).shape == (3,)
        next_rank = (mpi.rank + 1) % mpi.size
        assert [nodes.local_to_global(int(row)) for row in read.cell_nodes(0)] == list(range(3 * next_rank, 3 * next_rank + 3))
        for node in range(3):
            support = read.node_cells(node)
            assert support.shape == (1,)
            if complete or mpi.size == 1:
                assert cells.local_to_global(int(support[0])) == previous
            else:
                assert support[0] == -1 - previous
        for row in range(nodes.local):
            global_row = nodes.local_to_global(row)
            assert read.coord(row)[0] == global_row
            assert mesh.original_entity_id(Geom.EntityKind.Node, row) == global_row


@pytest.mark.mpi
def test_independent_boundaries_supplied_ids_and_pod_metadata():
    data = _input(boundaries=True)
    mpi = data["mpi"]
    data["node_ids"] = _ids(3, 3 * mpi.rank, 3)
    data["cell_ids"] = _ids(1, mpi.rank, 0)
    data["boundary_ids"] = _ids(1, mpi.rank, 4)
    cell_info = data["cell_info"]
    assert cell_info.element_type is Geom.ElemInfo
    snapshot = cell_info[0, 0]
    assert isinstance(snapshot, Geom.ElemInfo)
    snapshot.zone = 23
    assert cell_info[0, 0].zone == -1
    cell_info[0, 0] = snapshot
    assert cell_info[0, 0].zone == 23
    cell_info.element(0, 0).zone = 24
    mesh = Geom.build_native_mesh(**data)
    boundaries = mesh.entities(Geom.EntityKind.Boundary)
    cells = mesh.entities(Geom.EntityKind.Cell)
    with mesh.acquire_read(inverse=True) as read:
        assert read.cell_info(0).zone == 24
        adjacent = read.boundary_cells(0)
        assert adjacent.shape == (2,)
        assert cells.local_to_global(int(adjacent[0])) == (mpi.rank + mpi.size - 1) % mpi.size
        assert adjacent[1] == np.iinfo(np.int64).min
        for row in range(boundaries.local):
            global_row = boundaries.local_to_global(row)
            assert read.boundary_info(row).zone == 2 + global_row
            assert mesh.original_entity_id(Geom.EntityKind.Boundary, row) == 10**12 + 17 * global_row + 4
    with pytest.raises(RuntimeError):
        cell_info.element(0, 0).zone = 99


@pytest.mark.mpi
def test_shared_input_buffers_and_escaped_rows_retain_protection():
    data = _input()
    coordinates, metadata = data["coordinates"], data["cell_info"]
    mesh = Geom.build_native_mesh(**data, complete_node_star=True)
    read = mesh.acquire_read(inverse=True)
    with coordinates.read() as owned:
        assert read.coord(0).__array_interface__["data"][0] == owned.unsafe_address()
    escaped_coord = read.coord(0)[::2]
    escaped_topology = read.cell_nodes(0)[::-1]
    expected_coord = escaped_coord.copy()
    expected_topology = escaped_topology.copy()
    read.close()
    del mesh, read
    gc.collect()
    for array in (coordinates, metadata):
        with pytest.raises(RuntimeError):
            array.write()
    np.testing.assert_array_equal(escaped_coord, expected_coord)
    np.testing.assert_array_equal(escaped_topology, expected_topology)
    with pytest.raises(ValueError):
        escaped_coord.setflags(write=True)
    del escaped_coord, escaped_topology
    gc.collect()
    with coordinates.write() as write:
        write[0, 0] = -7.0
    metadata.element(0, 0).zone = 25
    assert coordinates[0, 0] == -7.0
    assert metadata[0, 0].zone == 25


@pytest.mark.mpi
@pytest.mark.parametrize("padded", [False, True])
def test_uniform_csr_and_padded_companions_share_original_values(padded):
    data = _input(boundaries=True)
    for key, width in [("coordinates", 3), ("cell_info", 1), ("boundary_info", 1)]:
        data[key] = _variable_uniform(data[key], width, padded)
    coordinates, metadata = data["coordinates"], data["cell_info"]
    mesh = Geom.build_native_mesh(**data)
    with mesh.acquire_read(inverse=True) as read:
        with coordinates.read() as owned:
            assert read.coord(0).__array_interface__["data"][0] == owned.unsafe_address()
        np.testing.assert_array_equal(read.coord(0), [3 * data["mpi"].rank, 0, 0])
        assert read.cell_info(0).getElemType() == Geom.Elem.ElemType.Tri3
    assert coordinates.layout.is_variable
    assert metadata.layout.is_variable
    for array in (coordinates, metadata):
        with pytest.raises(RuntimeError):
            array.write()


@pytest.mark.mpi
@pytest.mark.parametrize("failure", ["index_dtype", "opaque_metadata", "non_array", "coordinate_width", "coordinate_csr_width"])
def test_one_rank_conversion_failure_is_collective_and_retry_succeeds(failure):
    data = _input()
    bad = dict(data)
    if MPI.COMM_WORLD.rank == 0:
        if failure == "index_dtype":
            bad["cell_nodes"] = _rows([[0, 1, 2]], np.int32)
        elif failure == "opaque_metadata":
            bad["cell_info"] = R.Array(R.RowLayout.dense(1, 1), np.dtype([("type", "i4"), ("zone", "i4")]))
        elif failure == "non_array":
            bad["cell_info"] = np.zeros((1, 1), dtype=np.dtype([("type", "i4"), ("zone", "i4")]))
        elif failure == "coordinate_width":
            bad["coordinates"] = R.Array(R.RowLayout.dense(3, 2), np.dtype(np.float64))
        else:
            bad["coordinates"] = R.Array(R.RowLayout.ragged([3, 2, 3]), np.dtype(np.float64))
    with pytest.raises(RuntimeError, match="input validation failed"):
        Geom.build_native_mesh(**bad)
    mesh = Geom.build_native_mesh(**data, complete_node_star=True)
    with mesh.acquire_read(inverse=True) as read:
        assert read.cell_nodes(0).size == 3


@pytest.mark.mpi
def test_native_builder_participates_on_empty_ranks():
    data = _input(empty_ranks=True)
    mesh = Geom.build_native_mesh(**data, complete_node_star=True)
    assert mesh.NumCell() == int(MPI.COMM_WORLD.rank == 0)
    assert mesh.NumCellGlobal() == 1
    assert mesh.NumCellGhost() == 0
