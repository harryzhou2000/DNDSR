"""Completed Python setup uses the owned construction sequence and read leases."""
from pathlib import Path

import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import read_mesh, prepare_mesh, prepare_native_mesh


def world():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return mpi


def mesh_file():
    return str(Path(__file__).resolve().parents[2] / "data/mesh/UniformSquare_10.cgns")


def ids(mesh, kind):
    local = [mesh.original_entity_id(kind, row) for row in range(mesh.entities(kind).owned)]
    return {value for rows in MPI.COMM_WORLD.allgather(local) for value in rows}


@pytest.mark.mpi
def test_python_setup_partition_precedes_refinement_and_retains_original_labels(monkeypatch):
    events, original = [], {}
    partition = Geom.partition_native_mesh
    elevate = Geom.elevate_native_mesh

    def record_partition(*args, **kwargs):
        events.append("partition")
        return partition(*args, **kwargs)

    def record_elevation(mesh, *args, **kwargs):
        events.append("elevation")
        for kind in (Geom.EntityKind.Node, Geom.EntityKind.Cell, Geom.EntityKind.Boundary):
            original[kind] = ids(mesh, kind)
        return elevate(mesh, *args, **kwargs)

    monkeypatch.setattr(Geom, "partition_native_mesh", record_partition)
    monkeypatch.setattr(Geom, "elevate_native_mesh", record_elevation)
    result = read_mesh(mesh_file(), mpi=world(), dim=2, elevation="O2", complete_node_star=True)
    assert events == ["partition", "elevation"]
    assert result.mesh.IsO2()
    assert ids(result.mesh, Geom.EntityKind.Cell) == original[Geom.EntityKind.Cell]
    assert ids(result.mesh, Geom.EntityKind.Boundary) == original[Geom.EntityKind.Boundary]
    assert original[Geom.EntityKind.Node] < ids(result.mesh, Geom.EntityKind.Node)


@pytest.mark.mpi
def test_owned_preparation_detaches_or_rejects_in_place_active_reads():
    mpi = world()
    source = Geom.read_native_cgns(mpi, mesh_file(), 2, complete_node_star=True)
    original_cells = ids(source, Geom.EntityKind.Cell)
    assert not Geom.native_mesh_requirements(source)["faces"]
    active = source.acquire_read(inverse=True)
    detached = prepare_native_mesh(source, reorder_parts=2)
    assert not Geom.native_mesh_requirements(source)["faces"]
    assert ids(detached, Geom.EntityKind.Cell) == original_cells
    assert sum(MPI.COMM_WORLD.allgather(detached.NumFace())) == 220
    assert ids(detached, Geom.EntityKind.Cell) == ids(source, Geom.EntityKind.Cell)
    with pytest.raises(RuntimeError):
        prepare_mesh(source, None, reorder_parts=2, build_serial_out=False)
    assert not Geom.native_mesh_requirements(source)["faces"]
    active.close()
    prepare_mesh(source, None, reorder_parts=2, build_serial_out=False)
    assert sum(MPI.COMM_WORLD.allgather(source.NumFace())) == 220
    assert ids(source, Geom.EntityKind.Cell) == original_cells


@pytest.mark.mpi
def test_combined_setup_rejects_rank_mismatches_before_publication():
    mpi = world()
    source = Geom.read_native_cgns(mpi, mesh_file(), 2)
    original_cells = ids(source, Geom.EntityKind.Cell)
    with pytest.raises(RuntimeError):
        Geom.prepare_native_mesh_setup(source, reorder_parts=-1)
    if mpi.size > 1:
        with pytest.raises(RuntimeError):
            Geom.prepare_native_mesh_setup(source, reorder_parts=1, in_place=mpi.rank == 0)
    assert not Geom.native_mesh_requirements(source)["faces"]
    assert ids(source, Geom.EntityKind.Cell) == original_cells
