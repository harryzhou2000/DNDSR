"""Explicit native helpers preserve labels and keep legacy defaults separate."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import (
    build_bnd_mesh, default_serializer_factory, prepare_mesh,
    prepare_native_mesh, read_mesh, serialize_mesh,
)

NAME = "CUSTOM_HELPER_WALL_19"


def _fixture(tmp_path):
    world = MPI.COMM_WORLD
    path, error = None, None
    if world.rank == 0:
        try:
            root = Path(__file__).resolve().parents[2]
            spec = importlib.util.spec_from_file_location("helpers_cgns", root / "scripts/generate_multiblock_cgns.py")
            generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generator)
            path = str(generator.generate(1, tmp_path, root))
            lib = generator._load_cgns(root)
            size = generator.cgsize_t
            lib.cg_boco_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, size,
                ctypes.POINTER(size), ctypes.POINTER(ctypes.c_int)]
            lib.cg_boco_gridlocation_write.argtypes = [ctypes.c_int] * 5
            handle = ctypes.c_int()
            generator._check(lib.cg_open(path.encode(), 2, ctypes.byref(handle)), "modify", lib)
            try:
                section, boundary = ctypes.c_int(), ctypes.c_int()
                generator._check(lib.cg_section_write(handle.value, 1, 1, b"Exterior", 3, 2, 5, 0,
                    generator._as_cgsize([1, 2, 2, 3, 3, 4, 4, 1]), ctypes.byref(section)), "edges", lib)
                generator._check(lib.cg_boco_write(handle.value, 1, 1, NAME.encode(), 1, 4, 2,
                    generator._as_cgsize([2, 5]), ctypes.byref(boundary)), "wall", lib)
                generator._check(lib.cg_boco_gridlocation_write(handle.value, 1, 1, boundary.value, 8), "edge-center", lib)
            finally:
                generator._check(lib.cg_close(handle.value), "close", lib)
        except Exception as exception:
            error = str(exception)
    path, error = world.bcast((path, error), root=0)
    assert error is None, error
    return path


@pytest.mark.mpi
def test_native_helpers_names_prepare_extract_and_restart(tmp_path):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = _fixture(tmp_path)
    # Original entry remains mesh-only; companion retains root-assigned names
    # even when another rank parses and most ranks have no owned cell.
    direct, names = Geom.read_native_cgns_with_names(mpi, path, 2, reader_rank=mpi.size - 1)
    assert direct.has_native_state
    assert NAME in names.n2id_map
    assert all(part == names.n2id_map for part in MPI.COMM_WORLD.allgather(names.n2id_map))
    result = read_mesh(path, mpi, 2, native=True, bisect=1)
    assert result.reader is None
    assert result.mesh.has_native_state
    assert result.mesh.NumCellGlobal() == 4
    assert result.mesh.NumNodeGlobal() == 9
    label = result.name_to_id.n2id_map[NAME]
    with pytest.raises(ValueError, match="prepare_native_mesh"):
        prepare_mesh(result.mesh, result.reader)
    before = [result.mesh.original_entity_id(Geom.EntityKind.Cell, row) for row in range(result.mesh.NumCell())]
    prepared = prepare_native_mesh(result.mesh, coord_scale=2.0,
        wall_dist_predicate=lambda zone: zone == label,
        wall_dist_options={"method": 1, "subdivide_quad": 4, "minWallDist": 0.0})
    assert [prepared.original_entity_id(Geom.EntityKind.Cell, row) for row in range(prepared.NumCell())] == before
    assert prepared.NumFaceGlobal() == 12
    with result.mesh.acquire_read() as source_read:
        for row in range(result.mesh.NumNodeProc()):
            assert np.max(source_read.coord(row)[:2]) <= 1.0
    assert not Geom.native_mesh_requirements(result.mesh)["faces"]
    assert not Geom.native_mesh_requirements(result.mesh)["wall_distance"]
    assert Geom.native_mesh_requirements(prepared)["wall_distance"]
    # Repeating preparation retains published numbering instead of overwriting
    # existing topology. Runtime handles keep the original result unchanged.
    again = prepare_native_mesh(prepared)
    assert again.NumFaceGlobal() == 12
    with again.acquire_read(faces=True, wall_distance=True) as read:
        for row in range(again.NumNodeProc()):
            p = read.coord(row)
            expected = min(p[0], 2.0 - p[0], p[1], 2.0 - p[1])
            np.testing.assert_allclose(np.linalg.norm(read.wall_distance(row)), expected, atol=1e-12)
        for row in range(again.NumBnd()):
            assert read.boundary_info(row).zone == label
    boundary = build_bnd_mesh(again)
    assert boundary.reader_bnd is None
    assert boundary.mesh_bnd.has_native_state
    assert boundary.mesh_bnd.NumCellGlobal() == 8
    with pytest.raises(ValueError, match="serial"):
        build_bnd_mesh(again, build_serial_out=True)
    for kind in ("H5", "JSON"):
        factory = default_serializer_factory()
        config = factory.to_dict()
        config["type"] = kind
        factory.from_dict(config)
        base = MPI.COMM_WORLD.bcast(str(tmp_path / ("helpers-" + kind)) if mpi.rank == 0 else None, root=0)
        serialize_mesh(again, base, mpi, serializer_factory=factory)
        restored = read_mesh(base + (".dnds.h5" if kind == "H5" else ""), mpi, 2,
            native=True, read_mode="h5", serializer_factory=factory)
        assert restored.reader is None and restored.name_to_id is None
        assert [restored.mesh.original_entity_id(Geom.EntityKind.Cell, row) for row in range(restored.mesh.NumCell())] == before
        assert Geom.native_mesh_requirements(restored.mesh)["wall_distance"]
        assert restored.mesh.NumFaceGlobal() == 12
        with restored.mesh.acquire_read(faces=True, wall_distance=True) as read:
            for row in range(restored.mesh.NumNodeProc()):
                p = read.coord(row)
                np.testing.assert_allclose(np.linalg.norm(read.wall_distance(row)), min(p[0], 2-p[0], p[1], 2-p[1]), atol=1e-12)


@pytest.mark.mpi
def test_native_helpers_reject_unsupported_options(tmp_path):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = _fixture(tmp_path)
    for options in ({"partition_options": {}}, {"periodic_geometry": {}},
                    {"periodic_tolerance": 1e-5}, {"elevation": "bad"},
                    {"elevation": "O2", "bisect": 1}, {"bisect": 5}):
        with pytest.raises(ValueError):
            read_mesh(path, mpi, 2, native=True, **options)
    primary = read_mesh(path, mpi, 2, native=True).mesh
    with pytest.raises(ValueError, match="requires wall_dist_predicate"):
        prepare_native_mesh(primary, wall_dist_options={"method": 0})


@pytest.mark.mpi
def test_native_helpers_reject_mixed_authority_before_dispatch(tmp_path):
    if MPI.COMM_WORLD.size == 1:
        pytest.skip("mixed authority requires multiple ranks")
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = _fixture(tmp_path)
    native = read_mesh(path, mpi, 2, native=True).mesh
    legacy_result = read_mesh(path, mpi, 2)
    mixed = legacy_result.mesh if mpi.rank == 0 else native
    for operation in (
        lambda: prepare_mesh(mixed, legacy_result.reader, build_serial_out=False),
        lambda: prepare_native_mesh(mixed),
        lambda: build_bnd_mesh(mixed),
        lambda: serialize_mesh(mixed, str(tmp_path / "rejected"), mpi),
    ):
        with pytest.raises(ValueError, match="authority differs"):
            operation()
