"""Unified mesh helpers for reading, preparing, and serializing meshes.

See ``docs/dev/mesh_helpers_design.md`` for the design document.

Public API
----------
read_mesh           -- any source (CGNS / DNDSR H5) to distributed mesh
prepare_mesh        -- distributed mesh to solver-ready (faces, ghost N2CB, ...)
build_bnd_mesh      -- extract boundary surface mesh
build_fv            -- finite volume construction
serialize_mesh      -- write partitioned mesh to H5
mesh_h5_path        -- build conventional H5 filename

Backward-compatible wrappers (deprecated)
-----------------------------------------
create_mesh_from_CGNS  -- calls read_mesh + prepare_mesh
create_bnd_mesh        -- calls build_bnd_mesh
default_serializer_factory
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Mapping
from typing import Callable

from DNDSR import DNDS, Geom


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MeshReadResult:
    """Result of :func:`read_mesh`; native construction has no legacy reader."""

    mesh: Geom.UnstructuredMesh
    reader: Geom.UnstructuredMeshSerialRW | None
    name_to_id: Geom.AutoAppendName2ID | None  # None for H5 modes


@dataclasses.dataclass
class BndMeshResult:
    """Result of :func:`build_bnd_mesh`."""

    mesh_bnd: Geom.UnstructuredMesh
    reader_bnd: Geom.UnstructuredMeshSerialRW | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARTITION_OPTIONS: dict = {
    "metisType": "KWAY",
    "metisUfactor": 20,
    "metisSeed": 0,
    "metisNcuts": 3,
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def default_serializer_factory() -> DNDS.Serializer.SerializerFactory:
    """Create an H5 SerializerFactory with sensible defaults."""
    fac = DNDS.Serializer.SerializerFactory()
    fac.from_dict(
        {
            "type": "H5",
            "hdfDeflateLevel": 0,
            "hdfChunkSize": 0,
            "hdfCollOnMeta": False,
            "hdfCollOnData": False,
            "jsonBinaryDeflateLevel": 5,
            "jsonUseCodecOnUInt8": True,
        }
    )
    return fac


def mesh_h5_path(
    base: str, mpi_size: int, elevation: str = "", bisect: int = 0
) -> str:
    """Build the conventional H5 filename for a partitioned mesh.

    Convention: ``{base}_part_{mpi_size}[_elevated][_bisectN].dnds.h5``
    """
    name = f"{base}_part_{mpi_size}"
    if elevation == "O2":
        name += "_elevated"
    if bisect > 0:
        name += f"_bisect{bisect}"
    name += ".dnds.h5"
    return name


def _detect_read_mode(mesh_file: str) -> str:
    """Infer read mode from file extension."""
    if mesh_file.endswith(".dnds.h5"):
        return "h5"
    if mesh_file.endswith(".cgns"):
        return "cgns"
    raise ValueError(
        f"Cannot auto-detect read mode for '{mesh_file}'. "
        "Pass read_mode='cgns' or read_mode='h5' explicitly."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _native_authority(mesh: Geom.UnstructuredMesh) -> bool:
    """Agree the dispatch branch before entering different collective paths."""
    import numpy as np
    mode = np.array(int(mesh.has_native_state), dtype=np.int32)
    lower, upper = np.zeros((), dtype=np.int32), np.zeros((), dtype=np.int32)
    mpi = mesh.getMPI()
    DNDS.MPI.Allreduce(mode, lower, "MPI_MIN", mpi)
    DNDS.MPI.Allreduce(mode, upper, "MPI_MAX", mpi)
    if int(lower) != int(upper):
        raise ValueError("Native/legacy mesh authority differs across ranks")
    return bool(mode)


def _build_ghost_primary(mesh: Geom.UnstructuredMesh) -> None:
    """Build connectivity and ghost for a freshly-distributed mesh.

    This 5-step sequence is identical for all read paths.
    """
    mesh.RecoverNode2CellAndNode2Bnd()
    mesh.RecoverCell2CellAndBnd2Cell()
    mesh.BuildGhostPrimary()
    mesh.AdjGlobal2LocalPrimary()
    mesh.AdjGlobal2LocalN2CB()


def _read_cgns(
    mesh: Geom.UnstructuredMesh,
    reader: Geom.UnstructuredMeshSerialRW,
    mesh_file: str,
    mpi: DNDS.MPIInfo,
    dim: int,
    periodic_tolerance: float,
    partition_options: dict,
    elevation: str,
    bisect: int,
) -> Geom.AutoAppendName2ID:
    """CGNS serial read + partition + optional elevation / bisection."""
    name_to_id = reader.ReadFromCGNSSerial(mesh_file)
    reader.Deduplicate1to1Periodic(periodic_tolerance)
    reader.BuildCell2Cell()
    reader.MeshPartitionCell2Cell(partition_options)
    reader.PartitionReorderToMeshCell2Cell()

    _build_ghost_primary(mesh)

    # --- Optional O2 elevation ------------------------------------------------
    if elevation == "O2":
        if bisect:
            raise ValueError(
                "bisect must be 0 when elevation='O2'.  "
                "Elevation and bisection are mutually exclusive."
            )
        if not mesh.IsO2():
            mesh_o2 = Geom.UnstructuredMesh(mpi, dim)
            mesh_o2.BuildO2FromO1Elevation(mesh)
            mesh_o2, mesh = mesh, mesh_o2  # noqa: F841
            reader.mesh = mesh
            _build_ghost_primary(mesh)

    # --- Optional bisection ---------------------------------------------------
    if not (0 <= bisect <= 4):
        raise ValueError(f"bisect must be 0..4, got {bisect}")
    for iteration in range(1, 1 + bisect):
        mesh_o2 = Geom.UnstructuredMesh(mpi, dim)
        mesh_o2.BuildO2FromO1Elevation(mesh)

        mesh_o2.RecoverNode2CellAndNode2Bnd()
        mesh_o2.RecoverCell2CellAndBnd2Cell()
        mesh_o2.BuildGhostPrimary()

        mesh_o1b = Geom.UnstructuredMesh(mpi, dim)
        mesh_o1b.BuildBisectO1FormO2(mesh_o2)

        mesh, mesh_o1b = mesh_o1b, mesh  # noqa: F841
        reader.mesh = mesh
        _build_ghost_primary(mesh)

        n_cell = mesh.NumCellGlobal()
        n_node = mesh.NumNodeGlobal()
        if mpi.rank == 0:
            print(
                f"Mesh Direct Bisect {iteration} done, nCell [{n_cell}], nNode [{n_node}]")

    return name_to_id


def _read_h5(
    mesh: Geom.UnstructuredMesh,
    mesh_file: str,
    mpi: DNDS.MPIInfo,
    partition_options: dict,
    serializer_factory: DNDS.Serializer.SerializerFactory,
) -> None:
    """H5 distributed read: even-split + ParMetis repartition."""
    _, mesh_part_path = serializer_factory.ModifyFilePath(
        mesh_file, mpi, "part_%d", True
    )
    serializer = serializer_factory.BuildSerializer(mpi)
    if mpi.rank == 0:
        print(
            f"Mesh reader: distributed read via [{serializer_factory.to_dict()['type']}]")
    serializer.OpenFile(mesh_part_path, True)
    mesh.ReadSerializeAndDistribute(serializer, "meshPart", partition_options)
    serializer.CloseFile()

    _build_ghost_primary(mesh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_mesh(
    mesh_file: str,
    mpi: DNDS.MPIInfo,
    dim: int,
    *,
    periodic_geometry: dict | None = None,
    periodic_tolerance: float = 1e-9,
    read_mode: str | None = None,
    partition_options: dict | None = None,
    elevation: str = "",
    bisect: int = 0,
    serializer_factory: DNDS.Serializer.SerializerFactory | None = None,
    native: bool = False,
    cell_layers: int = 1,
    complete_node_star: bool = False,
) -> MeshReadResult:
    """Read a mesh from any supported source and distribute it.

    Auto-detects read mode from file extension when *read_mode* is ``None``:

    * ``.cgns`` -- serial CGNS read + Metis partition
    * ``.dnds.h5`` -- distributed H5 read (even-split + ParMetis)

    Returns a :class:`MeshReadResult` with the distributed mesh (ghost cells
    built, local indices).  The mesh is **not** solver-ready: call
    :func:`prepare_mesh` next for legacy input. With ``native=True``, use
    ``mesh = prepare_native_mesh(result.mesh)`` and retain its returned facade;
    ``reader`` is None. Native CGNS uses even contiguous ownership, retains BC
    names on every rank, and rejects legacy partition/periodic options. Native
    H5 reads the full native archive with its saved ownership, not legacy H5.
    All ranks must supply the same helper options and participate in calls.
    """
    if native:
        if periodic_geometry is not None or periodic_tolerance != 1e-9:
            raise ValueError("Native reading does not apply periodic identification/geometry options")
        if partition_options is not None:
            raise ValueError("Native reading uses even contiguous ownership; partition_options require legacy reading")
        if elevation not in ("", "O2") or not 0 <= bisect <= 4 or (elevation == "O2" and bisect):
            raise ValueError("Use elevation='' or 'O2', bisect=0..4, without combining O2 and bisection")
        mode = read_mode or _detect_read_mode(mesh_file)
        if mode == "cgns":
            if serializer_factory is not None:
                raise ValueError("serializer_factory applies to native H5 archives only")
            mesh, names = Geom.read_native_cgns_with_names(
                mpi, mesh_file, dim, cell_layers=cell_layers, complete_node_star=complete_node_star)
        elif mode == "h5":
            if complete_node_star:
                raise ValueError("Native archive preserves its saved node-star requirement; complete_node_star cannot override it")
            factory = serializer_factory or default_serializer_factory()
            base = mesh_file[:-len(".dnds.h5")] if mesh_file.endswith(".dnds.h5") else mesh_file
            path, _ = factory.ModifyFilePath(base, mpi, "part_%d", True)
            serializer = factory.BuildSerializer(mpi)
            serializer.OpenFile(path, True)
            try:
                mesh = Geom.UnstructuredMesh.read_native_archive(mpi, serializer, "meshPart", ghost_layers=cell_layers)
            finally:
                serializer.CloseFile()
            if mesh.getDim() != dim:
                raise ValueError("Native archive dimension differs from requested dim")
            names = None
        else:
            raise ValueError(f"Unknown read_mode: {mode!r}")
        if elevation == "O2" and not mesh.IsO2():
            mesh = Geom.elevate_native_mesh(mesh, cell_layers, complete_node_star)
        for _ in range(bisect):
            if not mesh.IsO2():
                mesh = Geom.elevate_native_mesh(mesh, cell_layers, complete_node_star)
            mesh = Geom.bisect_native_mesh(mesh, cell_layers, complete_node_star)
        return MeshReadResult(mesh, None, names)
    if cell_layers != 1 or complete_node_star:
        raise ValueError("cell_layers and complete_node_star apply only to native reading")
    if read_mode is None:
        read_mode = _detect_read_mode(mesh_file)

    if periodic_geometry is None:
        periodic_geometry = {
            "translation1": [1, 0, 0],
            "rotationCenter1": [0, 0, 0],
            "eulerAngles3": [0, 0, 0],
        }

    if partition_options is None:
        partition_options = dict(_DEFAULT_PARTITION_OPTIONS)

    mesh = Geom.UnstructuredMesh(mpi, dim)
    reader = Geom.UnstructuredMeshSerialRW(mesh, 0)

    assert os.path.isfile(mesh_file), f"Mesh file not found: {mesh_file}"

    mesh.SetPeriodicGeometry(**periodic_geometry)

    name_to_id: Geom.AutoAppendName2ID | None = None

    if read_mode == "cgns":
        name_to_id = _read_cgns(
            mesh, reader, mesh_file, mpi, dim,
            periodic_tolerance, partition_options, elevation, bisect,
        )
        # reader.mesh may have been swapped by elevation/bisection; make sure
        # the returned reader points to the final mesh.
        mesh = reader.mesh
    elif read_mode == "h5":
        if serializer_factory is None:
            serializer_factory = default_serializer_factory()
        _read_h5(mesh, mesh_file, mpi, partition_options, serializer_factory)
    else:
        raise ValueError(
            f"Unknown read_mode: {read_mode!r}. Use 'cgns' or 'h5'."
        )

    return MeshReadResult(mesh=mesh, reader=reader, name_to_id=name_to_id)


def prepare_mesh(
    mesh: Geom.UnstructuredMesh,
    reader: Geom.UnstructuredMeshSerialRW | None,
    *,
    reorder_parts: int = 1,
    reorder_inner_parts: int = 1,
    build_serial_out: bool = True,
    wall_dist_predicate: Callable[[int], bool] | None = None,
    wall_dist_options: dict | None = None,
    coord_scale: float = 1.0,
    coord_rot_z_deg: float = 0.0,
    rectify_near_plane: int = 0,
    rectify_threshold: float = 1e-12,
) -> None:
    """Prepare a distributed mesh for solver use (mutates *mesh* in place).

    Steps: cell reorder, face interpolation, ghost N2CB, optional wall
    distance, optional serial output, periodic nodes, VTK connectivity,
    optional coordinate transforms.
    """
    if _native_authority(mesh):
        raise ValueError("Use prepare_native_mesh(mesh), which returns a new native facade")
    if reader is None:
        raise ValueError("Legacy preparation requires a serial reader")
    mpi = mesh.getMPI()

    # 1. Cell reordering
    mesh.ReorderLocalCells(nParts=reorder_parts,
                           nPartsInner=reorder_inner_parts)

    # 2. Face interpolation
    mesh.InterpolateFace()
    mesh.AssertOnFaces()

    # 3. Ghost N2CB cycle
    mesh.AdjLocal2GlobalN2CB()
    mesh.BuildGhostN2CB()
    mesh.AdjGlobal2LocalN2CB()
    print(f"{mpi.rank}, NumBndGhost {mesh.NumBndGhost()}")

    # 4. Wall distance (optional)
    if wall_dist_predicate is not None:
        opts = Geom.UnstructuredMesh.WallDistOptions()
        if wall_dist_options is not None:
            if isinstance(wall_dist_options, Mapping):
                for key, value in wall_dist_options.items():
                    setattr(opts, key, value)
            else:
                opts = wall_dist_options
        mesh.BuildNodeWallDist(wall_dist_predicate, opts)

    # 5. Serial output (optional)
    if build_serial_out:
        mesh.AdjLocal2GlobalPrimary()
        reader.BuildSerialOut()
        mesh.AdjGlobal2LocalPrimary()

    # 6. Coordinate transforms (all default to no-op)
    _apply_coord_transforms(
        mesh, coord_scale, coord_rot_z_deg,
        rectify_near_plane, rectify_threshold,
    )

    # 7. Periodic nodes + VTK
    mesh.RecreatePeriodicNodes()
    mesh.BuildVTKConnectivity()


def prepare_native_mesh(
    mesh: Geom.UnstructuredMesh,
    *,
    ghost_layers: int = 1,
    edges: bool = False,
    wall_dist_predicate: Callable[[int], bool] | None = None,
    wall_dist_options: dict | None = None,
    coord_scale: float = 1.0,
    coord_rot_z_deg: float = 0.0,
    rectify_near_plane: int = 0,
    rectify_threshold: float = 1e-12,
) -> Geom.UnstructuredMesh:
    """Return native face topology with optional transforms and wall vectors.

    Retain the returned facade. Source topology/geometry remain unchanged.
    All ranks must supply identical options and participate in this collective.
    This explicit route uses existing ownership and does not build legacy serial
    output caches or perform a partition reorder. Use native output/archive APIs.
    """
    if not _native_authority(mesh):
        raise ValueError("prepare_native_mesh requires native construction")
    if wall_dist_options is not None and wall_dist_predicate is None:
        raise ValueError("wall_dist_options requires wall_dist_predicate")
    # Native topology completion rejects overwriting existing families. A
    # no-new-family call still returns a detached facade with refreshed closure.
    requirements = Geom.native_mesh_requirements(mesh)
    prepared = Geom.with_native_topology(mesh, faces=not requirements["faces"], edges=edges and not requirements["edges"], ghost_layers=ghost_layers)
    _apply_coord_transforms(prepared, coord_scale, coord_rot_z_deg, rectify_near_plane, rectify_threshold)
    if wall_dist_predicate is not None:
        options = Geom.UnstructuredMesh.WallDistOptions()
        if isinstance(wall_dist_options, Mapping):
            for key, value in wall_dist_options.items():
                if not hasattr(options, key):
                    raise ValueError(f"Unknown wall distance option: {key}")
                setattr(options, key, value)
        elif wall_dist_options is not None:
            options = wall_dist_options
        prepared.BuildNodeWallDist(wall_dist_predicate, options)
    return prepared


def _apply_coord_transforms(
    mesh: Geom.UnstructuredMesh,
    scale: float,
    rot_z_deg: float,
    rectify_near_plane: int,
    rectify_threshold: float,
) -> None:
    """Apply coordinate transforms matching the C++ solver order."""
    if rot_z_deg == 0.0 and scale == 1.0 and rectify_near_plane == 0:
        return

    import math
    import numpy as np

    if rot_z_deg != 0.0:
        rz = rot_z_deg / 180.0 * math.pi
        c, s = math.cos(rz), math.sin(rz)
        rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        mesh.apply_geometry_transform(rot_z)

    if scale != 1.0:
        mesh.apply_geometry_transform(np.eye(3), scale)

    if rectify_near_plane != 0:
        def rectify(c):
            if rectify_near_plane & 1:
                c[0] = 0.0 if abs(c[0]) < rectify_threshold else c[0]
            if rectify_near_plane & 2:
                c[1] = 0.0 if abs(c[1]) < rectify_threshold else c[1]
            if rectify_near_plane & 4:
                c[2] = 0.0 if abs(c[2]) < rectify_threshold else c[2]
        mesh.transform_coordinates(rectify)


def build_bnd_mesh(
    mesh: Geom.UnstructuredMesh,
    *,
    build_serial_out: bool | None = None,
) -> BndMeshResult:
    """Extract the boundary surface mesh from *mesh*.

    The default builds serial output for legacy input. Native input returns a
    compact native boundary mesh with ``reader_bnd=None``; explicitly requesting
    legacy serial output in native mode raises ValueError.
    """
    if _native_authority(mesh):
        if build_serial_out:
            raise ValueError("Native boundary extraction has no legacy serial reader/output cache")
        return BndMeshResult(Geom.extract_native_boundary(mesh), None)
    if build_serial_out is None:
        build_serial_out = True
    mesh_bnd = Geom.UnstructuredMesh(mesh.getMPI(), mesh.getDim() - 1)
    reader_bnd = Geom.UnstructuredMeshSerialRW(mesh_bnd, 0)
    mesh.ConstructBndMesh(mesh_bnd)
    if build_serial_out:
        mesh_bnd.AdjLocal2GlobalPrimaryForBnd()
        reader_bnd.BuildSerialOut()
        mesh_bnd.AdjGlobal2LocalPrimaryForBnd()
    mesh_bnd.RecreatePeriodicNodes()
    mesh_bnd.BuildVTKConnectivity()
    return BndMeshResult(mesh_bnd=mesh_bnd, reader_bnd=reader_bnd)


def build_fv(
    mpi: DNDS.MPIInfo,
    mesh: Geom.UnstructuredMesh,
    settings: dict | None = None,
):
    """Construct a :class:`CFV.FiniteVolume` with the full 14-step pipeline.

    *settings* is merged into the default FV settings.  Returns the
    constructed ``FiniteVolume``.
    """
    from DNDSR import CFV

    fv = CFV.FiniteVolume(mpi, mesh)
    merged = fv.GetSettings()
    if settings is not None:
        merged.update(settings)
    if mpi.rank == 0:
        print(f"FV Settings: \n{merged}")
    fv.ParseSettings(merged)

    # Cell construction
    fv.SetCellAtrBasic()
    fv.ConstructCellVolume()
    fv.ConstructCellBary()
    fv.ConstructCellCent()
    fv.ConstructCellIntJacobiDet()
    fv.ConstructCellIntPPhysics()
    fv.ConstructCellAlignedHBox()
    fv.ConstructCellMajorHBoxCoordInertia()

    # Face construction
    fv.SetFaceAtrBasic()
    fv.ConstructFaceCent()
    fv.ConstructFaceArea()
    fv.ConstructFaceIntJacobiDet()
    fv.ConstructFaceIntPPhysics()
    fv.ConstructFaceUnitNorm()
    fv.ConstructFaceMeanNorm()

    fv.ConstructCellSmoothScale()
    fv.finalize_metrics()

    n_bytes = fv.getArrayBytes() / 1024**2
    n_bytes_mesh = mesh.getArrayBytes() / 1024**2
    if mpi.rank == 0:
        print(f"Bytes     : {n_bytes:.4g} MB")
        print(f"Bytes mesh: {n_bytes_mesh:.4g} MB")

    return fv


def serialize_mesh(
    mesh: Geom.UnstructuredMesh,
    output_path: str,
    mpi: DNDS.MPIInfo,
    *,
    serializer_factory: DNDS.Serializer.SerializerFactory | None = None,
) -> None:
    """Write a partitioned mesh to H5 for later distributed read.

    Native input writes the full native archive (all published families, IDs,
    companions and ancestry), readable with ``read_mesh(native=True)``. Boundary
    name dictionaries are separate MeshReadResult metadata and are not archived.
    Legacy input keeps the C++ ``partitionMeshOnly`` serialization path.
    """
    native = _native_authority(mesh)
    if serializer_factory is None:
        serializer_factory = default_serializer_factory()
    if native and output_path.endswith(".dnds.h5"):
        output_path = output_path[:-len(".dnds.h5")]

    out_mod, out_path = serializer_factory.ModifyFilePath(
        output_path, mpi, "part_%d", False
    )
    serializer = serializer_factory.BuildSerializer(mpi)
    serializer.OpenFile(out_mod, False)
    try:
        if native:
            mesh.write_native_archive(serializer, "meshPart")
        else:
            mesh.AdjLocal2GlobalPrimary()
            try:
                mesh.WriteSerialize(serializer, "meshPart")
            finally:
                mesh.AdjGlobal2LocalPrimary()
    finally:
        serializer.CloseFile()



# ---------------------------------------------------------------------------
# Backward-compatible wrappers (deprecated)
# ---------------------------------------------------------------------------

def create_mesh_from_CGNS(
    meshFile: str,
    mpi: DNDS.MPIInfo,
    dim: int = 2,
    periodic_tolerance: float = 1e-9,
    inner_process_parts: int = 1,
    second_level_parts: int = 1,
    periodic_geometry={
        "translation1": [1, 0, 0],
        "rotationCenter1": [0, 0, 0],
        "eulerAngles3": [0, 0, 0],
    },
    readMeshMode: str = "Serial",
    meshElevation: str = "",
    meshDirectBisect: int = 0,
    serializerFactory: DNDS.Serializer.SerializerFactory = None,
    outPltMode: str = "Serial",
):
    """Read and prepare a mesh.  Deprecated -- use :func:`read_mesh` + :func:`prepare_mesh`.

    Preserves the original signature and return type ``(mesh, reader, name2ID)``
    for backward compatibility.
    """
    # Map old readMeshMode names to new read_mode
    _mode_map = {
        "Serial": "cgns",
        "Parallel": "h5",
        "Distributed": "h5",
    }
    if readMeshMode not in _mode_map:
        raise ValueError(f"Unknown readMeshMode: {readMeshMode}")
    read_mode = _mode_map[readMeshMode]

    # For H5 Parallel / Distributed modes, build the conventional path
    # and factory the same way the old code did.
    if serializerFactory is None:
        serializerFactory = default_serializer_factory()

    h5_file = meshFile
    if readMeshMode in ("Parallel", "Distributed"):
        # The old code built the H5 path from the CGNS filename + suffixes.
        # Replicate that convention here.
        h5_file = (
            meshFile
            + "_part_"
            + f"{mpi.size}"
            + ("_elevated" if meshElevation == "O2" else "")
            + (f"_bisect{meshDirectBisect}" if meshDirectBisect > 0 else "")
        )

    result = read_mesh(
        h5_file if read_mode == "h5" else meshFile,
        mpi,
        dim,
        periodic_geometry=periodic_geometry,
        periodic_tolerance=periodic_tolerance,
        read_mode=read_mode,
        elevation=meshElevation,
        bisect=meshDirectBisect,
        serializer_factory=serializerFactory if read_mode == "h5" else None,
    )

    prepare_mesh(
        result.mesh,
        result.reader,
        reorder_parts=inner_process_parts,
        reorder_inner_parts=second_level_parts,
        build_serial_out=(outPltMode == "Serial"),
    )

    # Return the old-style tuple.  name_to_id is None for H5 modes
    # (was a bug in the old code -- UnboundLocalError).
    return result.mesh, result.reader, result.name_to_id


def create_bnd_mesh(
    mesh: Geom.UnstructuredMesh, outPltMode: str = "Serial"
):
    """Extract boundary mesh.  Deprecated -- use :func:`build_bnd_mesh`.

    Preserves the original return type ``(meshBnd, readerBnd)``.
    """
    result = build_bnd_mesh(mesh, build_serial_out=(outPltMode == "Serial"))
    return result.mesh_bnd, result.reader_bnd
