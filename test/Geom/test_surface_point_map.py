"""Geometry-only distributed search, independent of UnstructuredMesh."""
import gc

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R


@pytest.mark.mpi
@pytest.mark.parametrize("backend", [Geom.BoundingBoxBackend.Native, Geom.BoundingBoxBackend.CGAL])
def test_shared_surface_snapshot_and_warped_quad_projection(backend):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    owner = (rank + 1) % size
    coordinates = R.Array(R.RowLayout.dense(1, 12))
    coordinates.assign(np.array([[2*rank, 0, 1, 2*rank+1, 0, 1,
                                  2*rank+1, 1, 1.25, 2*rank, 1, 1]], dtype=float))
    surface = Geom.SurfacePatch.Create(
        [Geom.SurfaceElementRecord(rank, Geom.Elem.ElemType.Quad4)], coordinates, backend=backend)
    assert surface.Backend() == backend
    assert surface.WithBackend(backend).Backend() == backend
    with pytest.raises(RuntimeError):
        coordinates.assign(np.zeros((1, 12)))
    a, b = .23, .67
    expected = np.array([2*owner+a, b, 1+.25*a*b])
    normal = np.array([-.25*b, -.25*a, 1.])
    normal /= np.linalg.norm(normal)
    points = R.Array(R.RowLayout.dense(1, 3))
    ids = R.Array(R.RowLayout.dense(1, 1), np.int64)
    points.assign((expected + .02*normal).reshape(1, 3))
    ids.assign(np.array([[17]], dtype=np.int64))
    settings = Geom.SurfaceSearchSettings()
    settings.maxGap = .03
    settings.coarseBoxesPerRank = 3
    for mode in (Geom.SurfaceRoutingMode.AllGatherSurface, Geom.SurfaceRoutingMode.CoarseBoxes):
        mapping = Geom.DistributedSurfacePointMap.Match(mpi, surface, points, ids, mode, settings)
        match = mapping.Get(0)
        assert match.queryId == 17
        assert match.geometryRank == owner
        assert match.elementId == owner
        np.testing.assert_allclose(match.surfacePoint, expected, atol=1e-11, rtol=0)
        np.testing.assert_allclose(match.reference, [2*a-1, 2*b-1, 0], atol=1e-11, rtol=0)
        np.testing.assert_allclose(match.normal, normal, atol=1e-11, rtol=0)
        assert match.distance == pytest.approx(.02, abs=1e-11)
    del coordinates, surface
    gc.collect()
    # Map ownership retains the geometry allocation and metadata through the
    # immutable surface snapshot, after the original Python handles disappear.
    np.testing.assert_allclose(mapping.Get(0).surfacePoint, expected, atol=1e-11, rtol=0)
