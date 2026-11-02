"""CPU Cauchy frontend on the shared native mesh and Runtime Array backend.

Build/install cauchy_pybind11 and its dependencies before running under MPI.
These tests exercise the real residual/SSPRK3 path, not the placement mock.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS, Geom
from DNDSR.Cauchy import Domain


def build_domain(layout="aos", flux="Roe", workers=0, method="ssprk3"):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(path), 2))
    # Zone names/values come from the actual fixture, not rank-local assumptions.
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
    primitive = [1.2, .3, -.2, .4, 1.1]
    config = {"defaults": {
        "execution": {"layout": layout, "hostThreads": workers},
        "numerics": {"flux": flux, "cfv": {"maxOrder": 1}},
        "initialPrimitive": primitive,
        "boundaries": {str(zone): {"kind": "farfield", "primitive": primitive} for zone in zones},
    }, "time": {"method": method}}
    if method == "ditr":
        config["time"]["ditr"] = {"minIterations": 1}
    return Domain(mesh, json.dumps(config)), primitive


@pytest.mark.mpi
@pytest.mark.parametrize("layout,flux,workers", [("aos", "Roe", 0), ("soa", "AUSMPlusUp", 2), ("aosoa32", "HLLEP", 2)])
def test_shared_state_real_residual_ssprk3_and_restart(tmp_path, layout, flux, workers):
    domain, primitive = build_domain(layout, flux, workers)
    rho, u, v, w, p = primitive
    exact = [rho, rho*u, rho*v, rho*w, p/.4 + .5*rho*(u*u+v*v+w*w)]
    flow = domain.flow
    np.testing.assert_allclose(flow.numpy(), np.tile(exact, (flow.shape[0], 1)), atol=1e-14)
    assert domain.flow.identity == flow.identity
    assert domain.stable_step(.5) > 0
    np.testing.assert_allclose(domain.residual().numpy(), 0, atol=3e-13)
    domain.advance(.001)
    assert domain.step == 1
    assert domain.time == .001
    with flow.read():
        domain.prepare_step(.001)
        with pytest.raises(RuntimeError):
            domain.commit_step()
    domain.commit_step()
    assert domain.step == 2
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "cauchy.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"python"}')
    with pytest.raises(RuntimeError):
        domain.read_checkpoint(path, "{" if MPI.COMM_WORLD.rank == 0 else '{}')
    saved = flow.numpy().copy()
    domain.advance(.001)
    domain.read_checkpoint(path, '{"test":"python"}')
    assert domain.step == 2
    assert domain.time == .002
    np.testing.assert_array_equal(flow.numpy(), saved)
    # A shared Python handle retains its allocation after the domain dies.
    del domain
    np.testing.assert_array_equal(flow.numpy(), saved)


@pytest.mark.mpi
@pytest.mark.parametrize("method,stages", [("bdf2", 1), ("esdirk4", 5), ("ditr", 2)])
def test_configured_implicit_method_and_retained_shared_state(tmp_path, method, stages):
    domain, primitive = build_domain(method=method)
    flow = domain.flow
    before = flow.numpy().copy()
    domain.advance(.001)
    assert len(domain.implicit_stages) == stages
    if method == "ditr":
        assert domain.coupled_iterations == 1
        assert domain.coupled_target_ratio <= 1
    assert all(report.converged for report in domain.implicit_stages)
    assert domain.step == 1 and domain.time == .001
    np.testing.assert_allclose(flow.numpy(), before, atol=2e-14)
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "implicit.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"implicit-python"}', include_reconstruction=True)
    domain.advance(.002)
    domain.read_checkpoint(path, '{"test":"implicit-python"}')
    assert domain.step == 1 and domain.time == .001
    domain.advance(.002)
    assert all(report.converged for report in domain.implicit_stages)
    # The same Python Array handle continues to reference the bound allocation.
    assert domain.flow.identity == flow.identity
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)


@pytest.mark.mpi
def test_rank_local_invalid_config_is_collectively_rejected():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(path), 2))
    with pytest.raises(RuntimeError):
        Domain(mesh, "{" if MPI.COMM_WORLD.rank == 0 else "{}")
