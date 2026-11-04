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
from DNDSR.Cauchy import Domain, Simulation, PhysicalStepLimit, DerivedQuantity


@pytest.mark.mpi
def test_configured_mixture_fields_implicit_step_and_native_worker_refresh():
    import os

    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    root = Path(__file__).resolve().parents[2]
    mechanism = Path(os.environ.get("DNDS_MECH_PATH", root / "external/cfd_externals/install/data")) / "h2o2.yaml"
    if not mechanism.is_file():
        pytest.skip("installed h2o2 mechanism is unavailable")
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    with mesh.acquire_read(faces=True) as read:
        zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(zones))
    config = {"defaults": {
        "equation": {"eos": "idealGasMixture", "source": "chemical", "mixture": {
            "mechanismFile": str(mechanism), "rho0": 1.225, "U0": 340, "L0": 2, "T0": 288.15}},
        "initialPrimitive": [.8, 0, 0, 0, 1], "initialTemperature": 1400 / 288.15,
        "initialMassFractions": {"H2": .027, "O2": .22, "H2O": .02},
        "execution": {"layout": "soa", "hostThreads": 2, "overlap": True},
        "numerics": {"reconstruction": "constant", "faceFallback": False, "positivity": True},
        "implicit": {"diagonal": "block", "pseudoCFL": 1e4, "maxPseudoCFL": 1e8, "absoluteTolerance": 1e-8,
                     "residualComponents": ["rhoY:H2", "rhoY:H2O"],
                     "linear": {"method": "gmres", "preconditioner": "ilu"}},
        "boundaries": {str(zone): {"kind": "extrapolate"} for zone in zones},
    }, "time": {"method": "bdf1"}}
    try:
        domain = Domain(mesh, json.dumps(config))
    except RuntimeError as error:
        if "requires Cantera" in str(error):
            pytest.skip("Cantera is disabled in this build")
        raise
    fields = domain.fields
    assert set(fields) == {"flow", "species"}
    before_flow, before_species = fields["flow"].numpy().copy(), fields["species"].numpy().copy()
    residual = domain.residual_fields()
    assert set(residual) == set(fields)
    derived = domain.make_derived_fields([DerivedQuantity.Temperature, DerivedQuantity.Pressure])
    assert derived.refresh(domain)
    assert derived.current(domain)
    assert domain.stable_step(.5) > 0
    domain.advance(1e-6)
    np.testing.assert_allclose(fields["flow"].numpy(), before_flow, rtol=1e-11, atol=1e-12)
    local_change = np.max(np.abs(fields["species"].numpy() - before_species), initial=0.)
    assert MPI.COMM_WORLD.allreduce(local_change, op=MPI.MAX) > 1e-10
    assert all(len(report.residual) == 14 for report in domain.implicit_stages)
    assert not derived.current(domain)
    assert derived.refresh(domain)
    assert derived.current(domain)


def build_domain(layout="aos", flux="Roe", workers=0, method="ssprk3", steady=False, time_pp=False, lvr_variant=None):
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
    }, "time": {"method": method, "steady": steady, "positivity": {"enabled": time_pp}}}
    if method == "ditr":
        config["time"]["ditr"] = {"minIterations": 1}
    if lvr_variant is not None:
        config["defaults"]["numerics"].update({
            "reconstruction": "variational", "cfv": {"maxOrder": 3},
            "limiter": "LVR", "limitedVR": {"variant": lvr_variant},
            "positivity": True, "faceFallback": False,
        })
    return Domain(mesh, json.dumps(config)), primitive


@pytest.mark.mpi
@pytest.mark.parametrize("split", [False, True])
def test_coordinated_simulation_python_owners_and_reversed_subgroups(split):
    import gc

    world = MPI.COMM_WORLD
    if split and world.size == 1:
        pytest.skip("split-domain fixture needs at least two ranks")
    comm = world.Split(world.rank % 2, -world.rank) if split else world
    mesh_mpi = DNDS.MPIInfo(comm.handle)
    control = DNDS.MPIInfo(world.handle)
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mesh_mpi, str(path), 2))
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*world.allgather(local_zones))
    ranks = [list(reversed([r for r in range(world.size) if r % 2 == side])) if split
             else list(range(world.size)) for side in range(2)]
    meshes = [mesh if not split or world.rank % 2 == side else None for side in range(2)]
    config = {"domains": [{"name": str(side), "ranks": ranks[side]} for side in range(2)],
              "defaults": {"numerics": {"reconstruction": "constant"},
                           "boundaries": {str(zone): {"kind": "farfield"} for zone in zones}},
              "time": {"method": "bdf2"}}
    simulation = Simulation(meshes, control, json.dumps(config))
    simulation.advance(.001)
    simulation.advance(.001)
    assert simulation.step == 2
    assert simulation.interface_diagnostics().query_points == 0
    local = world.rank % 2 if split else 1
    assert simulation.local(local)
    domain = simulation.domain(local)
    assert domain.time == simulation.time
    # The receiver zone exists, but the donor selection does not. This reaches
    # the patch guard rather than the earlier unused-BC configuration check.
    bad = json.loads(json.dumps(config))
    receiver_zone = min(zones)
    bad["defaults"]["boundaries"][str(receiver_zone)] = {"kind": "interface"}
    bad["interfaces"] = [{"donor": str(side), "receiver": str(1-side),
                          "donorZone": 999, "receiverZone": receiver_zone} for side in range(2)]
    with pytest.raises(RuntimeError, match="globally empty"):
        Simulation(meshes, control, json.dumps(bad))
    # reference_internal retains the Simulation, while a shared native Array
    # independently retains its allocation after both frontend handles vanish.
    flow = domain.flow
    del simulation
    assert domain.step == 2
    del domain
    gc.collect()
    values = flow.numpy()
    np.testing.assert_allclose(values, np.tile([1, 0, 0, 0, 2.5], (values.shape[0], 1)), rtol=0, atol=2e-14)
    del values, flow, meshes, mesh
    gc.collect()
    if split:
        comm.Free()


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
@pytest.mark.parametrize("method,stages", [("bdf2", 1), ("esdirk4", 5), ("ditr", 1)])
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
    assert all(report.reconstruction_iterations == 0 and report.reconstruction_relative_update == 0
               for report in domain.implicit_stages)  # Direct gradients do not run iterative VR.
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


@pytest.mark.mpi
def test_explicit_derived_refresh_shared_storage_and_freshness():
    domain, primitive = build_domain(layout="soa")
    fields = domain.make_derived_fields([DerivedQuantity.Pressure, DerivedQuantity.Temperature])
    assert fields.quantities == [DerivedQuantity.Temperature, DerivedQuantity.Pressure]
    assert not fields.current(domain)
    assert fields.refresh(domain)
    rho, _, _, _, p = primitive
    np.testing.assert_allclose(fields.values.numpy(), np.tile([p/rho, p], (domain.flow.shape[0], 1)))
    assert fields.current(domain)
    assert not fields.refresh(domain, only_if_stale=True)
    retained = fields.values
    before = domain.flow.numpy().copy()
    if before.shape[0]:
        domain.flow[0, 4] = before[0, 4] + .1
        assert not fields.current(domain)
    fields.refresh(domain)
    assert retained.identity == fields.values.identity
    saved = retained.numpy().copy()
    del fields, domain
    np.testing.assert_array_equal(retained.numpy(), saved)


@pytest.mark.mpi
def test_steady_bdf1_override_keeps_the_physical_clock_frozen():
    domain, _ = build_domain(method="esdirk4", steady=True)
    before = domain.flow.numpy().copy()
    domain.advance(.03)
    assert domain.steady and domain.time == 0 and domain.step == 1
    assert len(domain.implicit_stages) == 1 and domain.implicit_stages[0].converged
    assert domain.previous_dt == 0
    np.testing.assert_allclose(domain.flow.numpy(), before, atol=3e-14)


@pytest.mark.mpi
@pytest.mark.parametrize("method", ["ssprk3", "bdf2"])
def test_physical_step_selection_retains_state_and_restores_growth_history(tmp_path, method):
    domain, _ = build_domain(layout="soa", workers=2, method=method, time_pp=True)
    flow = domain.flow
    before = flow.numpy().copy()
    assert domain.previous_dt == 0
    initial = domain.limit_physical_step(.01)
    assert isinstance(initial, PhysicalStepLimit)
    assert initial.dt == .01 and initial.minimum_alpha == 1
    domain.advance(.001)
    with flow.read():
        bound = domain.limit_physical_step(.1)
    assert bound.dt == .002 and bound.growth_limited
    assert bound.limited_cells == 0 and bound.minimum_alpha == 1
    assert domain.previous_dt == .001 and domain.time == .001 and domain.step == 1
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
    domain.prepare_step(.003)
    domain.discard_step()
    assert domain.previous_dt == .001
    assert domain.limit_physical_step(.1).dt == .002
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "time-step.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"physical-step-python"}')
    domain.advance(.004)
    assert domain.previous_dt == .004
    domain.read_checkpoint(path, '{"test":"physical-step-python"}')
    assert domain.previous_dt == .001
    assert domain.limit_physical_step(.1).dt == .002
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)


@pytest.mark.mpi
@pytest.mark.parametrize("variant,layout", [("v2", "soa"), ("v3", "aosoa32")])
def test_native_lvr_config_and_warm_restart_through_shared_python_arrays(tmp_path, variant, layout):
    domain, _ = build_domain(layout=layout, method="bdf1", lvr_variant=variant)
    flow = domain.flow
    before = flow.numpy().copy()
    domain.advance(.001)
    assert domain.step == 1 and domain.time == .001
    assert domain.implicit_stages[0].converged
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "lvr.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"lvr-python"}', include_reconstruction=True)
    domain.advance(.001)
    domain.read_checkpoint(path, '{"test":"lvr-python"}')
    assert domain.flow.identity == flow.identity
    assert domain.step == 1 and domain.time == .001
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
