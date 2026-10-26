"""Native EulerP setup agrees with the established numerical evaluator route."""
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.EulerP.EulerP_Solver import Solver
from DNDSR.Geom.utils import read_mesh, prepare_mesh


def _evaluate(mpi, native):
    solver = Solver(mpi)
    source = Path(__file__).resolve().parents[2] / "data/mesh/Uniform32_Periodic.cgns"
    geometry = {"translation1": [1., 0., 0.], "translation2": [0., 1., 0.]}
    if native:
        solver.ReadMesh(str(source), 2, other_options={"periodic_geometry": geometry})
    else:
        # Test-only numerical baseline: the old volume setup still feeds the
        # evaluator. Retired legacy boundary/output producers are not required
        # to compare exactly the same physical RHS on the two volume layouts.
        result = read_mesh(str(source), mpi, 2, native=False, periodic_geometry=geometry)
        prepare_mesh(result.mesh, result.reader, build_serial_out=False)
        solver.mesh, solver.reader, solver.name2Id = result.mesh, result.reader, result.name_to_id
        solver.meshBnd = solver.readerBnd = None
    solver.BuildFV({"intOrder": 1, "maxOrder": 1})
    solver.BuildEval()
    solver.WrapEval()
    solver.BuildDataArray()
    solver.WrapData()
    solver.data["u"].setConstant(np.array([1., 0., 0., 0., 2.5]).reshape(-1, 1))
    for cell in range(solver.mesh.NumCell()):
        x = solver.fv.GetCellBary(cell)
        density = 1 + 0.1 * x[0] + 0.05 * x[1]
        solver.data["u"][cell] = np.array([density, 0.1, 0.05, 0., 2.8]).reshape(-1, 1)
    for _, array in solver.data.items():
        array.trans.initPersistentPull(solver.runningDevice)
    solver.data["u"].trans.startPersistentPull(solver.runningDevice)
    solver.data["u"].trans.waitPersistentPull(solver.runningDevice)
    solver.CalculateOneRHS(zero_grad=True)
    for cell in range(solver.mesh.NumCellProc()):
        np.testing.assert_array_equal(np.asarray(solver.data["uGrad"][cell]), 0.0)
        np.testing.assert_array_equal(np.asarray(solver.data["uGradPrim"][cell]), 0.0)
    catalog = {}
    for cell in range(solver.mesh.NumCell()):
        point = tuple(np.round(solver.fv.GetCellBary(cell), 12))
        catalog[point] = np.asarray(solver.data["rhs"][cell]).reshape(-1).copy()
    combined = {}
    for part in MPI.COMM_WORLD.allgather(catalog):
        combined.update(part)
    return solver, combined


@pytest.mark.mpi
def test_native_solver_setup_evaluator_rhs_matches_legacy():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    with pytest.raises(ValueError, match="native=False is retired"):
        Solver(mpi).ReadMesh("unused.cgns", 2, native=False)
    legacy, expected = _evaluate(mpi, False)
    native, actual = _evaluate(mpi, True)
    assert native.mesh.has_native_state and native.meshBnd.has_native_state
    assert native.reader.mesh is native.mesh and native.reader.dataIsSerialOut
    assert native.readerBnd.mesh is native.meshBnd and native.readerBnd.dataIsSerialOut
    assert not legacy.mesh.has_native_state
    assert native.name2Id.n2id_map == legacy.name2Id.n2id_map
    assert set(actual) == set(expected) and len(actual) == 1024
    assert any(np.linalg.norm(value) > 1e-8 for value in actual.values())
    for point, value in actual.items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, expected[point], rtol=1e-10, atol=1e-10)
