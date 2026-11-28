"""Tests for the DNDSR Cauchy Monitor Hub, multi-run bookkeeping, and live steering."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import pytest

from DNDSR.Cauchy.monitor import RunCatalog, create_app, find_available_port

REPO_ROOT = Path(__file__).resolve().parents[2]
CAUCHY_EXE = REPO_ROOT / "build" / "app" / "cauchy.exe"
STATIC_DIR = REPO_ROOT / "python" / "DNDSR" / "Cauchy" / "monitor" / "static"


def _write_fake_finished_run(base_dir: Path, name: str = "demo") -> Path:
    prefix = base_dir / name
    resolved = {
        "schemaVersion": 1,
        "domains": [{"name": "fluid", "ranks": [0], "settings": {}}],
        "time": {"method": "bdf1", "steady": True, "dt": 0.01, "steps": 1},
        "output": {"prefix": str(prefix)},
    }
    (base_dir /
     f"{name}-resolved.json").write_text(json.dumps(resolved), encoding="utf-8")
    hist_csv = base_dir / f"{name}-D0-history.csv"
    with hist_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "stage", "coupled_stage", "iteration", "attempt", "time", "dt", "wall", "status",
                    "res_rho", "residual_target_ratio", "pseudo_cfl", "linear_iterations"])
        w.writerow([1, 0, -1, 1, 0, 0.0, 0.01, 0.1, 0, 1.0e-1, 10.0, 10.0, 4])
        w.writerow([1, 0, -1, 2, 0, 0.0, 0.01, 0.2, 0, 2.0e-2, 2.0, 13.0, 5])
        w.writerow([1, -1, -1, -1, 0, 0.0, 0.01,
                   0.25, 2, 2.0e-2, 2.0, 13.0, 9])

    bc_csv = base_dir / f"{name}-D0-bc-Wall_Upper.csv"
    with bc_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "stage", "coupled_stage", "iteration", "attempt", "time", "dt", "wall", "status",
                    "area", "flux_rho", "flux_rhou", "flux_rhov", "flux_rhow", "flux_rhoE",
                    "force_t_x", "force_t_y", "force_t_z"])
        w.writerow([1, 0, -1, 1, 0, 0.0, 0.01, 0.1, 0, 2.0,
                   0.0, -0.5, -1.0, 0.0, 0.0, -0.05, 0.0, 0.0])
        w.writerow([1, 0, -1, 2, 0, 0.0, 0.01, 0.2, 0, 2.0,
                   0.0, -0.4, -1.0, 0.0, 0.0, -0.04, 0.0, 0.0])

    manifest = {
        "schema": "dndsr.cauchy.run-output.v1",
        "resolvedConfig": str(base_dir / f"{name}-resolved.json"),
        "historySchema": "dndsr.cauchy.detailed-history.v3",
        "domains": [{
            "index": 0,
            "name": "fluid",
            "history": str(hist_csv),
            "components": ["rho", "rhou", "rhov", "rhow", "rhoE"],
            "boundaryIntegrations": [{
                "zone": 7,
                "rawName": "Wall / Upper",
                "canonicalName": "Wall_Upper",
                "csv": str(bc_csv),
                "sign": "negative_outward_rhs_flux",
            }],
        }],
    }
    man_path = base_dir / f"{name}-run-manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    (base_dir / f"{name}-summary.json").write_text(
        json.dumps({"steps": 1, "time": 0.0, "wallSeconds": 0.25,
                   "status": "steady-converged", "steadyMode": True}),
        encoding="utf-8",
    )
    return prefix


def test_hub_port_auto_discovery_skips_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]
        chosen = find_available_port(
            "127.0.0.1", start_port=busy_port, scan_count=20)
        assert chosen > busy_port


def test_frontend_has_zero_cdn_and_serves_local_assets(tmp_path):
    for name in ("index.html", "styles.css", "app.js"):
        content = (STATIC_DIR / name).read_text(encoding="utf-8")
        # No external asset URLs or CDNs anywhere in HTML/CSS/JS
        assert not re.search(r'(?:src|href|url)\s*[:=(]\s*["\']?\s*(?:https?:)?//', content, re.I), (
            f"External asset/CDN URL found in {name}"
        )
        assert not re.search(r"https?://(?!127\.0\.0\.1)",
                             content), f"External URL found in {name}"

    app = create_app(
        watch_dirs=[],
        registry_dir=tmp_path / "reg",
        state_file=tmp_path / "state.json",
    )
    client = app.test_client()
    assert client.get("/").status_code == 200
    plotly_resp = client.get("/static/vendor/plotly.min.js")
    assert plotly_resp.status_code == 200
    assert len(plotly_resp.data) > 100_000


def test_hub_offline_bookkeeping_history_boundaries_and_auth(tmp_path):
    watch_dir = tmp_path / "runs_out"
    watch_dir.mkdir()
    _write_fake_finished_run(watch_dir, "airfoil")

    app = create_app(
        watch_dirs=[watch_dir],
        registry_dir=tmp_path / "reg",
        state_file=tmp_path / "hub_state.json",
        auth_token="secret-hub-key",
    )
    client = app.test_client()

    # Auth guard
    assert client.get("/api/runs").status_code == 401
    headers = {"Authorization": "Bearer secret-hub-key"}

    hub_info = client.get("/api/hub", headers=headers).get_json()
    assert hub_info["runCount"] == 1
    assert hub_info["plotlyAvailable"] is True

    runs = client.get("/api/runs", headers=headers).get_json()["runs"]
    assert len(runs) == 1
    rid = runs[0]["runId"]
    assert runs[0]["state"] == "steady-converged"
    assert runs[0]["online"] is False
    assert runs[0]["boundaryCount"] == 1

    # Disk discovery remains available, but history requires an HTTP endpoint.
    hist_nl = client.get(
        f"/api/runs/{rid}/history?domain=0&kind=nonlinear", headers=headers).get_json()
    assert hist_nl["totalRows"] == 0
    assert hist_nl["source"] == "unavailable"
    assert "companion" in hist_nl["warning"]

    # Per-BC flux & tangential force
    bnd = client.get(
        f"/api/runs/{rid}/boundaries?domain=0&bc=Wall_Upper", headers=headers).get_json()
    assert bnd["selected"]["canonicalName"] == "Wall_Upper"
    assert bnd["series"]["totalRows"] == 0
    assert bnd["series"]["source"] == "unavailable"

    # Persistence across catalog reload
    catalog2 = RunCatalog(watch_dirs=[], registry_dir=tmp_path /
                          "reg", state_file=tmp_path / "hub_state.json")
    assert any(r["runId"] == rid for r in catalog2.list_runs(rescan=False))

    # Detach removes from catalog without deleting disk files
    assert client.delete(f"/api/runs/{rid}",
                         headers=headers).status_code == 200
    assert (watch_dir / "airfoil-run-manifest.json").is_file()


@pytest.mark.skipif(not CAUCHY_EXE.is_file(), reason="cauchy.exe not built")
def test_hub_live_cauchy_steering_and_graceful_stop_e2e(tmp_path):
    """Launch real cauchy.exe, discover via registry, steer live config, and stop gracefully."""
    mesh = REPO_ROOT / "data" / "mesh" / "IV10_20.cgns"
    if not mesh.is_file():
        pytest.skip("IV10_20.cgns not available")

    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()
    prefix = tmp_path / "live_case"
    cfg_path = tmp_path / "case.json"
    cfg_path.write_text(
        json.dumps({
            "defaults": {
                "mesh": {
                    "file": str(mesh),
                    "dimension": 2,
                    "periodicLengths": [10, 10, 0],
                    "boundaryIds": {
                        "PERIODIC_1": -1,
                        "PERIODIC_1_DONOR": -4,
                        "PERIODIC_2": -2,
                        "PERIODIC_2_DONOR": -5,
                    },
                },
                "initialPrimitive": [1, 1, 1, 0, 1],
                "initialProfile": {"kind": "isentropicVortex", "strength": 1},
                "numerics": {"reconstruction": "variational", "cfv": {"maxOrder": 2}},
                "implicit": {
                    "iterations": 200,
                    "minIterations": 200,
                    "relativeTolerance": 1e-15,
                    "pseudoCFL": 10.0,
                    "maxPseudoCFL": 100.0,
                    "linear": {"iterations": 10},
                },
            },
            "time": {"method": "bdf1", "steady": False, "steps": 50, "dt": 1e-4},
            "output": {
                "prefix": str(prefix),
                "checkpointInterval": 0,
                "visualizationInterval": 0,
                "finalSnapshot": False,
                "implicit": {"historyInterval": 1, "consoleInterval": 0},
                "control": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 53200,
                    "portScanCount": 50,
                    "registerRun": True,
                    "registryDir": str(reg_dir),
                },
            },
        }),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["DNDS_RUN_REGISTRY_DIR"] = str(reg_dir)
    proc = subprocess.Popen(
        [str(CAUCHY_EXE), str(cfg_path)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        app = create_app(
            watch_dirs=[tmp_path],
            registry_dir=reg_dir,
            state_file=tmp_path / "hub_state.json",
        )
        client = app.test_client()

        # Wait for solver HTTP server to come online via registry discovery
        run_id = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            runs = client.get("/api/runs").get_json()["runs"]
            online = [r for r in runs if r["online"]]
            if online:
                run_id = online[0]["runId"]
                break
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(
                    f"cauchy.exe exited prematurely ({proc.returncode}):\n{out}")
            time.sleep(0.1)
        assert run_id is not None, "cauchy.exe HTTP control server did not register in time"

        detail = client.get(f"/api/runs/{run_id}/detail").get_json()
        assert detail["summary"]["online"] is True
        assert "output" in detail["mutableSchema"]["categories"]
        assert "steering" in detail["mutableSchema"]["categories"]
        # Ensure bearer token is never leaked to browser clients
        assert "token" not in detail["summary"]

        # Send an output-only live update at iteration boundary
        out_resp = client.patch(
            f"/api/runs/{run_id}/config",
            json={"applyAt": "iteration", "patch": {
                "/output/implicit/historyInterval": 2}},
        )
        assert out_resp.status_code == 202
        assert out_resp.get_json()["category"] == "output"

        # Send a simulation-steering live update at iteration boundary
        steer_resp = client.patch(
            f"/api/runs/{run_id}/config",
            json={"applyAt": "iteration", "patch": {
                "/implicit/pseudoCFL": 25.0}},
        )
        assert steer_resp.status_code == 202
        assert steer_resp.get_json()["category"] == "steering"

        # Wait briefly for patches to be applied collectively
        applied = False
        patch_deadline = time.time() + 15.0
        while time.time() < patch_deadline:
            d = client.get(f"/api/runs/{run_id}/detail").get_json()
            cfg = d.get("configuration") or {}
            if cfg.get("output", {}).get("implicit", {}).get("historyInterval") == 2:
                applied = True
                break
            time.sleep(0.1)
        assert applied, "live config patch was not applied"

        # Trigger graceful shutdown at unsteady iteration boundary
        stop_resp = client.post(
            f"/api/runs/{run_id}/stop", json={"applyAt": "iteration"})
        assert stop_resp.status_code == 202

        stdout, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, stdout

        summary = json.loads(
            (tmp_path / "live_case-summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "graceful-stop"
        assert summary["stopBoundary"] == "iteration"
        assert Path(summary["checkpoint"]).is_file()

        # After solver exits, registry entry is cleaned up and hub keeps run bookkept offline
        assert list(reg_dir.glob("*.json")) == []
        offline_detail = client.get(f"/api/runs/{run_id}/detail").get_json()
        assert offline_detail["summary"]["online"] is False
        assert offline_detail["summary"]["state"] == "graceful-stop"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
