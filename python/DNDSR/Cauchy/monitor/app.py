"""Flask-based cross-platform web hub for monitoring and steering Cauchy runs."""

from __future__ import annotations

import hmac
import json
from pathlib import Path
import socket
import time
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from .catalog import RunCatalog

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _find_plotly_bundle() -> Path | None:
    try:
        import plotly  # type: ignore

        candidate = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    return None


def find_available_port(host: str = "127.0.0.1", start_port: int = 53100, scan_count: int = 100) -> int:
    """Find a free TCP port scanning sequentially from start_port, then OS ephemeral."""
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    if start_port > 0:
        end_port = min(65535, start_port + max(1, scan_count) - 1)
        for port in range(start_port, end_port + 1):
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, port))
                    return int(sock.getsockname()[1])
                except OSError:
                    continue
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def create_app(
    *,
    watch_dirs: list[str | Path] | None = None,
    registry_dir: str | Path | None = None,
    state_file: str | Path | None = None,
    auth_token: str | None = None,
    catalog: RunCatalog | None = None,
) -> Flask:
    """Create the Cauchy Monitor Hub Flask application."""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    hub_catalog = catalog or RunCatalog(
        watch_dirs=watch_dirs,
        registry_dir=registry_dir,
        state_file=state_file,
    )
    plotly_path = _find_plotly_bundle()
    app.config["CAUCHY_CATALOG"] = hub_catalog
    app.config["CAUCHY_AUTH_TOKEN"] = auth_token or ""

    def _authorized() -> bool:
        expected = app.config["CAUCHY_AUTH_TOKEN"]
        if not expected:
            return True
        header = request.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else request.headers.get("X-Hub-Token", "")
        if not supplied:
            supplied = request.args.get("token", "")
        return hmac.compare_digest(supplied, expected)

    @app.before_request
    def _check_hub_auth():
        if request.path.startswith("/api/") and not _authorized():
            return jsonify({"error": "unauthorized", "message": "hub access token required"}), 401
        return None

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/vendor/plotly.min.js")
    def vendor_plotly():
        if plotly_path and plotly_path.is_file():
            resp = send_file(plotly_path, mimetype="application/javascript")
            resp.headers["Cache-Control"] = "public, max-age=86400"
            return resp
        return Response("// plotly not installed; canvas fallback active\n",
                        status=404, mimetype="application/javascript")

    @app.get("/api/hub")
    def hub_info():
        runs = hub_catalog.list_runs(rescan=True)
        return jsonify({
            "schema": "dndsr.cauchy.hub.v1",
            "plotlyAvailable": bool(plotly_path),
            "watchDirs": hub_catalog.watch_dirs,
            "registryDir": hub_catalog.registry_dir,
            "runCount": len(runs),
            "onlineCount": sum(1 for r in runs if r["online"]),
        })

    @app.get("/api/runs")
    def list_runs():
        rescan = request.args.get("rescan", "1") != "0"
        return jsonify({"runs": hub_catalog.list_runs(rescan=rescan)})

    @app.post("/api/runs/attach")
    def attach_run():
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "invalid_json", "message": "request body must be an object"}), 400
        try:
            summary = hub_catalog.attach(
                path=body.get("path") or None,
                url=body.get("url") or None,
                token=body.get("token") or None,
                label=body.get("label") or None,
            )
            return jsonify(summary), 201
        except ValueError as err:
            return jsonify({"error": "attach_failed", "message": str(err)}), 400

    @app.delete("/api/runs/<run_id>")
    def detach_run(run_id: str):
        if hub_catalog.detach(run_id):
            return jsonify({"detached": True, "runId": run_id})
        return jsonify({"error": "not_found"}), 404

    @app.get("/api/runs/<run_id>/detail")
    def run_detail(run_id: str):
        try:
            return jsonify(hub_catalog.get_run_detail(run_id))
        except KeyError:
            return jsonify({"error": "not_found"}), 404

    @app.get("/api/runs/<run_id>/history")
    def run_history(run_id: str):
        try:
            domain = int(request.args.get("domain", "0"))
            kind = request.args.get("kind", "all")
            max_points = int(request.args.get("maxPoints", "2000"))
        except ValueError:
            return jsonify({"error": "invalid_query"}), 400
        try:
            return jsonify(hub_catalog.get_history(run_id, domain=domain, kind=kind, max_points=max_points))
        except KeyError:
            return jsonify({"error": "not_found"}), 404

    @app.get("/api/runs/<run_id>/boundaries")
    def run_boundaries(run_id: str):
        try:
            domain = int(request.args.get("domain", "0"))
            bc_name = request.args.get("bc") or None
            max_points = int(request.args.get("maxPoints", "2000"))
        except ValueError:
            return jsonify({"error": "invalid_query"}), 400
        try:
            return jsonify(
                hub_catalog.get_boundary_history(run_id, domain=domain, bc_name=bc_name, max_points=max_points)
            )
        except KeyError:
            return jsonify({"error": "not_found"}), 404

    @app.get("/api/runs/<run_id>/stream")
    def run_stream(run_id: str):
        interval = max(0.5, min(10.0, float(request.args.get("interval", "2.0"))))

        def generate():
            while True:
                try:
                    detail = hub_catalog.get_run_detail(run_id)
                except KeyError:
                    yield f"event: error\ndata: {json.dumps({'error': 'not_found'})}\n\n"
                    break
                yield f"event: status\ndata: {json.dumps(detail)}\n\n"
                if not detail["summary"]["online"]:
                    break
                time.sleep(interval)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/<run_id>/stop")
    def stop_run(run_id: str):
        body = request.get_json(silent=True) or {}
        apply_at = str(body.get("applyAt", "auto")) if isinstance(body, dict) else "auto"
        code, data = hub_catalog.send_stop(run_id, apply_at=apply_at)
        return jsonify(data), code

    @app.post("/api/runs/<run_id>/snapshot")
    def snapshot_run(run_id: str):
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "invalid_json"}), 400
        code, data = hub_catalog.send_snapshot(
            run_id,
            checkpoint=bool(body.get("checkpoint", True)),
            visualization=bool(body.get("visualization", True)),
            apply_at=str(body.get("applyAt", "step")),
        )
        return jsonify(data), code

    @app.route("/api/runs/<run_id>/config", methods=["PATCH", "POST"])
    def patch_run_config(run_id: str):
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "invalid_json"}), 400
        patch = body.get("patch") if isinstance(body.get("patch"), dict) else {
            k: v for k, v in body.items() if k != "applyAt"
        }
        apply_at = str(body.get("applyAt", "step"))
        code, data = hub_catalog.send_config_patch(run_id, patch=patch, apply_at=apply_at)
        return jsonify(data), code

    return app


def serve_hub(
    *,
    host: str = "127.0.0.1",
    port: int = 53100,
    scan_count: int = 100,
    watch_dirs: list[str | Path] | None = None,
    registry_dir: str | Path | None = None,
    state_file: str | Path | None = None,
    auth_token: str | None = None,
) -> None:
    """Start the threaded hub HTTP server on an auto-discovered free port."""
    chosen_port = find_available_port(host=host, start_port=port, scan_count=scan_count)
    app = create_app(
        watch_dirs=watch_dirs,
        registry_dir=registry_dir,
        state_file=state_file,
        auth_token=auth_token,
    )
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"DNDSR Cauchy Monitor Hub listening on http://{display_host}:{chosen_port}", flush=True)
    if host == "127.0.0.1":
        print(f"  Remote SSH tunnel: ssh -L {chosen_port}:127.0.0.1:{chosen_port} <user>@<host>", flush=True)
    app.run(host=host, port=chosen_port, threaded=True, use_reloader=False)
