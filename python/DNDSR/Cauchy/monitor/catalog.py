"""Multi-run discovery, persistent bookkeeping, CSV parsing, and solver proxy."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def default_registry_dir() -> Path:
    env = os.environ.get("DNDS_RUN_REGISTRY_DIR")
    if env:
        return Path(env)
    return Path.home() / ".dndsr" / "runs"


def default_state_file() -> Path:
    env = os.environ.get("DNDS_HUB_STATE_FILE")
    if env:
        return Path(env)
    return Path.home() / ".dndsr" / "hub_catalog.json"


def _stable_id(seed: str) -> str:
    return "run-" + hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sorted_csv_segments(base_csv: Path) -> list[Path]:
    """Return segmented CsvLog files (<bare>.csv, <bare>_1.csv, <bare>_2.csv, ...) in order."""
    parent = base_csv.parent
    if not parent.is_dir():
        return []
    stem = base_csv.stem
    suffix = base_csv.suffix or ".csv"
    segments: list[tuple[int, Path]] = []
    if base_csv.is_file():
        segments.append((0, base_csv))
    pattern = str(parent / f"{glob.escape(stem)}_*{suffix}")
    for candidate_str in glob.glob(pattern):
        candidate = Path(candidate_str)
        tail = candidate.stem[len(stem) + 1:]
        if tail.isdigit():
            segments.append((int(tail), candidate))
    segments.sort(key=lambda pair: pair[0])
    return [p for _, p in segments]


def _parse_csv_value(raw: str) -> int | float | str | None:
    text = raw.strip()
    if not text:
        return None
    try:
        if "." not in text and "e" not in text and "E" not in text:
            return int(text)
    except ValueError:
        pass
    try:
        val = float(text)
        if math.isfinite(val):
            return val
        return None
    except ValueError:
        return text


def _read_segmented_csv(base_csv: Path) -> tuple[list[str], list[dict[str, Any]]]:
    segments = _sorted_csv_segments(base_csv)
    header: list[str] = []
    rows: list[dict[str, Any]] = []
    for seg in segments:
        try:
            with seg.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                if not header:
                    header = list(reader.fieldnames)
                for raw_row in reader:
                    parsed: dict[str, Any] = {}
                    for col, val in raw_row.items():
                        if col is None:
                            continue
                        parsed[col] = _parse_csv_value(val or "")
                    rows.append(parsed)
        except OSError:
            continue
    return header, rows


def _downsample_indices(total: int, max_points: int) -> list[int]:
    if max_points <= 0 or total <= max_points:
        return list(range(total))
    if max_points == 1:
        return [total - 1]
    step = (total - 1) / float(max_points - 1)
    indices = sorted({min(total - 1, int(round(i * step))) for i in range(max_points)})
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return indices


def _rows_to_columns(header: list[str], rows: list[dict[str, Any]], max_points: int) -> dict[str, Any]:
    total = len(rows)
    idxs = _downsample_indices(total, max_points)
    sampled = [rows[i] for i in idxs]
    columns: dict[str, list[Any]] = {col: [] for col in header}
    for row in sampled:
        for col in header:
            columns[col].append(row.get(col))
    return {
        "totalRows": total,
        "returnedRows": len(sampled),
        "downsampled": len(sampled) < total,
        "indices": idxs,
        "columns": columns,
    }


class RunCatalog:
    """Thread-safe multi-run registry, bookkeeper, and authenticated solver proxy."""

    def __init__(
        self,
        watch_dirs: list[str | Path] | None = None,
        registry_dir: str | Path | None = None,
        state_file: str | Path | None = None,
        http_timeout: float = 2.5,
    ) -> None:
        self._lock = threading.RLock()
        self._watch_dirs: list[Path] = [Path(d).expanduser() for d in (watch_dirs or [])]
        self._registry_dir: Path = Path(registry_dir).expanduser() if registry_dir else default_registry_dir()
        self._state_file: Path | None = Path(state_file).expanduser() if state_file else default_state_file()
        self._http_timeout = float(http_timeout)
        self._runs: dict[str, dict[str, Any]] = {}
        self._csv_cache: dict[str, tuple[tuple[tuple[str, int, int], ...], list[str], list[dict[str, Any]]]] = {}
        self._load_state()
        self.scan()

    @property
    def watch_dirs(self) -> list[str]:
        return [str(p) for p in self._watch_dirs]

    @property
    def registry_dir(self) -> str:
        return str(self._registry_dir)

    def _load_state(self) -> None:
        if not self._state_file:
            return
        data = _read_json_safe(self._state_file)
        if not isinstance(data, dict):
            return
        for entry in data.get("runs", []):
            if isinstance(entry, dict) and entry.get("runId"):
                self._runs[entry["runId"]] = entry

    def _save_state(self) -> None:
        if not self._state_file:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            payload = {"schema": "dndsr.cauchy.hub-catalog.v1", "runs": list(self._runs.values())}
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            tmp.replace(self._state_file)
        except OSError:
            pass

    def _upsert_from_control(self, data: dict[str, Any], source: str, control_file: Path | None = None) -> str:
        run_id = str(data.get("runId") or "")
        prefix = str(data.get("prefix") or "")
        url = str(data.get("url") or "")
        if not url and data.get("host") and data.get("port"):
            url = f"http://{data['host']}:{data['port']}"
        if not run_id:
            run_id = _stable_id(prefix or url or str(control_file or time.time()))
        existing = self._runs.get(run_id, {})
        record = {
            "runId": run_id,
            "label": existing.get("label") or (Path(prefix).name if prefix else run_id),
            "source": source,
            "prefix": prefix or existing.get("prefix", ""),
            "runManifest": str(data.get("runManifest") or existing.get("runManifest", "")),
            "controlFile": str(control_file) if control_file else existing.get("controlFile", ""),
            "workingDirectory": str(data.get("workingDirectory") or existing.get("workingDirectory", "")),
            "url": url or existing.get("url", ""),
            "token": str(data.get("token") or existing.get("token", "")),
            "pid": data.get("pid", existing.get("pid")),
            "startedAt": str(data.get("startedAt") or existing.get("startedAt", "")),
            "activeHint": bool(data.get("active", True)),
            "lastSeen": time.time(),
        }
        if not record["runManifest"] and record["prefix"]:
            record["runManifest"] = record["prefix"] + "-run-manifest.json"
        self._runs[run_id] = record
        return run_id

    def _upsert_from_manifest(self, manifest_path: Path, source: str) -> str | None:
        data = _read_json_safe(manifest_path)
        if not isinstance(data, dict) or data.get("schema") != "dndsr.cauchy.run-output.v1":
            return None
        name = manifest_path.name
        suffix = "-run-manifest.json"
        prefix = str(manifest_path)[:-len(suffix)] if name.endswith(suffix) else str(manifest_path.with_suffix(""))
        control_path = Path(prefix + "-control.json")
        if control_path.is_file():
            ctrl = _read_json_safe(control_path)
            if isinstance(ctrl, dict) and ctrl.get("schema") == "dndsr.cauchy.control.v1":
                return self._upsert_from_control(ctrl, source, control_path)
        # Check if already registered under this prefix
        abs_prefix = str(Path(prefix).resolve())
        for existing_id, rec in self._runs.items():
            if rec.get("prefix") and str(Path(rec["prefix"]).resolve()) == abs_prefix:
                rec["runManifest"] = str(manifest_path.resolve())
                rec["lastSeen"] = time.time()
                return existing_id
        run_id = _stable_id(abs_prefix)
        existing = self._runs.get(run_id, {})
        self._runs[run_id] = {
            "runId": run_id,
            "label": existing.get("label") or Path(prefix).name,
            "source": source,
            "prefix": abs_prefix,
            "runManifest": str(manifest_path.resolve()),
            "controlFile": existing.get("controlFile", ""),
            "workingDirectory": str(manifest_path.parent.resolve()),
            "url": existing.get("url", ""),
            "token": existing.get("token", ""),
            "pid": existing.get("pid"),
            "startedAt": existing.get("startedAt", ""),
            "activeHint": False,
            "lastSeen": time.time(),
        }
        return run_id

    def scan(self) -> None:
        """Refresh run list from registry directory, watch dirs, and known control files."""
        with self._lock:
            changed = False
            if self._registry_dir.is_dir():
                for item in sorted(self._registry_dir.glob("*.json")):
                    ctrl = _read_json_safe(item)
                    if isinstance(ctrl, dict) and ctrl.get("schema") == "dndsr.cauchy.control.v1":
                        self._upsert_from_control(ctrl, "registry", Path(ctrl.get("prefix", "") + "-control.json") if ctrl.get("prefix") else item)
                        changed = True
            for watch_dir in self._watch_dirs:
                if not watch_dir.is_dir():
                    continue
                for ctrl_path in sorted(watch_dir.rglob("*-control.json")):
                    ctrl = _read_json_safe(ctrl_path)
                    if isinstance(ctrl, dict) and ctrl.get("schema") == "dndsr.cauchy.control.v1":
                        self._upsert_from_control(ctrl, "watch", ctrl_path)
                        changed = True
                for man_path in sorted(watch_dir.rglob("*-run-manifest.json")):
                    if self._upsert_from_manifest(man_path, "watch"):
                        changed = True
            # Refresh any known control files on disk (e.g. token rotation or active->false)
            for rec in list(self._runs.values()):
                cf = rec.get("controlFile")
                if cf and Path(cf).is_file():
                    ctrl = _read_json_safe(Path(cf))
                    if isinstance(ctrl, dict) and ctrl.get("schema") == "dndsr.cauchy.control.v1":
                        rec["activeHint"] = bool(ctrl.get("active", False))
                        if ctrl.get("active") and ctrl.get("token"):
                            rec["token"] = str(ctrl["token"])
                        if ctrl.get("url"):
                            rec["url"] = str(ctrl["url"])
            if changed:
                self._save_state()

    def attach(
        self,
        *,
        path: str | None = None,
        url: str | None = None,
        token: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Manually attach a local run (control file, manifest, or prefix) or remote endpoint."""
        with self._lock:
            if path:
                p = Path(path).expanduser()
                # Support direct control.json, run-manifest.json, directory, or bare prefix
                if p.is_file() and p.name.endswith("-control.json"):
                    ctrl = _read_json_safe(p)
                    if not ctrl:
                        raise ValueError(f"cannot read control file: {p}")
                    run_id = self._upsert_from_control(ctrl, "manual", p)
                elif p.is_file() and p.name.endswith("-run-manifest.json"):
                    run_id = self._upsert_from_manifest(p, "manual")
                    if not run_id:
                        raise ValueError(f"invalid Cauchy run manifest: {p}")
                elif Path(str(p) + "-control.json").is_file():
                    cp = Path(str(p) + "-control.json")
                    ctrl = _read_json_safe(cp)
                    if not ctrl:
                        raise ValueError(f"cannot read control file: {cp}")
                    run_id = self._upsert_from_control(ctrl, "manual", cp)
                elif Path(str(p) + "-run-manifest.json").is_file():
                    mp = Path(str(p) + "-run-manifest.json")
                    run_id = self._upsert_from_manifest(mp, "manual")
                    if not run_id:
                        raise ValueError(f"invalid Cauchy run manifest: {mp}")
                else:
                    raise ValueError(f"path is not a Cauchy control file, manifest, or output prefix: {p}")
                if label:
                    self._runs[run_id]["label"] = label
                self._save_state()
                return self.summarize_run(run_id)

            if url:
                clean_url = url.strip().rstrip("/")
                if not clean_url.startswith(("http://", "https://")):
                    clean_url = "http://" + clean_url
                if not token:
                    raise ValueError("bearer token is required when attaching by URL")
                status_code, payload = self._http_request(clean_url, token, "GET", "/api/v1/status")
                if status_code != 200 or not isinstance(payload, dict):
                    raise ValueError(f"failed to query solver at {clean_url} (HTTP {status_code})")
                run_id = str(payload.get("runId") or _stable_id(clean_url))
                self._runs[run_id] = {
                    "runId": run_id,
                    "label": label or Path(str(payload.get("prefix", ""))).name or clean_url,
                    "source": "remote",
                    "prefix": str(payload.get("prefix", "")),
                    "runManifest": str(payload.get("runManifest", "")),
                    "controlFile": "",
                    "workingDirectory": "",
                    "url": clean_url,
                    "token": token,
                    "pid": payload.get("pid"),
                    "startedAt": str(payload.get("startedAt", "")),
                    "activeHint": True,
                    "lastSeen": time.time(),
                }
                self._save_state()
                return self.summarize_run(run_id)

            raise ValueError("provide either 'path' or 'url' + 'token'")

    def detach(self, run_id: str) -> bool:
        with self._lock:
            removed = self._runs.pop(run_id, None) is not None
            if removed:
                self._save_state()
            return removed

    def _http_request(
        self,
        base_url: str,
        token: str,
        method: str,
        endpoint: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        url = base_url.rstrip("/") + endpoint
        if params:
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                url += "?" + urlencode(filtered)
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        raw_body = None
        if body is not None:
            raw_body = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=raw_body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self._http_timeout) as resp:
                text = resp.read().decode("utf-8")
                return resp.status, json.loads(text) if text else {}
        except HTTPError as err:
            try:
                text = err.read().decode("utf-8")
                return err.code, json.loads(text) if text else {"error": str(err)}
            except Exception:
                return err.code, {"error": str(err)}
        except (URLError, TimeoutError, OSError) as err:
            return 0, {"error": "unreachable", "message": str(err)}

    def _query_live(self, rec: dict[str, Any], method: str, endpoint: str,
                    body: dict[str, Any] | None = None,
                    params: dict[str, Any] | None = None) -> tuple[int, Any]:
        if not rec.get("url") or not rec.get("token"):
            return 0, {"error": "no_endpoint"}
        return self._http_request(rec["url"], rec["token"], method, endpoint, body=body, params=params)

    def _resolve_disk_path(self, rec: dict[str, Any], raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        wd = rec.get("workingDirectory")
        if wd:
            candidate = Path(wd) / p
            if candidate.exists():
                return candidate
        man = rec.get("runManifest")
        if man:
            candidate = Path(man).parent / p.name
            if candidate.exists():
                return candidate
        return p

    def _load_manifest(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        man_path = rec.get("runManifest")
        if man_path:
            data = _read_json_safe(self._resolve_disk_path(rec, man_path))
            if data:
                return data
        prefix = rec.get("prefix")
        if prefix:
            return _read_json_safe(self._resolve_disk_path(rec, prefix + "-run-manifest.json"))
        return None

    def _load_summary(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        prefix = rec.get("prefix")
        if not prefix:
            return None
        return _read_json_safe(self._resolve_disk_path(rec, prefix + "-summary.json"))

    def _load_resolved_config(self, rec: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any] | None:
        if manifest and manifest.get("resolvedConfig"):
            data = _read_json_safe(self._resolve_disk_path(rec, str(manifest["resolvedConfig"])))
            if data:
                return data
        prefix = rec.get("prefix")
        if prefix:
            return _read_json_safe(self._resolve_disk_path(rec, prefix + "-resolved.json"))
        return None

    def _read_csv_cached(self, base_csv: Path) -> tuple[list[str], list[dict[str, Any]]]:
        segments = _sorted_csv_segments(base_csv)
        sig = tuple((str(s), s.stat().st_mtime_ns, s.stat().st_size) for s in segments if s.is_file())
        key = str(base_csv)
        with self._lock:
            cached = self._csv_cache.get(key)
            if cached and cached[0] == sig:
                return cached[1], cached[2]
        header, rows = _read_segmented_csv(base_csv)
        with self._lock:
            self._csv_cache[key] = (sig, header, rows)
        return header, rows

    def summarize_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            rec = dict(self._runs[run_id])
        manifest = self._load_manifest(rec)
        summary = self._load_summary(rec)

        live_status = None
        online = False
        if rec.get("url") and rec.get("token"):
            code, data = self._query_live(rec, "GET", "/api/v1/status")
            if code == 200 and isinstance(data, dict):
                live_status = data
                online = True

        state = "offline"
        if online and live_status:
            state = str(live_status.get("state", "running"))
        elif summary:
            state = str(summary.get("status", "finished"))

        domains_meta = (manifest or {}).get("domains", [])
        bc_count = sum(len(d.get("boundaryIntegrations", [])) for d in domains_meta)

        step = (live_status or {}).get("step", (summary or {}).get("steps", 0))
        stage = (live_status or {}).get("stage", -1)
        iteration = (live_status or {}).get("iteration", -1)
        phys_time = (live_status or {}).get("time", (summary or {}).get("time", 0.0))
        wall = (live_status or {}).get("wallSeconds", (summary or {}).get("wallSeconds", 0.0))

        return {
            "runId": run_id,
            "label": rec.get("label") or run_id,
            "source": rec.get("source", "unknown"),
            "online": online,
            "canSteer": online,
            "state": state,
            "prefix": rec.get("prefix", ""),
            "url": rec.get("url", ""),
            "pid": rec.get("pid"),
            "startedAt": rec.get("startedAt", ""),
            "step": step,
            "stage": stage,
            "iteration": iteration,
            "time": phys_time,
            "wallSeconds": wall,
            "steady": (live_status or {}).get("steady", (summary or {}).get("steadyMode", False)),
            "targetSteps": (live_status or {}).get("targetSteps"),
            "physicalTimeTarget": (live_status or {}).get("physicalTimeTarget", (summary or {}).get("physicalTimeTarget")),
            "domainCount": len(domains_meta) if domains_meta else (live_status or {}).get("domains", 1),
            "boundaryCount": bc_count,
            "stopRequested": bool((live_status or {}).get("stopRequested", False)),
            "appliedConfigRevision": (live_status or {}).get("appliedConfigRevision", 0),
        }

    def list_runs(self, rescan: bool = True) -> list[dict[str, Any]]:
        if rescan:
            self.scan()
        with self._lock:
            ids = list(self._runs.keys())
        summaries = [self.summarize_run(rid) for rid in ids]
        summaries.sort(key=lambda r: (not r["online"], -(r.get("wallSeconds") or 0), r["label"]))
        return summaries

    def get_run_detail(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            rec = dict(self._runs[run_id])
        summary_card = self.summarize_run(run_id)
        manifest = self._load_manifest(rec)
        summary_file = self._load_summary(rec)

        live_status = None
        live_config = None
        live_boundaries = None
        if summary_card["online"]:
            c1, d1 = self._query_live(rec, "GET", "/api/v1/status")
            if c1 == 200 and isinstance(d1, dict):
                live_status = d1
            c2, d2 = self._query_live(rec, "GET", "/api/v1/config")
            if c2 == 200 and isinstance(d2, dict):
                live_config = d2
            c3, d3 = self._query_live(rec, "GET", "/api/v1/boundaries")
            if c3 == 200 and isinstance(d3, dict):
                live_boundaries = d3

        resolved_config = (live_config or {}).get("configuration") or self._load_resolved_config(rec, manifest)
        return {
            "summary": summary_card,
            "liveStatus": live_status,
            "finalSummary": summary_file,
            "manifest": manifest,
            "configuration": resolved_config,
            "mutableSchema": (live_config or {}).get("mutableSchema"),
            "updates": (live_status or live_config or {}).get("updates", []),
            "boundaries": live_boundaries or {"domains": {}, "latest": {}},
        }

    def get_history(
        self,
        run_id: str,
        domain: int = 0,
        kind: str = "all",
        max_points: int = 2000,
    ) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            rec = dict(self._runs[run_id])

        manifest = self._load_manifest(rec)
        header: list[str] = []
        rows: list[dict[str, Any]] = []
        source = "none"

        if manifest and isinstance(manifest.get("domains"), list):
            for dom in manifest["domains"]:
                if int(dom.get("index", -1)) == domain and dom.get("history"):
                    csv_path = self._resolve_disk_path(rec, str(dom["history"]))
                    header, rows = self._read_csv_cached(csv_path)
                    if rows:
                        source = "disk"
                    break

        if not rows and rec.get("url") and rec.get("token"):
            code, data = self._query_live(rec, "GET", "/api/v1/history", params={"domain": domain, "limit": 256})
            if code == 200 and isinstance(data, dict):
                recs = data.get("records", [])
                seen: dict[str, None] = {}
                for item in recs:
                    vals = item.get("values", {})
                    for k in vals:
                        seen.setdefault(k, None)
                    rows.append(vals)
                header = list(seen.keys())
                source = "live_ring"

        if kind == "nonlinear":
            rows = [r for r in rows if r.get("status") == 0]
        elif kind == "step":
            rows = [r for r in rows if r.get("status") == 2]
        elif kind == "stage":
            rows = [r for r in rows if r.get("status") == 1]

        result = _rows_to_columns(header, rows, max_points)
        result.update({"runId": run_id, "domain": domain, "kind": kind, "source": source, "header": header})
        return result

    def get_boundary_history(
        self,
        run_id: str,
        domain: int = 0,
        bc_name: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            rec = dict(self._runs[run_id])

        manifest = self._load_manifest(rec)
        available: list[dict[str, Any]] = []
        if manifest and isinstance(manifest.get("domains"), list):
            for dom in manifest["domains"]:
                if int(dom.get("index", -1)) == domain:
                    available = list(dom.get("boundaryIntegrations", []))
                    break

        live_boundaries = None
        if rec.get("url") and rec.get("token"):
            code, data = self._query_live(rec, "GET", "/api/v1/boundaries")
            if code == 200 and isinstance(data, dict):
                live_boundaries = data
                if not available:
                    available = list(data.get("domains", {}).get(str(domain), []))

        selected = None
        if available:
            if bc_name:
                for spec in available:
                    if spec.get("canonicalName") == bc_name or str(spec.get("zone")) == bc_name:
                        selected = spec
                        break
            if selected is None:
                selected = available[0]

        series: dict[str, Any] = {
            "totalRows": 0,
            "returnedRows": 0,
            "downsampled": False,
            "indices": [],
            "columns": {},
            "header": [],
        }
        if selected and selected.get("csv"):
            csv_path = self._resolve_disk_path(rec, str(selected["csv"]))
            header, rows = self._read_csv_cached(csv_path)
            series = _rows_to_columns(header, rows, max_points)
            series["header"] = header

        return {
            "runId": run_id,
            "domain": domain,
            "available": available,
            "selected": selected,
            "latest": (live_boundaries or {}).get("latest", {}).get(str(domain), {}),
            "series": series,
        }

    def send_stop(self, run_id: str, apply_at: str = "auto") -> tuple[int, Any]:
        with self._lock:
            if run_id not in self._runs:
                return 404, {"error": "not_found"}
            rec = dict(self._runs[run_id])
        code, data = self._query_live(rec, "POST", "/api/v1/stop", body={"applyAt": apply_at})
        return (502 if code == 0 else code), data

    def send_snapshot(
        self, run_id: str, checkpoint: bool = True, visualization: bool = True, apply_at: str = "step"
    ) -> tuple[int, Any]:
        with self._lock:
            if run_id not in self._runs:
                return 404, {"error": "not_found"}
            rec = dict(self._runs[run_id])
        code, data = self._query_live(
            rec,
            "POST",
            "/api/v1/snapshot",
            body={"checkpoint": checkpoint, "visualization": visualization, "applyAt": apply_at},
        )
        return (502 if code == 0 else code), data

    def send_config_patch(
        self, run_id: str, patch: dict[str, Any], apply_at: str = "step"
    ) -> tuple[int, Any]:
        with self._lock:
            if run_id not in self._runs:
                return 404, {"error": "not_found"}
            rec = dict(self._runs[run_id])
        code, data = self._query_live(
            rec,
            "PATCH",
            "/api/v1/config",
            body={"applyAt": apply_at, "patch": patch},
        )
        return (502 if code == 0 else code), data
