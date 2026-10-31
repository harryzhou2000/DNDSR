#!/usr/bin/env python3
"""Bounded simulated-kernel campaign; results are not solver validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--exe", required=True)
p.add_argument("--output", required=True)
p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--phase", choices=["compute", "comm", "pipeline"], required=True)
p.add_argument("--launcher", default="")
p.add_argument("--device", type=int)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--layouts", default="0,1,8,32,101,132")
p.add_argument("--rows", type=int, default=1048576)
p.add_argument("--components", default="7,32")
p.add_argument("--timeout", type=int, default=240)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
exe = Path(a.exe).resolve()
record = {"args": vars(a), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
          "start": time.time(), "rows": []}
(out / "manifest.json").write_text(json.dumps(record, indent=2))
layouts = [int(x) for x in a.layouts.split(",")]
print(f"records={out.resolve()}", flush=True)
with (out / "results.csv").open("w") as csv:
    for rep in range(a.repeats):
        order = layouts[rep % len(layouts):] + layouts[:rep % len(layouts)]
        for nc in [int(x) for x in a.components.split(",")]:
            for layout in order:
                cmd = shlex.split(a.launcher) + [str(exe), a.phase, a.backend, str(a.rows), str(nc), "1", str(layout)]
                if a.device is not None:
                    cmd.append(str(a.device))
                name = f"r{rep}-c{nc}-l{layout}"
                print(name + ": " + shlex.join(cmd), flush=True)
                start = time.time()
                with (out / f"{name}.stderr").open("w") as err:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, text=True,
                                            timeout=a.timeout, check=False, env=os.environ.copy())
                if result.returncode:
                    raise RuntimeError(f"{name} failed {result.returncode}; inspect stderr")
                lines = result.stdout.splitlines()
                expected = {"compute": 3, "comm": 8, "pipeline": 16}[a.phase]
                if len(lines) != expected or any(not x.startswith(a.phase + ",") for x in lines):
                    raise RuntimeError(f"{name}: unexpected result rows {len(lines)}")
                csv.write(result.stdout)
                csv.flush()
                record["rows"].append({"case": name, "command": cmd, "seconds": time.time() - start,
                                       "result_rows": len(lines), "status": "verified"})
                (out / "manifest.json").write_text(json.dumps(record, indent=2))
record["end"] = time.time()
record["status"] = "verified"
(out / "manifest.json").write_text(json.dumps(record, indent=2))
