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
p.add_argument("--variants", help="CUDA access: 0 direct,1 shared,2 cp.async,3 sync pipeline,8/32 subgroup,64/128/256 block,1256 one resident block/SM")
p.add_argument("--padding", help="SoA component-plane padding values at fixed logical N")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
exe = Path(a.exe).resolve()
record = {"args": vars(a), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
          "start": time.time(), "rows": []}
(out / "manifest.json").write_text(json.dumps(record, indent=2))
layouts = [int(x) for x in a.layouts.split(",")]
variants = [int(x) for x in a.variants.split(",")] if a.variants else [0]
if a.variants and (a.phase != "compute" or a.backend != "cuda" or a.device is None):
    raise ValueError("access variants require CUDA compute with an explicit device")
paddings = [int(x) for x in a.padding.split(",")] if a.padding else [0]
cases = [(layout, variant, pad) for layout in layouts for variant in variants
         for pad in (paddings if layout in (1, 101) else [0])]
print(f"records={out.resolve()}", flush=True)
with (out / "results.csv").open("w") as csv:
    for rep in range(a.repeats):
        order = cases[rep % len(cases):] + cases[:rep % len(cases)]
        for nc in [int(x) for x in a.components.split(",")]:
            for layout, variant, pad in order:
                cmd = shlex.split(a.launcher) + [str(exe), a.phase, a.backend, str(a.rows), str(nc), "1", str(layout)]
                if a.device is not None or a.padding:
                    cmd.append(str(a.device or 0))
                if a.variants or a.padding:
                    cmd.append(str(variant))
                if a.padding:
                    cmd.append(str(pad))
                name = f"r{rep}-c{nc}-l{layout}" + (f"-v{variant}" if a.variants else "") + (f"-p{pad}" if a.padding else "")
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
