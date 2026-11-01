#!/usr/bin/env python3
"""Bounded fixed-size species layout/mapping/padding comparison."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', required=True)
p.add_argument('--output', required=True)
p.add_argument('--rows', type=int, default=262144)
p.add_argument('--species', default='16,32')
p.add_argument('--layouts', default='0,1,8,32')
p.add_argument('--split', default='0,1')
p.add_argument('--variants', default='0,1,2,8,32,128')
p.add_argument('--padding', default='0')
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--launcher', default='taskset -c 20')
p.add_argument('--device', type=int, default=1)
p.add_argument('--backend', choices=['cpu', 'cuda'], default='cuda')
a = p.parse_args()
ints = lambda s: [int(x) for x in s.split(',')]
cases = [(ns, layout, split, variant, pad)
         for ns, layout, split, variant in itertools.product(ints(a.species), ints(a.layouts), ints(a.split), ints(a.variants))
         for pad in (ints(a.padding) if layout == 1 else [0])]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
exe = Path(a.exe).resolve()
record = {'args': vars(a), 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
          'start': time.time(), 'cases': []}
print(f'output={out.resolve()}; cases={len(cases)*a.repeats}', flush=True)
with (out/'results.csv').open('w') as csv:
    for rep in range(a.repeats):
        order = cases[rep % len(cases):]+cases[:rep % len(cases)]
        for ns, layout, split, variant, pad in order:
            name = f'r{rep}-s{ns}-l{layout}-d{split}-v{variant}-p{pad}'
            cmd = shlex.split(a.launcher)+[str(exe), str(a.rows), str(ns), str(layout), str(split), str(variant), '1', str(a.device), str(pad)]
            print(name, flush=True)
            start = time.time()
            with (out/(name+'.stderr')).open('w') as stderr:
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True, timeout=240, env=os.environ.copy())
            lines = r.stdout.splitlines()
            prefix = 'species-cpu,' if a.backend == 'cpu' else 'species,'
            if r.returncode or len(lines) != 2 or any(not line.startswith(prefix) for line in lines):
                raise RuntimeError(f'{name}: exit={r.returncode}, unexpected output; inspect log')
            csv.write(r.stdout)
            csv.flush()
            record['cases'].append({'case': name, 'command': cmd, 'elapsed': time.time()-start, 'status': 'verified'})
            (out/'manifest.json').write_text(json.dumps(record, indent=2))
record['end'] = time.time()
record['status'] = 'verified'
(out/'manifest.json').write_text(json.dumps(record, indent=2))
