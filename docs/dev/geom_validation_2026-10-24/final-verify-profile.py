import importlib
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path.cwd()
build = (root / sys.argv[1]).resolve()
expected_cuda = sys.argv[2] == 'cuda'

def build_id(path):
    text = subprocess.check_output(['readelf', '-n', str(path)], text=True)
    return re.search(r'Build ID: (\w+)', text).group(1)

records=[]
for directory, name in [('DNDS','dnds'),('Geom','geom'),('CFV','cfv'),('EulerP','eulerP')]:
    module = importlib.import_module(f'DNDSR.{directory}._ext.{name}_pybind11')
    installed = pathlib.Path(module.__file__).resolve()
    assert installed.parent == root/'python/DNDSR'/directory/'_ext', installed
    original = build/'src'/directory/installed.name
    shared = root/'python/DNDSR/_lib'/f'lib{name}_shared.so'
    original_shared = build/'src'/directory/shared.name
    for used, made in [(installed,original),(shared,original_shared)]:
        uid, mid = build_id(used), build_id(made)
        assert uid==mid,(used,made,uid,mid)
        records.append({'loaded_or_package_path':str(used),'build_path':str(made),'elf_build_id':uid})
from DNDSR import DNDS
assert bool(DNDS.Runtime.cuda_enabled)==expected_cuda, DNDS.Runtime.cuda_enabled
print(json.dumps({'build':str(build),'python':sys.executable,'cuda_enabled':bool(DNDS.Runtime.cuda_enabled),'verified_binaries':records},indent=2))
