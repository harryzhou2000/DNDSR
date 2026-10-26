import json, os, pathlib, subprocess, sys, time
root=pathlib.Path('/home/harry/projects/DNDSR-array-geom')
mode=sys.argv[1]
mpi='/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpiexec'
python=str(root/'venv-interop/bin/python')
files=['test/Geom/test_native_pair_views.py','test/Geom/test_mesh_edit.py','test/Geom/test_native_affine_geometry.py','test/Geom/test_native_mesh_helpers.py','test/EulerP/test_native_solver.py','test/Geom/test_basic_geom.py','test/CFV/test_cfv_dissdisp.py::test_native_affine_shear_geometry','test/Geom/test_mesh_facade.py','test/Geom/test_overset_facade.py','test/Geom/test_mesh_demos.py','test/EulerP/test_basic_eulerP.py']
env=dict(os.environ,PYTHONPATH=str(root/'python'),OMP_NUM_THREADS='2')
records=[]
for np in ([1,2] if mode=='cuda' else [1,2,4,8]):
    command=[mpi,'--oversubscribe','-np',str(np),python]
    if mode=='cuda': command += ['/tmp/geom-native-rank-gpu.py']
    command += ['-m','pytest','--with-mpi','-q','-rs',*files]
    log=pathlib.Path(f'/tmp/geom-legacy-read-python-{mode}-np{np}.log')
    start=time.monotonic()
    with log.open('w') as output:
        result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=600)
    row={'np':np,'mode':mode,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-start,'log':str(log)}
    records.append(row)
    pathlib.Path(f'/tmp/geom-legacy-read-python-{mode}-matrix.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if result.returncode:sys.exit(result.returncode)
