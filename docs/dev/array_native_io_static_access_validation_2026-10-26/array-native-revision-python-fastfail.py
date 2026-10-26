import json, os, pathlib, subprocess, sys, time
root=pathlib.Path('/home/harry/projects/DNDSR-array-geom')
mode=sys.argv[1]
mpi='/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpiexec'
python=str(root/'venv-interop/bin/python')
files=['test/DNDS/test_runtime_array.py','test/DNDS/test_runtime_array_bridge.py','test/DNDS/test_runtime_element_schema.py','test/DNDS/test_runtime_engine.py','test/DNDS/test_runtime_row_builder.py','test/DNDS/test_runtime_transient.py','test/DNDS/test_runtime_dlpack.py','test/demos/test_runtime_solver.py','test/Geom/test_native_pair_views.py','test/Geom/test_native_mesh_helpers.py','test/Geom/test_mesh_facade.py','test/Geom/test_native_mesh_refinement.py','test/EulerP/test_native_solver.py','test/Geom/test_basic_geom.py']
build='build-refactor-cuda' if mode=='cuda' else 'build-refactor'
env=dict(os.environ,PYTHONPATH=str(root/'python')+':'+str(root/build/'test/cpp'),OMP_NUM_THREADS='2')
records=[]
for np in ([2] if mode=='cuda' else [1,2,4,8]):
    command=[mpi,'--oversubscribe','-np',str(np),python]
    if mode=='cuda': command+=[str(root/'test/demos/one_rank_per_gpu.py'),python]
    command+=['-m','pytest','--with-mpi','-q','-rs','--maxfail=1',*files]
    log=pathlib.Path(f'/tmp/array-native-revision-python-{mode}-fastfail-np{np}.log')
    start=time.monotonic()
    with log.open('w') as output:
        result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=600)
    row={'np':np,'mode':mode,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-start,'log':str(log)}
    records.append(row)
    pathlib.Path(f'/tmp/array-native-revision-python-{mode}-fastfail-matrix.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if result.returncode:sys.exit(result.returncode)
