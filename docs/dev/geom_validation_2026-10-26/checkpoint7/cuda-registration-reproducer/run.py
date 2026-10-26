import pathlib,subprocess,os
root=pathlib.Path('/tmp/geom-cub-registration-repro')
nvcc='/usr/local/cuda-13.2/bin/nvcc'
flags=['-std=c++17','-O3','--generate-code=arch=compute_80,code=[compute_80,sm_80]','--cudart=shared']
with (root/'commands.log').open('w') as log:
    for name, extra in [('rdc',['-rdc=true']),('whole',[])]:
        command=[nvcc,*flags,*extra,str(root/'a.cu'),str(root/'b.cu'),'-o',str(root/name)]
        log.write(repr(command)+'\n');log.flush()
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT);log.write(f'compile_exit={p.returncode}\n');log.flush()
        if p.returncode: continue
        command=['/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpirun','-np','1','/usr/local/cuda-13.2/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','91',str(root/name)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['OMPI_MCA_accelerator']='null'
        log.write(repr(command)+' env CUDA_VISIBLE_DEVICES=1 OMPI_MCA_accelerator=null\n');log.flush()
        with (root/(name+'-memcheck.log')).open('w') as output:
            p=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=60)
        log.write(f'memcheck_exit={p.returncode}\n');log.flush()
