import pathlib,subprocess,os
root=pathlib.Path('/tmp/geom-cub-registration-repro');nvcc='/usr/local/cuda-13.2/bin/nvcc'
flags=['-std=c++17','-O3','--generate-code=arch=compute_80,code=[compute_80,sm_80]','--cudart=shared']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['OMPI_MCA_accelerator']='null'
with (root/'probes.log').open('w') as log:
    # This option is documented whole-program-only: test explicitly rather than
    # recommend it as an unverified RDC workaround.
    configurations=[('rdc-stub',['-rdc=true','-static-global-template-stub=true'],[root/'a.cu',root/'b.cu']),
                    ('rdc-one',['-rdc=true'],[root/'one.cu'])]
    (root/'one.cu').write_text((root/'a.cu').read_text()+'\n'+(root/'b.cu').read_text())
    for name,extra,sources in configurations:
        command=[nvcc,*flags,*extra,*map(str,sources),'-o',str(root/name)]
        log.write(repr(command)+'\n');log.flush();p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT);log.write(f'compile_exit={p.returncode}\n');log.flush()
        if p.returncode:continue
        cmd=['/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpirun','-np','1','/usr/local/cuda-13.2/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','91',str(root/name)]
        log.write(repr(cmd)+'\n');log.flush()
        with (root/(name+'-memcheck.log')).open('w') as out:p=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=60)
        log.write(f'memcheck_exit={p.returncode}\n');log.flush()
    # Driver's own extended logging, without sanitizer injection.
    cmd=[str(root/'rdc')];env['CUDA_LOG_FILE']=str(root/'rdc-driver.log')
    log.write(repr(cmd)+' CUDA_LOG_FILE='+env['CUDA_LOG_FILE']+'\n');log.flush()
    with (root/'rdc-uninstrumented.log').open('w') as out:p=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=30)
    log.write(f'direct_exit={p.returncode}\n')
