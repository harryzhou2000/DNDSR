import subprocess, os, shlex, pathlib, hashlib, json, time
p = pathlib.Path(__file__).resolve().parent
mpi = pathlib.Path('/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin')
cuda = pathlib.Path('/usr/local/cuda-13.2/bin')
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = 'GPU-0848655f-4530-ea83-ca9b-c091c5c6f36a'
commands = p / 'commands.txt'
results = {}
def run(name, command, timeout=60):
    with commands.open('a') as log:
        log.write(f'\n{name}:\nCUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]} ' + shlex.join(map(str,command)) + '\n')
    with (p / (name + '.log')).open('w') as log:
        started=time.time()
        result = subprocess.run(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    results[name]={'returncode':result.returncode,'seconds':time.time()-started}
    with commands.open('a') as log:
        log.write(f'exit={result.returncode}\n')
    print(name,results[name],flush=True)
run('provenance', ['sh', '-c', f'{cuda}/nvcc --version; {cuda}/compute-sanitizer --version; {mpi}/mpirun --version; nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.free --format=csv,noheader'])
compile_flags=subprocess.check_output([str(mpi/'mpicxx'),'--showme:compile'],text=True).strip()
link_flags=subprocess.check_output([str(mpi/'mpicxx'),'--showme:link'],text=True).strip()
link=[]
for option in shlex.split(link_flags):
    if option.startswith('-Wl,'):
        link+=['-Xlinker',option[4:]]
    else:
        link.append(option)
run('build',[cuda/'nvcc','-std=c++17','-O2','-lineinfo','-arch=sm_80','--cudart=shared',*shlex.split(compile_flags),p/'mpi_cuda.cu',*link,'-o',p/'mpi_cuda'])
if results['build']['returncode']:
    raise SystemExit('build failed')
run('libraries',['ldd',p/'mpi_cuda'])
launch=[mpi/'mpirun','-np','1']
sanitizer=[cuda/'compute-sanitizer','--tool','memcheck','--error-exitcode','91']
run('mpi-before-cuda-memcheck',launch+sanitizer+[p/'mpi_cuda'])
run('mpi-before-cuda-plain',launch+[p/'mpi_cuda'])
run('cuda-before-mpi-memcheck',launch+sanitizer+[p/'mpi_cuda','before'])
run('cuda-before-mpi-plain',launch+[p/'mpi_cuda','before'])
(p/'results.json').write_text(json.dumps(results,indent=2)+'\n')
files=[p/'mpi_cuda.cu',p/'run.py',p/'mpi_cuda',cuda/'nvcc',cuda/'compute-sanitizer',mpi/'mpirun',mpi.parent/'lib/openmpi/mca_accelerator_cuda.so',mpi.parent.parent/'opal/mca/accelerator/cuda/accelerator_cuda_component.c']
(p/'sha256.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
