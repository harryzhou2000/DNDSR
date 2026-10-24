from pathlib import Path
import concurrent.futures, datetime, hashlib, json, re, shlex, subprocess, time
root=Path('/home/harry/projects/DNDSR-array-geom')
out=Path('/tmp/geom-final-consumer-syntax')
tasks=[('NS_2EQ','EulerEvaluator_NS_2EQ.cpp'),('NS_2EQ','EulerEvaluator_EvaluateDt_NS_2EQ.cpp'),('NS_2EQ','EulerEvaluator_EvaluateRHS_NS_2EQ.cpp'),('NS_EX','EulerEvaluator_NS_EX.cpp'),('NS_EX','EulerEvaluator_EvaluateDt_NS_EX.cpp'),('NS_EX','EulerEvaluator_EvaluateRHS_NS_EX.cpp'),('NS_2EQ','EulerSolver_Init_NS_2EQ.cpp'),('NS_EX','EulerSolver_PrintData_NS_EX.cpp')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'configured_flags_source':'active build-refactor flags.make/build.make; active compile_commands.json absent','compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True).splitlines()[0],'sources':{},'checks':[]}
for p in sorted((root/'src/Euler').glob('*.h*')):
 manifest['sources'][str(p.relative_to(root))]=sha(p)
for p in sorted((root/'src/Geom/Mesh').glob('*.hpp')):
 manifest['sources'][str(p.relative_to(root))]=sha(p)
def run(item):
 model,name=item
 flags=root/f'build-refactor/src/Euler/CMakeFiles/euler_library_{model}.dir/flags.make'
 build=flags.with_name('build.make')
 source=root/'src/Euler/_explicit_instantiation'/name
 values=flags.read_text()
 args=['/usr/bin/c++']
 for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
  args+=shlex.split(re.search('^'+key+' = (.*)$',values,re.M).group(1))
 args+=['-fsyntax-only',str(source)]
 record={'source':str(source.relative_to(root)),'source_sha256':sha(source),'flags_source':str(flags),'flags_sha256':sha(flags),'build_rule_source':str(build),'build_rule_sha256':sha(build),'cwd':str(root/'build-refactor/src/Euler'),'argv':args}
 stem=source.stem
 (out/(stem+'.command.json')).write_text(json.dumps(record,indent=2)+'\n')
 start=time.monotonic()
 try:
  p=subprocess.run(args,cwd=record['cwd'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
  result=p.stdout;record['status']=p.returncode
 except subprocess.TimeoutExpired as e:
  result=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else (e.stdout or '')
  record['status']='timeout300'
 record['elapsed_seconds']=round(time.monotonic()-start,3)
 record['log']=stem+'.log'
 (out/record['log']).write_text(result+'\nstatus='+str(record['status'])+'\n')
 print(name+' status='+str(record['status'])+' elapsed='+str(record['elapsed_seconds']),flush=True)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 manifest['checks']=list(pool.map(run,tasks))
manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest['source_unchanged']=all(sha(root/p)==digest for p,digest in manifest['sources'].items()) and all(sha(root/r['source'])==r['source_sha256'] for r in manifest['checks'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'summary.txt').write_text('\n'.join(r['source']+' status='+str(r['status'])+' elapsed='+str(r['elapsed_seconds']) for r in manifest['checks'])+'\nsource_unchanged='+str(manifest['source_unchanged'])+'\n')
print('manifest='+str(out/'manifest.json')+' source_unchanged='+str(manifest['source_unchanged']),flush=True)
