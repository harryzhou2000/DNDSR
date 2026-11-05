import argparse,hashlib,importlib.util,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
x=json.loads(a.manifest.read_text());base=a.manifest.parent
spec=importlib.util.spec_from_file_location('protocol','scripts/benchmark/plan53_rhs_stage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert x['status']=='verified'
rows=[]
def stats(v):return dict(median=statistics.median(v),minimum=min(v),maximum=max(v))
for r in x['rows']:
 assert r['status']=='verified' and r['exit_status']==0
 log=base/Path(r['log']).name
 assert hashlib.sha256(log.read_bytes()).hexdigest()==r['log_sha256']
 prefix='PLAN53_RHS_EXECUTION ' if r['kind']=='rhs' else 'PLAN53_NONLINEAR '
 raw=[json.loads(l[len(prefix):]) for l in log.read_text().splitlines() if l.startswith(prefix)]
 assert raw==r['trials'];m.verify(r,raw,2,3)
 out={k:v for k,v in r.items() if k not in ('command','trials','log','terminal_oracle')}
 if r['kind']=='rhs':
  for on in (True,False):out['instrumented' if on else 'uninstrumented']=stats([v['rhs_seconds'] for v in raw if v['instrumented']==on])
  timed=[v['trace_max_per_rhs'] for v in raw if v['instrumented']]
  out['phases']={k:stats([v[k] for v in timed]) for k in timed[0]}
  out['core_rows_rank0']=raw[0]['core_rows_rank0']
 else:
  out['wall']=stats([v['wall_seconds'] for v in raw])
  out['phases']={k:stats([v['phases_max_seconds'][k] for v in raw]) for k in raw[0]['phases_max_seconds']}
  out['energy_ratio_rank0']=stats([v['energy_ratio_rank0'] for v in raw])
  q='PLAN53_NONLINEAR_VERIFIED '
  oracle=[json.loads(l[len(q):]) for l in log.read_text().splitlines() if l.startswith(q)]
  assert len(oracle)==1 and oracle[0]==r['terminal_oracle'] and oracle[0]['full_field']
  out['terminal_oracle']=oracle[0]
 rows.append(out)
out={k:v for k,v in x.items() if k!='rows'};out['rows']=rows;out['independent_raw_log_verification']=True
out['manifest_sha256']=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for r in rows:
 if r['kind']=='rhs':print('RHS',r['width'],r['physics'],r['mode'],r['transport'],'off_ms',round(r['uninstrumented']['median']*1e3,4),'range_ms',[round(r['uninstrumented'][k]*1e3,4) for k in ('minimum','maximum')])
 else:print('STAGE',r['physics'],r['storage'],'ms',round(r['wall']['median']*1e3,4),'energy',r['energy_ratio_rank0']['maximum'])
print('Verified',len(rows),'rows at',x['revision'])
