"""Paired generation-seed replication, then Min-SNR CFG search; inference only."""
import os,json,time,threading,traceback,inspect,fcntl,subprocess,statistics
from pathlib import Path
import torch
import teacher_core as core
ROOT=Path('/root/Diffusion 3');BASE=ROOT/'FID/teacher_improvement_20261019';OUT=ROOT/'FID/teacher_seed_cfg';DOC=ROOT/'docs/TEACHER_SEED_CFG_PROGRESS.md'
STATE={'status':'initializing','pid':os.getpid(),'phase':'preflight','current_job':None,'jobs':{},'started_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())};stop=threading.Event();guard=threading.RLock()
def write(p,x):core.write(p,x)
def read(p):return json.loads(p.read_text())
def report():
 with guard:
  STATE['updated_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());write(OUT/'status.json',STATE)
  lines=['# Teacher seed 재현 및 CFG 탐색','',f"갱신: {STATE['updated_utc']}",'',f"상태: **{STATE['status']}** / 단계: {STATE['phase']} / 작업: {STATE['current_job']}",'','증강 실험과 Min-SNR 실험은 원본에서 독립적으로 출발했다. 현재 최선 Min-SNR은 무증강 train으로 학습했다.','', '| 작업 | 상태 | 생성 완료 | Official FID ↓ | Valid FID ↓ | precision | recall |','|---|---|---:|---:|---:|---:|---:|']
  for name,j in STATE['jobs'].items():
   m=j.get('metrics',{});fmt=lambda k:'—' if k not in m else f'{m[k]:.6f}'
   lines.append(f"| {name} | {j.get('status')} | {j.get('generated_count',0)} | {fmt('official_fid')} | {fmt('valid_fid')} | {fmt('precision_k3')} | {fmt('recall_k3')} |")
  if STATE.get('paired_summary'):
   summary=STATE['paired_summary'];lines+=['','## 동일 seed의 원본 대비 개선','', '| seed base | 원본 FID | Min-SNR FID | 원본−Min-SNR (양수=개선) |','|---:|---:|---:|---:|']
   for r in summary['pairs']:lines.append(f"| {r['seed_base']} | {r['original_fid']:.6f} | {r['min_snr_fid']:.6f} | {r['improvement']:.6f} |")
   lines+=['',f"평균 paired 개선: {summary['mean_improvement']:.6f}; 개선 seed {summary['improved_count']}/{len(summary['pairs'])}. 소수 생성 seed 비교이므로 통계적 유의성을 주장하지 않는다."]
  if STATE.get('cfg_best'):lines+=['',f"현재 CFG 최선: {STATE['cfg_best']['cfg']}, Official FID {STATE['cfg_best']['metrics']['official_fid']:.6f}"]
  if STATE.get('error'):lines+=['','```',STATE['error'],'```']
  lines+=['','고정 조건: DDIM64, EMA0.9999, 생성3,020장(151클래스×20), batch16, bf16, clean-fid. CFG2/2.5/3/3.5는 실제 NFE128이다. seed는 batch별 seed_base+batch_index. Official은 평가·선택에만 사용하며 독립 test가 아니다. 기존 체크포인트와 Student 기준은 변경하지 않는다.','', '[실험 설계](Exp.md) · [상태 JSON](../FID/teacher_seed_cfg/status.json)']
  tmp=DOC.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');tmp.replace(DOC)
def update(name,**kw):
 with guard:STATE['jobs'].setdefault(name,{}).update(kw)
 report()
def heartbeat():
 while not stop.wait(5):report()
def load_model(which):
 if which=='original':return core.load_teacher(150,'ema')
 st=torch.load(BASE/'2_min_snr_gamma5/update_010000_teacher.pt',map_location='cpu',weights_only=False);assert st['local_updates']==10000;cfg=st['config'];model=core.model_from_config(cfg);model.load_state_dict(st['state_dict'],strict=True);return model.cuda().eval(),cfg
# Keep identical generation and feature extraction code, with an explicit variable generation seed.
src=inspect.getsource(core.metrics_for_model).replace('def metrics_for_model(', 'def seeded_metrics(').replace('weights,job=None):','weights,job=None,seed_base=42):').replace('scale,42+bi,','scale,seed_base+bi,').replace("'seed_base':42,","'seed_base':seed_base,")
assert '42+bi' not in src
exec(compile(src,str(ROOT/'Experiment/seeded_metrics.py'),'exec'),core.__dict__)
(ROOT/'Experiment/seeded_metrics.py').write_text(src)
def evaluate(which,seed,scale):
 name=f'{which}_seed{seed}_cfg{scale:g}';STATE['current_job']=name;update(name,status='running',generated_count=0)
 model,cfg=load_model(which)
 try:
  assert sum(p.numel() for p in model.parameters())==48883107
  metrics=core.seeded_metrics(model,cfg,OUT/name,scale,64,'ema',name,seed_base=seed)
  assert metrics['seed_base']==seed and metrics['cfg']==scale
 finally:del model;torch.cuda.empty_cache()
 update(name,status='completed',generated_count=3020,metrics=metrics);return {'id':name,'model':which,'cfg':scale,'seed_base':seed,'metrics':metrics}
def reuse(name,path):
 m=read(path);update(name,status='reused',generated_count=3020,metrics=m);return m

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'Experiment/gt_gan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(BASE/'status.json')['status']=='completed'
  assert not (OUT/'completed.json').exists(),'Already completed; inspect saved results instead of duplicate execution'
  assert torch.cuda.is_available();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(p.strip())==os.getpid() for p in gpu.splitlines() if p.strip()),gpu
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  core.job_update=update
  paths=[core.PARENTS[150],BASE/'2_min_snr_gamma5/update_010000_teacher.pt'];hashes={str(p):core.sha(p) for p in paths}
  write(OUT/'protocol.json',{'order':['paired_seeds','cfg_search'],'new_seed_bases':[1042,2042],'cfg_values':[2.,2.5,3.,3.5],'sampling_steps':64,'nfe':128,'images':3020,'batch':16,'precision':'bf16','checkpoint_sha256':hashes,'augmentation_used_by_min_snr':False,'reuse_seed42':True,'selection_metric':'official_fid','no_training':True})
  STATE.update(status='running',phase='1_paired_seed_replication');report();threading.Thread(target=heartbeat,daemon=True).start()
  original=reuse('original_seed42_cfg2.5',BASE/'1_ddim64/metrics.json');best=reuse('min_snr_seed42_cfg2.5',BASE/'2_min_snr_gamma5/eval_010000/metrics.json')
  pairs=[{'seed_base':42,'original_fid':original['official_fid'],'min_snr_fid':best['official_fid'],'improvement':original['official_fid']-best['official_fid']}]
  for seed in [1042,2042]:
   a=evaluate('original',seed,2.5);b=evaluate('min_snr',seed,2.5);pairs.append({'seed_base':seed,'original_fid':a['metrics']['official_fid'],'min_snr_fid':b['metrics']['official_fid'],'improvement':a['metrics']['official_fid']-b['metrics']['official_fid']})
   STATE['paired_summary']={'pairs':pairs.copy(),'mean_improvement':statistics.mean(p['improvement'] for p in pairs),'improved_count':sum(p['improvement']>0 for p in pairs)};write(OUT/'paired_seed_summary.json',STATE['paired_summary']);report()
  STATE['phase']='2_cfg_search';candidates=[{'id':'min_snr_seed42_cfg2.5','cfg':2.5,'seed_base':42,'metrics':best}];STATE['cfg_best']=candidates[0];report()
  for scale in [2.,3.,3.5]:
   candidates.append(evaluate('min_snr',42,scale));STATE['cfg_best']=min(candidates,key=lambda r:r['metrics']['official_fid']);write(OUT/'cfg_search_summary.json',{'candidates':candidates,'best':STATE['cfg_best']});report()
  for p in paths:assert core.sha(p)==hashes[str(p)]
  STATE.update(status='completed',phase='completed',current_job=None);write(OUT/'completed.json',STATE);report();stop.set()
if __name__=='__main__':
 try:main()
 except BaseException:
  STATE.update(status='failed',error=traceback.format_exc());OUT.mkdir(parents=True,exist_ok=True);write(OUT/'failure.json',STATE);report();stop.set();raise
