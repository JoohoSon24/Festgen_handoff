"""Paired CFG2.5/3.0 replication, reusing completed CFG2.5 evaluations."""
import os,time,threading,traceback,fcntl,subprocess,statistics
from pathlib import Path
import teacher_seed_cfg as q
ROOT=q.ROOT;PREV=q.OUT;OUT=ROOT/'FID/teacher_cfg_seed_verification';DOC=ROOT/'docs/TEACHER_CFG_SEED_VERIFICATION.md'
q.OUT=OUT
q.STATE={'status':'initializing','pid':os.getpid(),'phase':'preflight','current_job':None,'jobs':{},'pairs':[]}
def report():
 with q.guard:
  s=q.STATE;s['updated_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());q.write(OUT/'status.json',s)
  lines=['# CFG 2.5와 3.0의 추가 seed 검증','',f"갱신: {s['updated_utc']}",'',f"상태: {s['status']} / 작업: {s['current_job']}",'','| seed | CFG2.5 Official FID | CFG3.0 Official FID | 2.5−3.0 (양수=3.0 개선) |','|---:|---:|---:|---:|']
  for p in s['pairs']:lines.append(f"| {p['seed_base']} | {p['cfg2_5']['official_fid']:.6f} | {p['cfg3']['official_fid']:.6f} | {p['improvement']:.6f} |")
  if s['pairs']:
   pairs=s['pairs'];a=statistics.mean(p['cfg2_5']['official_fid'] for p in pairs);b=statistics.mean(p['cfg3']['official_fid'] for p in pairs)
   lines+=['',f"완료된 {len(pairs)}개 seed 평균: CFG2.5={a:.6f}, CFG3.0={b:.6f}, 평균 개선={a-b:.6f}. CFG3.0 개선 {sum(p['improvement']>0 for p in pairs)}/{len(pairs)}."]
  lines+=['','## 실행 상태','']
  for name,j in s['jobs'].items():lines.append(f"- {name}: {j['status']}, 생성 {j.get('generated_count',0)}/3020")
  if s.get('error'):lines+=['','```',s['error'],'```']
  lines+=['','고정 모델: 무증강 Min-SNR gamma5 추가10K, EMA0.9999, DDIM64, 실제 NFE128. batch16/bf16/클래스당20장/총3020장. 동일 seed의 CFG만 비교한다. Official/Valid FID와 precision/recall 전체 지표는 summary.json에 저장한다. 생성 seed3개 비교이며 통계적 유의성을 주장하지 않는다. Official 반복 선택은 독립 test가 아니다.','', '[상태와 전체 지표](../FID/teacher_cfg_seed_verification/status.json)']
  tmp=DOC.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');tmp.replace(DOC)
q.report=report
def add(seed,a,b):
 assert a['seed_base']==b['seed_base']==seed and a['cfg']==2.5 and b['cfg']==3.
 q.STATE['pairs'].append({'seed_base':seed,'cfg2_5':a,'cfg3':b,'improvement':a['official_fid']-b['official_fid']});q.write(OUT/'summary.json',{'pairs':q.STATE['pairs'],'mean_improvement':statistics.mean(p['improvement'] for p in q.STATE['pairs']),'improved_count':sum(p['improvement']>0 for p in q.STATE['pairs'])});report()
def main():
 OUT.mkdir(exist_ok=True)
 with (ROOT/'Experiment/gt_gan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert q.read(PREV/'status.json')['status']=='completed'
  assert not (OUT/'completed.json').exists(),'Already completed'
  assert q.torch.cuda.is_available();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(p.strip())==os.getpid() for p in gpu.splitlines() if p.strip()),gpu
  q.torch.set_num_threads(4);q.torch.backends.cudnn.benchmark=False;q.torch.backends.cudnn.deterministic=True;q.core.job_update=q.update
  cp=q.BASE/'2_min_snr_gamma5/update_010000_teacher.pt';digest=q.core.sha(cp)
  q.write(OUT/'protocol.json',{'seeds':[42,1042,2042],'new_cfg':3.,'reuse_cfg':2.5,'reuse_seed42':True,'checkpoint':str(cp),'checkpoint_sha256':digest,'sampling_steps':64,'nfe':128,'batch':16,'precision':'bf16','no_training':True})
  q.STATE.update(status='running',phase='cfg_seed_replication');report();threading.Thread(target=q.heartbeat,daemon=True).start()
  add(42,q.read(q.BASE/'2_min_snr_gamma5/eval_010000/metrics.json'),q.read(PREV/'min_snr_seed42_cfg3/metrics.json'))
  for seed in [1042,2042]:
   a=q.reuse(f'min_snr_seed{seed}_cfg2.5',PREV/f'min_snr_seed{seed}_cfg2.5/metrics.json');b=q.evaluate('min_snr',seed,3.)['metrics'];add(seed,a,b)
  assert q.core.sha(cp)==digest
  q.STATE.update(status='completed',phase='completed',current_job=None);q.write(OUT/'completed.json',q.STATE);report();q.stop.set()
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);q.STATE.update(status='failed',error=traceback.format_exc());q.write(OUT/'failure.json',q.STATE);report();q.stop.set();raise
