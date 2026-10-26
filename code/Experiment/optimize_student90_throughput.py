"""Pause only after durable resume; benchmark microbatch/checkpointing; resume unchanged training objective."""
import os,sys,time,json,signal,subprocess,copy,traceback,gc,threading,fcntl
from pathlib import Path
import torch
import student90_pipeline as p
ROOT=p.ROOT;OUT=p.OUT;BENCH=OUT/'throughput_benchmark.json';DOC=ROOT/'docs/STUDENT90_THROUGHPUT.md';S={'status':'waiting_for_durable_resume','results':[],'started_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())}
def report():
 S['updated_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());p.write(BENCH,S)
 lines=['# 90M Student 처리량 최적화','',f"상태: {S['status']} / {S['updated_utc']}",'','유효 batch64·objective·LR·모델·update 예산을 유지한다. 실행 microbatch와 activation checkpointing만 비교한다. 초기 학습의 durable checkpoint를 저장한 뒤 중단하고 해당 update에서 재개한다.','', '| microbatch | checkpoint 해상도 | 상태 | 초/update | 피크 VRAM GB |','|---:|---|---|---:|---:|']
 for r in S['results']:lines.append(f"| {r['microbatch']} | {r['checkpoint_resolutions']} | {r['status']} | {r.get('seconds_per_update','—')} | {r.get('peak_allocated_bytes',0)/1e9:.3f} |")
 if S.get('selected'):lines+=['','선택:','```json',json.dumps(S['selected'],ensure_ascii=False,indent=2),'```']
 if S.get('error'):lines+=['','```',S['error'],'```']
 DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
def wait_and_pause():
 while True:
  state=p.read(OUT/'status.json',{});resume=p.read(OUT/'active_resume.json',{}).get('path');path=Path(resume) if resume else OUT/'guidance32/latest.pt'
  if path.exists():break
  if state.get('status') in ['failed','completed']:raise RuntimeError('Trainer stopped before durable resume: '+str(state.get('status')))
  S['observed_update']=state.get('stages',{}).get('guidance32',{}).get('update');report();time.sleep(5)
 if state.get('status')=='paused_for_throughput_benchmark':
  S.update(status='benchmarking',resume_checkpoint=str(path));report();return path
 pid=state['pid'];args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True).strip();assert '/root/Diffusion 3/Experiment/student90_pipeline.py' in args
 os.kill(pid,signal.SIGTERM)
 for _ in range(60):
  try:
   stat=Path(f'/proc/{pid}/stat').read_text().split()[2]
   if stat=='Z':break
  except FileNotFoundError:break
  time.sleep(.5)
 else:raise RuntimeError('Trainer process did not stop')
 state.update(status='paused_for_throughput_benchmark');p.write(OUT/'status.json',state);S.update(status='benchmarking',resume_checkpoint=str(path));report();return path

def candidate(st,teacher,images,labels,micro,resolutions,profile='guidance'):
 model=ema=optimizer=None
 try:
  p.seed();cfg={**st['student_config'],'activation_checkpoint_resolutions':resolutions};model=p.shared.model_from_config(cfg).cuda().train();model.load_state_dict(st['model'],strict=True);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);optimizer.load_state_dict(copy.deepcopy(st['optimizer']));times=torch.tensor(p.C['nested_sampling_grids']['32' if profile=='guidance' else '16'],device='cuda');parent_times=torch.tensor(p.C['nested_sampling_grids']['32'],device='cuda');losses=[];durations=[]
  for iteration in range(4):
   torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss_value=0.
   for pos in range(0,64,micro):
    x0=images[pos:pos+micro].cuda();y=labels[pos:pos+micro].cuda();idx=torch.randint(len(times),(len(y),),device='cuda');t=times[idx];a,s=p.coefficients(p.alpha,t);x=a*x0+s*torch.randn_like(x0);top=t==times[0];x[top]=torch.randn_like(x[top]);target=p.target_for(teacher,x,t,y,profile=='guidance',parent_times,idx,True)
    with p.amp():prediction=model(x,t,y);loss=p.velocity_loss(prediction,target,a,s)
    assert torch.isfinite(loss);(loss*len(y)/64).backward();loss_value+=float(loss.detach())*len(y)/64
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));assert norm>0;optimizer.step();p.update_ema(ema,model,.999);torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if iteration==0:torch.cuda.reset_peak_memory_stats()
   else:durations.append(elapsed);losses.append(loss_value)
  peak=torch.cuda.max_memory_allocated();assert peak<20_000_000_000;assert all(param.grad is None for param in teacher.parameters())
  return {'microbatch':micro,'checkpoint_resolutions':resolutions,'profile':profile,'status':'pass','seconds_per_update':sum(durations)/len(durations),'measurements_seconds':durations,'peak_allocated_bytes':peak,'finite_losses':losses,'frozen_teacher_has_no_gradients':True}
 except torch.cuda.OutOfMemoryError:
  return {'microbatch':micro,'checkpoint_resolutions':resolutions,'profile':profile,'status':'oom'}
 except AssertionError as error:
  return {'microbatch':micro,'checkpoint_resolutions':resolutions,'profile':profile,'status':'failed_check','error':str(error)}
 finally:
  del model,ema,optimizer;gc.collect();torch.cuda.empty_cache()

def main():
 assert not (OUT/'execution_tuning.json').exists(),'Tuning already exists; inspect rather than duplicate benchmark'
 resume=wait_and_pause()
 with (ROOT/'Experiment/gt_gan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(x.strip())==os.getpid() for x in gpu.splitlines() if x.strip()),gpu
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;p.seed();st=torch.load(resume,map_location='cpu',weights_only=False);S['resume_update']=st['update'];S['resume_checkpoint_sha256']=p.shared.sha(resume);p.alpha=p.Diffusion(1000,.0001,.02,torch.device('cuda')).alpha_bar
  paths=Path(p.C['training']['train_manifest']).read_text().splitlines()[:64];p.shared.report=lambda:None;ds=p.shared.CachedImages(paths,p.read(p.shared.DATA/'category_to_id.json'));images=torch.stack([ds[i][0] for i in range(64)]);labels=ds.labels
  teacher,_,_=p.load_teacher(p.C['teacher']['checkpoint'],True)
  # Original setting is measured on the same resume state to establish an apples-to-apples speed ratio.
  choices=[(8,[64,32,16,8]),(16,[64,32,16,8]),(32,[64,32,16,8]),(64,[64,32,16,8]),(16,[]),(32,[]),(64,[]),(32,[64,32]),(64,[64,32])]
  for micro,resolutions in choices:
   S['current']={'microbatch':micro,'checkpoint_resolutions':resolutions};report();row=candidate(st,teacher,images,labels,micro,resolutions);S['results'].append(row);report();print('BENCH',row,flush=True)
  passed=sorted([r for r in S['results'] if r['status']=='pass'],key=lambda r:r['seconds_per_update']);assert passed
  selected=None
  for row in passed:
   # A 90M frozen teacher adds cost in later progressive stages; validate the selected execution setting there too.
   p.seed();teacher90=p.shared.model_from_config({**st['student_config'],'activation_checkpoint_resolutions':[]}).cuda().eval().requires_grad_(False);teacher90.load_state_dict(st['ema'],strict=True)
   check=candidate(st,teacher90,images,labels,row['microbatch'],row['checkpoint_resolutions'],'progressive');del teacher90;gc.collect();torch.cuda.empty_cache();row['progressive_validation']=check;report()
   if check['status']=='pass':selected=row;break
  assert selected is not None,'No configuration passed progressive validation'
  baseline=next(r for r in S['results'] if r['microbatch']==8 and r['checkpoint_resolutions']==[64,32,16,8]);speedup=baseline['seconds_per_update']/selected['seconds_per_update'];tuning={'microbatch':selected['microbatch'],'gradient_accumulation':64//selected['microbatch'],'effective_batch':64,'activation_checkpoint_resolutions':selected['checkpoint_resolutions'],'guidance_seconds_per_update':selected['seconds_per_update'],'progressive_seconds_per_update':selected['progressive_validation']['seconds_per_update'],'benchmark_speedup_over_original':speedup,'resume_checkpoint':str(resume),'resume_update':st['update'],'approved_change':'2026-10-20 user requested larger minibatch and restart; objective/LR/effective batch/update budget unchanged'}
  p.write(OUT/'execution_tuning.json',tuning);S.update(status='benchmark_complete_resuming_training',selected=tuning);report();del teacher,st;gc.collect();torch.cuda.empty_cache()
 # Hand the same process over to trainer; original lock is released before the trainer acquires it.
 handle=open(ROOT/'Experiment/student90_pipeline.log','a');os.dup2(handle.fileno(),1);os.dup2(handle.fileno(),2);os.execv(sys.executable,[sys.executable,'-X','utf8','-u',str(ROOT/'Experiment/student90_pipeline.py')])
if __name__=='__main__':
 try:main()
 except BaseException:
  S.update(status='failed',error=traceback.format_exc());report();raise
