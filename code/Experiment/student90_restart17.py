"""Resumable 90M CFG/progressive distillation, followed by GT-GAN NFE1 then NFE4."""
import os,sys,json,time,copy,random,threading,traceback,fcntl,shutil,math,hashlib,subprocess,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
import teacher_core as shared
from teacher.train import epoch_loader,update_ema
from teacher.diffusion import Diffusion
from distillation.core import coefficients,ddim_step,guided_x0,merged_target,target_velocity,velocity_loss,velocity_x0
from cleanfid import fid,features
from train_gt_gan import Discriminator,augment
from student90_model import Student90,export
ROOT=Path('/root/Diffusion 3');CONFIG_PATH=ROOT/'Experiment/configs/exp-20261020-student90-cfg25-restart17.json';C=json.loads(CONFIG_PATH.read_text());OUT=ROOT/'FID/student90_cfg25_restart17';DOC=ROOT/'docs/STUDENT90_PROGRESS.md'
S={'status':'initializing','pid':os.getpid(),'phase':'preflight','current_stage':None,'stages':{},'started_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())};guard=threading.RLock();stop=threading.Event();dataset=None;feature_net=None;alpha=None;student_config=None
REF=np.load(ROOT/'FID/20261018_142153/reference_features.npy');VALID=np.load(ROOT/'FID/20261018_142153/valid_features.npy');RM=REF.mean(0);RC=np.cov(REF,rowvar=False);VM=VALID.mean(0);VC=np.cov(VALID,rowvar=False)
def write(p,x):shared.write(p,x)
def read(p,default=None):return shared.read(p,default)
def room():
 free=shutil.disk_usage(ROOT).free
 if free<3*1024**3:raise RuntimeError(f'Disk free {free/1024**3:.2f}GiB <3GiB; existing experiment files preserved')
def save_torch(p,x):room();shared.shared.atom_torch(p,x)
def log_event(name,**kw):
 with guard:S['stages'].setdefault(name,{}).update(kw)
 report()
def report():
 with guard:
  S['updated_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());write(OUT/'status.json',S)
  lines=['# 90M Student 증류·fine-tune 실시간 진행','',f"갱신: {S['updated_utc']}",'',f"상태: **{S['status']}** / 단계: {S['phase']} / 작업: {S['current_stage']}",'','Teacher: 무증강 Min-SNR gamma5 추가10K EMA0.9999, CFG2.5. Student89,577,673개. 증류612K update 후 NFE1→NFE4 GT-GAN 각1000 update.','', '| 단계 | 상태 | update | best Official FID ↓ | best Valid FID ↓ |','|---|---|---:|---:|---:|']
  for name,j in S['stages'].items():
   b=j.get('best_metrics',{});lines.append(f"| {name} | {j.get('status','pending')} | {j.get('update',0)} | {b.get('official_fid','—')} | {b.get('valid_fid','—')} |")
   for ev in j.get('evaluations',[]):lines.append(f"| ↳ 평가 | 완료 | {ev['update']} | {ev['official_fid']:.6f} | {ev['valid_fid']:.6f} |")
  j=S['stages'].get(S['current_stage'],{});last=j.get('last_log')
  if last:
   lines+=['','## 최근 로그','','```json',json.dumps(last,indent=2,ensure_ascii=False),'```']
   if last.get('update',0)>=20 and last.get('training_seconds',0)>0:
    per=last['training_seconds']/last['update'];remaining=max(0,j.get('budget',0)-last['update']);lines+=['',f'현재 단계 실측 {per:.3f}초/update, 남은 순수 학습 약 {remaining*per/3600:.2f}시간. 별도 평가 시간과 후속 단계 속도는 포함하지 않는다.']
  if S.get('error'):lines+=['','## 중단 원인','','```',S['error'],'```']
  lines+=['','데이터: 개발 train16,324장만 학습. Official3020/Valid755는 loss·gradient에 사용하지 않는다. Official 반복 선택은 독립 test가 아니다. 기존 모델·저장소·고정 Student 기준은 보존한다. 실시간 loss는20 update마다 기록하며 FID는 평가 완료 시 갱신한다.','', '[승인 설정](STUDENT90_SETTINGS_REVIEW.md) · [상태 JSON](../FID/student90_cfg25_restart17/status.json) · [실행 로그](../Experiment/student90_restart17.log)']
  temp=DOC.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n',encoding='utf-8');temp.replace(DOC)
def heartbeat():
 while not stop.wait(5):
  try:report()
  except Exception:traceback.print_exc()
def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':np.random.get_state()}
def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);random.setstate(s['python']);np.random.set_state(s['numpy'])
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42);np.random.seed(42)
def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
def get_dataset():
 global dataset
 if dataset is None:
  paths=Path(C['training']['train_manifest']).read_text().splitlines();assert len(paths)==16324 and hashlib.sha256('\n'.join(paths).encode()).hexdigest()==C['training']['train_manifest_sha256']
  dataset=shared.CachedImages(paths,read(shared.DATA/'category_to_id.json'))
 return dataset

def evaluate_v(backbone,times,clip,folder,retain_images=False,student_cfg=None):
 global feature_net
 if (folder/'metrics.json').exists():return read(folder/'metrics.json')
 room();folder.mkdir(parents=True,exist_ok=True);images=folder/'generated';images.mkdir(exist_ok=True);state=rng_state();was=backbone.training;backbone.eval();calls=[0]
 handle=backbone.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1));cats=torch.arange(151).repeat_interleave(20);times=times.cuda()
 try:
  with torch.inference_mode(),amp():
   for bi,pos in enumerate(range(0,3020,16)):
    y=cats[pos:pos+16].cuda();z=torch.randn((len(y),3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42+bi));x=z;before=calls[0]
    for i,tt in enumerate(times.tolist()):
     t=torch.full_like(y,tt);nt=torch.full_like(y,int(times[i+1]) if i+1<len(times) else -1);a,s=coefficients(alpha,t);an,sn=coefficients(alpha,nt);x0=velocity_x0(backbone,x,t,y,alpha)
     if clip:x0=x0.clamp(-1,1)
     x=ddim_step(x,x0,a,s,an,sn)
    assert calls[0]-before==len(times) and torch.isfinite(x).all();x=x.clamp(-1,1)
    for i,im in enumerate(x):idx=pos+i;to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(cats[idx]):04d}_{idx%20:02d}.png')
    if bi%40==0:log_event(S['current_stage'],generated_count=pos+len(y))
  if feature_net is None:feature_net=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
  fs=fid.get_files_features([str(p) for p in sorted(images.glob('*.png'))],model=feature_net,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False);np.save(folder/'features.npy',fs);mu=fs.mean(0);cov=np.cov(fs,rowvar=False)
  result={'official_fid':float(fid.frechet_distance(mu,cov,RM,RC)),'valid_fid':float(fid.frechet_distance(mu,cov,VM,VC)),**shared.shared.common.pr(fs,REF),'nfe':len(times),'generated_count':3020,'official_reference_count':3020,'valid_reference_count':755,'seed_base':42,'batch_size':16,'precision':'bf16','parameters':89577673,'clip_denoised':clip,'fixed_guidance_scale':2.5};write(folder/'metrics.json',result)
  # Keep a small grid even when complete nonselected intermediate images are discarded.
  from PIL import Image
  grid=Image.new('RGB',(8*64,2*64),'white')
  for i,p in enumerate(sorted(images.glob('*.png'))[:16]):
   with Image.open(p) as im:grid.paste(im,(i%8*64,i//8*64))
  grid.save(folder/'preview.png')
  if not retain_images:shutil.rmtree(images)
  print('EVAL',folder,result,flush=True);return result
 finally:handle.remove();backbone.train(was);restore_rng(state)

def target_for(teacher,x,t,y,guided,parent_times,index,teacher_clip):
 a,s=coefficients(alpha,t)
 with torch.no_grad(),amp():
  if guided:x0=guided_x0(teacher,x,t,y,alpha,2.5)
  else:
   mid=parent_times[2*index+1];end_index=2*index+2;end=torch.where(end_index<len(parent_times),parent_times[end_index.clamp_max(len(parent_times)-1)],-1);am,sm=coefficients(alpha,mid);an,sn=coefficients(alpha,end);first=velocity_x0(teacher,x,t,y,alpha)
   if teacher_clip:first=first.clamp(-1,1)
   middle=ddim_step(x,first,a,s,am,sm);second=velocity_x0(teacher,middle,mid,y,alpha)
   if teacher_clip:second=second.clamp(-1,1)
   endpoint=ddim_step(middle,second,am,sm,an,sn);x0=merged_target(x,endpoint,a,s,an,sn)
  return target_velocity(x,x0,a,s)

def numeric_checks():
 # Meaningful target algebra checks independent of the 90M network.
 a=torch.tensor([.05,.4,.9],dtype=torch.float64)[:,None,None,None];s=(1-a*a).sqrt();an=torch.tensor([.3,.8,1.],dtype=torch.float64)[:,None,None,None];sn=(1-an*an).sqrt();x=torch.randn(3,3,4,4,dtype=torch.float64);x0=torch.randn_like(x);eps=(x-a*x0)/s;v=a*eps-s*x0;recovered=a*x-s*v;assert torch.allclose(recovered,x0,atol=1e-12)
 end=ddim_step(x,x0,a,s,an,sn);merged=merged_target(x,end,a,s,an,sn);assert torch.allclose(ddim_step(x,merged,a,s,an,sn),end,atol=1e-12)
 write(OUT/'numeric_checks.json',{'epsilon_to_v_x0_max_error':float((recovered-x0).abs().max()),'merged_endpoint_max_error':float((ddim_step(x,merged,a,s,an,sn)-end).abs().max())})

def load_teacher(parent,guided):
 st=torch.load(parent,map_location='cpu',weights_only=False)
 if guided:
  cfg=st['config'];network=shared.model_from_config(cfg);network.load_state_dict(st['state_dict'],strict=True);clip=True;times=torch.tensor(C['nested_sampling_grids']['32'],device='cuda')
 else:
  cfg=st['student_config'];network=shared.model_from_config(cfg);network.load_state_dict(st['ema'],strict=True);clip=st['clip_denoised'];times=torch.as_tensor(st['sampling_times'],device='cuda')
 return network.cuda().eval().requires_grad_(False),times,clip

def smoke():
 result_file=OUT/'gpu_smoke.json'
 if result_file.exists():return read(result_file)['microbatch']
 ds=get_dataset();teacher,times,clip=load_teacher(C['teacher']['checkpoint'],True);last_error=None
 for micro in [64,32,16,8,4,2,1]:
  model=optimizer=None
  try:
   seed();model=shared.model_from_config(student_config).cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);assert sum(p.numel() for g in optimizer.param_groups for p in g['params'])==89577673;torch.cuda.reset_peak_memory_stats();norms=[];backbone_grad=0.
   for update in range(2):
    optimizer.zero_grad(set_to_none=True)
    for pos in range(0,64,micro):
     x0=torch.stack([ds[i][0] for i in range(pos,pos+micro)]).cuda();y=ds.labels[pos:pos+micro].cuda();idx=torch.randint(32,(len(y),),device='cuda');t=times[idx];a,s=coefficients(alpha,t);x=a*x0+s*torch.randn_like(x0);top=t==times[0];x[top]=torch.randn_like(x[top]);target=target_for(teacher,x,t,y,True,times,idx,True)
     with amp():pred=model(x,t,y);loss=velocity_loss(pred,target,a,s)
     (loss*len(y)/64).backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));assert math.isfinite(norm) and norm>0;norms.append(norm)
    if update==1:backbone_grad=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if not n.startswith('out_conv') and p.grad is not None)**.5;assert backbone_grad>0
    optimizer.step();assert all(p.grad is None for p in teacher.parameters());assert torch.cuda.max_memory_allocated()<20_000_000_000
   result={'microbatch':micro,'effective_batch':64,'optimizer_steps_tested':2,'parameter_count':89577673,'gradient_norms':norms,'backbone_gradient_norm_second_step':backbone_grad,'frozen_teacher_has_no_gradients':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
   # Check architecture-aware export strict load and exact inference calls at both required NFE.
   model.eval()
   for n in [1,4]:
    cp=OUT/f'smoke_n{n}.ckpt';export(cp,model,student_config,C['nested_sampling_grids'][str(n)],False,{'smoke_only':True});loaded=Student90.load_checkpoint(cp,'one_nfe' if n==1 else 'few_nfe',device='cuda');y=torch.tensor([0,47],device='cuda');z=torch.randn((2,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));calls=[0];h=loaded.backbone.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
    with torch.no_grad(),amp():aa=loaded.rollout(z,y)
    assert calls[0]==n;bb=loaded.sample(tuple(z.shape),category=y,seed=42);assert calls[0]==2*n and torch.allclose(aa,bb,atol=1e-6);h.remove();result[f'nfe{n}_export_parity_max_error']=float((aa-bb).abs().max());del loaded;cp.unlink()
   write(result_file,result);print('GPU_SMOKE_PASS',result,flush=True);del model,optimizer,teacher;gc.collect();torch.cuda.empty_cache();seed();return micro
  except torch.cuda.OutOfMemoryError as error:
   last_error=str(error);del model,optimizer;gc.collect();torch.cuda.empty_cache();print('Microbatch OOM, retry',micro,flush=True)
 raise RuntimeError('GPU smoke exhausted microbatch choices: '+str(last_error))

def resume_config_allowed(saved_sha):
 current_sha=shared.sha(CONFIG_PATH)
 if saved_sha==current_sha:return True
 for migration in read(OUT/'budget_migrations.json',[]):
  if saved_sha!=migration['old_sha256'] or current_sha!=migration['new_sha256']:continue
  snapshot=Path(migration['old_config_snapshot'])
  assert shared.sha(snapshot)==saved_sha
  previous=read(snapshot)
  def normalize(value):
   value=copy.deepcopy(value);value.pop('total_distillation_updates',None);value.pop('budget_revision',None)
   for stage in value['stages']:stage.pop('updates',None)
   return value
  assert normalize(previous)==normalize(C),'Budget migration changed other settings'
  num=migration['factor_numerator'];den=migration['factor_denominator']
  assert all(n['updates']*den==o['updates']*num for o,n in zip(previous['stages'],C['stages']))
  assert C['total_distillation_updates']==sum(stage['updates'] for stage in C['stages'])
  return True
 return False

def save_selected(path,ema,times,clip,stage,update,metrics):
 room();save_torch(path,{'ema':ema.state_dict(),'student_config':student_config,'sampling_times':times.cpu(),'clip_denoised':clip,'stage':stage,'update':update,'metrics':metrics,'prediction_type':'v','fixed_guidance_scale':2.5,'train_manifest_sha256':C['training']['train_manifest_sha256'],'experiment_id':C['experiment_id']})

def train_stage(spec,parent,micro):
 name=spec['name'];folder=OUT/name;folder.mkdir(exist_ok=True);result_path=folder/'result.json'
 if result_path.exists():return read(result_path)
 S['current_stage']=name;S['phase']='distillation';seed();teacher,parent_times,teacher_clip=load_teacher(parent,name=='guidance32');guided=name=='guidance32';times=torch.tensor(C['nested_sampling_grids'][str(spec['sampling_steps'])],device='cuda');model=shared.model_from_config(student_config).cuda().train()
 if not guided:model.load_state_dict(teacher.state_dict(),strict=True)
 ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);assert sum(p.numel() for g in optimizer.param_groups for p in g['params'])==89577673;ds=get_dataset();updates=epoch=cursor=seen=0;elapsed=0.;evals=[];best=None;latest=folder/'latest.pt';selected=folder/'selected.pt'
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=False);assert st['parent']==str(parent) and st['stage']==name and resume_config_allowed(st['config_sha256']);model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);optimizer.load_state_dict(st['optimizer']);updates=st['update'];epoch=st['epoch'];cursor=st['cursor'];seen=st['images_seen'];elapsed=st['training_seconds'];best=st['best'];evals=st['evaluations'];restore_rng(st['rng']);del st
 log_event(name,status='training',update=updates,budget=spec['updates'],evaluations=evals,best_metrics=(best or {}).get('metrics',{}));write(folder/'config.json',{'spec':spec,'parent':str(parent),'microbatch':micro,'effective_batch':64,'student_config':student_config,'global_config_sha256':shared.sha(CONFIG_PATH)})
 def save_resume():
  save_torch(latest,{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'update':updates,'epoch':epoch,'cursor':cursor,'images_seen':seen,'training_seconds':elapsed,'rng':rng_state(),'parent':str(parent),'stage':name,'config_sha256':shared.sha(CONFIG_PATH),'best':best,'evaluations':evals,'student_config':student_config,'sampling_times':times.cpu(),'clip_denoised':guided})
  # Previous newly-generated resume files are removed only after this durable checkpoint exists.
  previous=folder.parent/'active_resume.json';old=read(previous,{})
  if old.get('path') and old['path']!=str(latest):
   old_path=Path(old['path'])
   if old_path.parent.parent==OUT and old_path.name=='latest.pt':old_path.unlink(missing_ok=True)
  write(previous,{'path':str(latest)})
 while updates<spec['updates']:
  loader=epoch_loader(ds,64,0,42,epoch)
  for batch_index,(images,labels) in enumerate(loader):
   if batch_index<cursor:continue
   start=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);total_loss=0.;batch=len(labels)
   for pos in range(0,batch,micro):
    x0=images[pos:pos+micro].cuda();y=labels[pos:pos+micro].cuda();idx=torch.randint(len(times),(len(y),),device='cuda');t=times[idx];a,s=coefficients(alpha,t);x=a*x0+s*torch.randn_like(x0);top=t==times[0];x[top]=torch.randn_like(x[top]);target=target_for(teacher,x,t,y,guided,parent_times,idx,teacher_clip)
    with amp():pred=model(x,t,y);loss=velocity_loss(pred,target,a,s)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation loss')
    (loss*len(y)/batch).backward();total_loss+=float(loss.detach())*len(y)/batch
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));optimizer.step();update_ema(ema,model,.999);torch.cuda.synchronize();elapsed+=time.monotonic()-start;updates+=1;cursor=batch_index+1;seen+=batch;peak=torch.cuda.max_memory_allocated();assert peak<20_000_000_000
   if updates==1 or updates%20==0:
    rec={'update':updates,'loss':total_loss,'gradient_norm':norm,'lr':2e-4,'training_seconds':elapsed,'images_seen':seen,'actual_batch_size':batch,'microbatch':micro,'activation_checkpoint_resolutions':student_config['activation_checkpoint_resolutions'],'peak_vram_bytes':peak}
    with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    log_event(name,update=updates,last_log=rec);print(name,rec,flush=True)
   due=updates%spec['evaluation_interval']==0 or updates==spec['updates']
   if due:
    log_event(name,status='evaluating');metrics=evaluate_v(ema,times,guided,folder/f'eval_{updates:06d}');evals.append({'update':updates,**metrics})
    if best is None or metrics['official_fid']<best['metrics']['official_fid']:best={'update':updates,'metrics':metrics};save_selected(selected,ema,times,guided,name,updates,metrics)
    log_event(name,status='training',evaluations=evals,best_metrics=best['metrics'])
   if updates%500==0 or due:save_resume()
   if updates>=spec['updates']:break
  else:epoch+=1;cursor=0
 result={'stage':name,'selected_checkpoint':str(selected),'best':best,'evaluations':evals,'training_updates':updates};write(result_path,result);log_event(name,status='completed',best_metrics=best['metrics']);del model,ema,teacher,optimizer;gc.collect();torch.cuda.empty_cache();return result

def export_and_verify(result):
 name=result['stage'];n=int(name[1:]);folder=OUT/name;cp=folder/f'{name}_distilled.ckpt'
 if cp.exists():return str(cp)
 st=torch.load(result['selected_checkpoint'],map_location='cpu',weights_only=False);backbone=shared.model_from_config(st['student_config']);backbone.load_state_dict(st['ema'],strict=True);room();export(cp,backbone,st['student_config'],st['sampling_times'],False,{'source_checkpoint':result['selected_checkpoint'],'selected_update':st['update'],'source_checkpoint_sha256':shared.sha(Path(result['selected_checkpoint']))});del backbone,st
 model=Student90.load_checkpoint(cp,'one_nfe' if n==1 else 'few_nfe',device='cuda');calls=[0];h=model.backbone.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1));y=torch.tensor([0,47],device='cuda');x=model.sample((2,3,64,64),category=y,seed=42);assert calls[0]==n and x.shape==(2,3,64,64) and torch.isfinite(x).all();h.remove();write(folder/'export_audit.json',{'checkpoint':str(cp),'parameters':model.count_parameters(),'actual_forward_calls':calls[0],'nfe':n,'sha256':shared.sha(cp),'loader':'student90_model.Student90.load_checkpoint; derived protected Model interface'});del model;gc.collect();torch.cuda.empty_cache();return str(cp)

def gan_finetune(name,source):
 folder=OUT/('gan_'+name);folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():return read(folder/'result.json')
 key='gan_'+name;S['current_stage']=key;S['phase']='gt_gan_fine_tuning';seed();mode='one_nfe' if name=='n1' else 'few_nfe';g=Student90.load_checkpoint(source,mode,'cuda');anchor=copy.deepcopy(g).requires_grad_(False).eval();ema=copy.deepcopy(g).requires_grad_(False).eval();d=Discriminator().cuda();gp=list(g.parameters());dp=list(d.parameters());assert sum(p.numel() for p in gp+dp)==91324298
 og=torch.optim.AdamW(gp,lr=1e-6,betas=(0.,.99),weight_decay=0);od=torch.optim.AdamW(dp,lr=1e-4,betas=(0.,.99),weight_decay=0);assert sum(p.numel() for group in og.param_groups+od.param_groups for p in group['params'])==91324298
 ds=get_dataset();groups={c:torch.where(ds.labels==c)[0] for c in range(151)};steps=warm=0;elapsed=0.;evals=[];best=None;latest=folder/'latest.pt';selected=folder/'selected.ckpt'
 def real_batch(y):
  indices=[int(groups[int(c)][torch.randint(len(groups[int(c)]),(1,)).item()]) for c in y.cpu()];return torch.stack([ds[i][0] for i in indices]).cuda()
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=False);assert st['source']==source and resume_config_allowed(st['config_sha256']);g.load_state_dict(st['g']);ema.load_state_dict(st['ema']);d.load_state_dict(st['d']);og.load_state_dict(st['og']);od.load_state_dict(st['od']);steps=st['update'];warm=st['warm'];elapsed=st['training_seconds'];best=st['best'];evals=st['evaluations'];restore_rng(st['rng']);del st
 log_event(key,status='preflight',update=steps,budget=1000,evaluations=evals,best_metrics=(best or {}).get('metrics',{}))
 # Differentiable rollout and saved-model sampler agree for both NFE1/4.
 if not (folder/'preflight.json').exists():
  state=rng_state();g.eval();y=torch.tensor([0,47],device='cuda');z=torch.randn((2,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
  with torch.no_grad(),amp():a=g.rollout(z,y)
  b=g.sample(tuple(z.shape),category=y,seed=42);assert torch.allclose(a,b,atol=1e-6)
  g.train()
  with amp():probe=-d(g.rollout(z,y,True),y).mean()
  probe.backward();norm=float(torch.nn.utils.clip_grad_norm_(gp,1,error_if_nonfinite=True));assert norm>0;og.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True);write(folder/'preflight.json',{'sampler_difference':float((a-b).abs().max()),'gradient_norm':norm,'joint_parameters':91324298});restore_rng(state)
 if best is None:
  metrics=evaluate_v(ema.backbone,ema.sampling_times,False,folder/'eval_000000',True);best={'update':0,'metrics':metrics,'checkpoint':source};evals.append({'update':0,**metrics})
 def save_resume():
  save_torch(latest,{'g':g.state_dict(),'ema':ema.state_dict(),'d':d.state_dict(),'og':og.state_dict(),'od':od.state_dict(),'update':steps,'warm':warm,'training_seconds':elapsed,'best':best,'evaluations':evals,'rng':rng_state(),'source':source,'config_sha256':shared.sha(CONFIG_PATH)})
  previous=OUT/'active_resume.json';old=read(previous,{})
  if old.get('path') and old['path']!=str(latest):
   p=Path(old['path'])
   if p.parent.parent==OUT and p.name=='latest.pt':p.unlink(missing_ok=True)
  write(previous,{'path':str(latest)})
 def d_step(step,is_warm=False):
  d.train().requires_grad_(True);od.zero_grad(set_to_none=True);total=0.
  for _ in range(8):
   y=torch.randint(151,(4,),device='cuda');z=torch.randn((4,3,64,64),device='cuda');real=real_batch(y)
   with torch.no_grad(),amp():fake=g.rollout(z,y)
   with amp():loss=F.relu(1-d(augment(real),y)).mean()+F.relu(1+d(augment(fake),y)).mean()
   (loss/8).backward();total+=float(loss.detach())/8
  if not is_warm and step%16==0:
   y=torch.randint(151,(4,),device='cuda');real=real_batch(y).requires_grad_(True);prediction=d(real,y);grad=torch.autograd.grad(prediction.sum(),real,create_graph=True)[0];reg=grad.square().flatten(1).sum(1).mean()*.5*16;reg.backward()
  torch.nn.utils.clip_grad_norm_(dp,10,error_if_nonfinite=True);od.step();return total
 while warm<100:
  log_event(key,status='discriminator_warmup');d_step(warm,True);warm+=1
  if warm%25==0:save_resume();print(key,'warmup',warm,flush=True)
 torch.cuda.reset_peak_memory_stats();log_event(key,status='training')
 while steps<1000:
  start=time.monotonic();dl=d_step(steps+1);d.eval().requires_grad_(False);g.train();og.zero_grad(set_to_none=True);advs=anchors=0.
  for _ in range(8):
   y=torch.randint(151,(4,),device='cuda');z=torch.randn((4,3,64,64),device='cuda')
   with torch.no_grad(),amp():target=anchor.rollout(z,y)
   with amp():
    fake=g.rollout(z,y,True);adv=-d(augment(fake),y).mean();anch=F.mse_loss(F.avg_pool2d(fake,4),F.avg_pool2d(target,4))+.1*F.l1_loss(fake,target);loss=.1*min(1.,(steps+1)/100)*adv+anch
   assert torch.isfinite(loss);(loss/8).backward();advs+=float(adv.detach())/8;anchors+=float(anch.detach())/8
  gn=float(torch.nn.utils.clip_grad_norm_(gp,1,error_if_nonfinite=True));og.step();update_ema(ema,g,.999);steps+=1;torch.cuda.synchronize();elapsed+=time.monotonic()-start;peak=torch.cuda.max_memory_allocated();assert peak<20_000_000_000
  if steps==1 or steps%10==0:
   rec={'update':steps,'d_hinge':dl,'g_adv':advs,'anchor':anchors,'gradient_norm':gn,'training_seconds':elapsed,'peak_vram_bytes':peak}
   with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   log_event(key,update=steps,last_log=rec,best_metrics=best['metrics']);print(key,rec,flush=True)
  if steps in [250,500,1000]:
   log_event(key,status='evaluating');metrics=evaluate_v(ema.backbone,ema.sampling_times,False,folder/f'eval_{steps:06d}');evals.append({'update':steps,**metrics})
   if metrics['official_fid']<best['metrics']['official_fid']:
    export(selected,ema.backbone,student_config,ema.sampling_times,False,{'source_checkpoint':source,'gan_update':steps,'metrics':metrics});best={'update':steps,'metrics':metrics,'checkpoint':str(selected)}
   log_event(key,status='training',evaluations=evals,best_metrics=best['metrics'])
  if steps%100==0 or steps in [250,500,1000]:save_resume()
 result={'stage':key,'source':source,'best':best,'evaluations':evals,'selection':'minimum Official FID including unchanged baseline','joint_parameters':91324298};write(folder/'result.json',result);log_event(key,status='completed',best_metrics=best['metrics']);del g,anchor,ema,d,og,od;gc.collect();torch.cuda.empty_cache();return result

def main():
 global alpha,student_config
 OUT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'Experiment/gt_gan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert C['launch_authorized'] is True,'User authorization required in approved config'
  assert not (OUT/'completed.json').exists(),'Already completed'
  assert torch.cuda.is_available();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True);assert all(int(row.split(',')[0])==os.getpid() for row in gpu.splitlines() if row.strip()),gpu
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;seed();room()
  assert shared.sha(Path(C['teacher']['checkpoint']))==C['teacher']['sha256']
  # Prevent original teacher reporter from receiving progress writes during CPU image caching.
  shared.report=report
  st=torch.load(C['teacher']['checkpoint'],map_location='cpu',weights_only=False);student_config={**st['config'],'model_channels':130,'prediction_type':'v','activation_checkpoint_resolutions':read(OUT/'execution_tuning.json',{}).get('activation_checkpoint_resolutions',[64,32,16,8])};del st
  alpha=Diffusion(1000,.0001,.02,torch.device('cuda')).alpha_bar
  mapping=read(shared.DATA/'category_to_id.json');train=Path(C['training']['train_manifest']).read_text().splitlines();hold=(shared.BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines();official=(shared.DATA/'val_split.txt').read_text().splitlines();assert not set(train)&(set(hold)|set(official))
  with torch.device('meta'):model=shared.model_from_config(student_config)
  assert sum(p.numel() for p in model.parameters())==89577673;del model
  write(OUT/'config.json',C);write(OUT/'preflight.json',{'teacher_sha256':C['teacher']['sha256'],'source_sha256':shared.sha(Path(__file__)),'model_source_sha256':shared.sha(Path(__file__).with_name('student90_model.py')),'student_parameters':89577673,'gan_joint_parameters':91324298,'train_holdout_overlap':0,'train_official_overlap':0,'gpu_processes_before_start':gpu,'git_sync':False})
  S.update(status='running',phase='numeric_checks');report();threading.Thread(target=heartbeat,daemon=True).start();numeric_checks();S['phase']='gpu_smoke';micro=smoke();write(OUT/'resolved_microbatch.json',{'microbatch':micro,'gradient_accumulation':64//micro,'effective_batch':64});S['phase']='teacher32_anchor';S['current_stage']='teacher32_anchor';log_event('teacher32_anchor',status='evaluating')
  anchor_folder=OUT/'teacher32_anchor'
  if (anchor_folder/'metrics.json').exists():anchor_metrics=read(anchor_folder/'metrics.json')
  else:
   teacher,cfg=shared.load_teacher(150,'ema');st=torch.load(C['teacher']['checkpoint'],map_location='cpu',weights_only=False);teacher.load_state_dict(st['state_dict'],strict=True);del st;anchor_metrics=shared.metrics_for_model(teacher,cfg,anchor_folder,2.5,32,'ema');del teacher;gc.collect();torch.cuda.empty_cache()
  log_event('teacher32_anchor',status='completed',best_metrics=anchor_metrics);parent=C['teacher']['checkpoint'];results=[];exports={}
  for spec in C['stages']:
   result=train_stage(spec,parent,micro);results.append(result)
   if spec['name']=='guidance32':
    delta=result['best']['metrics']['official_fid']-anchor_metrics['official_fid'];write(OUT/'guidance_quality_gate.json',{'teacher32_official_fid':anchor_metrics['official_fid'],'student32_official_fid':result['best']['metrics']['official_fid'],'delta':delta,'max_allowed_increase':15.})
    if delta>15:raise RuntimeError(f'Guidance quality gate failed: FID increased {delta:.3f} >15; progression stopped')
   parent=result['selected_checkpoint']
   if spec['name'] in ['n4','n1']:exports[spec['name']]=export_and_verify(result)
   write(OUT/'distillation_summary.json',{'results':results,'exports':exports})
  gan_results=[]
  for name in ['n1','n4']:gan_results.append(gan_finetune(name,exports[name]));write(OUT/'gan_summary.json',gan_results)
  assert shared.sha(Path(C['teacher']['checkpoint']))==C['teacher']['sha256'];S.update(status='completed',phase='completed',current_stage=None);write(OUT/'completed.json',{'status':S,'distillation':results,'gan':gan_results});report();stop.set()
if __name__=='__main__':
 try:main()
 except BaseException:
  S.update(status='failed',error=traceback.format_exc());OUT.mkdir(parents=True,exist_ok=True);write(OUT/'failure.json',S);report();stop.set();raise
