"""Reproduce/select existing teachers, diagnose timesteps, matched MSE/MinSNR arms."""
import os,sys,json,time,copy,random,hashlib,traceback,fcntl,threading,subprocess,math
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image
from cleanfid import fid,features
import train_gt_gan as shared
ROOT=shared.ROOT;BASE=shared.BASE;DATA=shared.DATA
# Read existing protected dataset module only to import the unchanged trainer helpers.
sys.path.append(str(BASE))
from teacher.train import epoch_loader,drop_classes,update_ema
from teacher.diffusion import Diffusion
from teacher.model import ClassConditionalUNet
OUT=ROOT/'FID/teacher_improvement_20261019';DOC=ROOT/'docs/Progress.md'
PARENTS={k:BASE/f'experiments/teacher_budget_noaug_fresh_200k/teacher_b64_noaug_fresh_{k}k/final.pt' for k in [150,200]}
CONFIG={'id':'teacher-improve-20261019','parent_ids':['noaug-fresh150k','noaug-fresh200k'],'hypothesis':'checkpoint/CFG/EMA choice and timestep loss weighting can improve fixed-size teacher','teacher_parameters':48883107,'phase2_updates':10000,'phase2_evaluate_updates':[5000,10000],'min_snr_gamma':5.,'selection_metric':'official_fid','sampler':'ddim','clip_denoised':True,'seed_base':42,'batch_eval':16,'generated_count':3020,'precision':'bf16','training_data':'development train only','no_new_external_training_data':True,'note':'teacher NFE may exceed4; not an eligible student submission. Repeated Official selection is development use, not independent test.'}
STATE={'status':'initializing','pid':os.getpid(),'phase':'preflight','jobs':{},'current_job':None,'started_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())};lockstate=threading.RLock();stopped=threading.Event();feature_net=None;selected=None;champion=None;cache=None
ref=np.load(ROOT/'FID/20261018_142153/reference_features.npy');val=np.load(ROOT/'FID/20261018_142153/valid_features.npy');rm=ref.mean(0);rc=np.cov(ref,rowvar=False);vm=val.mean(0);vc=np.cov(val,rowvar=False)
def read(p,default=None):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
def sha(p):return shared.digest(p)
def report():
 with lockstate:
  STATE['updated_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());write(OUT/'status.json',STATE)
  lines=['# Diffusion 3 실시간 진행','',f"갱신: {STATE['updated_utc']}",'',f"상태: {STATE['status']} / 단계: {STATE['phase']} / 작업: {STATE['current_job']}",'','실행 순서: 기존 번호 1 → 5 → 2 → 3 → 4.','', '| 작업 | 상태 | update | Official FID | Valid FID |','|---|---|---:|---:|---:|']
  for key,j in STATE['jobs'].items():
   m=j.get('metrics',{});lines.append(f"| {key} | {j.get('status')} | {j.get('update','—')} | {m.get('official_fid','—')} | {m.get('valid_fid','—')} |")
   for ev in j.get('evaluations',[]):lines.append(f"| ↳ 평가 | 완료 | {ev['update']} | {ev['official_fid']:.6f} | {ev['valid_fid']:.6f} |")
  if champion:lines+=['',f"현재 최선: {champion['id']}, Official FID {champion['metrics']['official_fid']:.6f}"]
  if STATE.get('error'):lines+=['','```',STATE['error'],'```']
  lines+=['','설계: [Exp.md](Exp.md). Official 반복 선택은 독립 test가 아니다. 학습에 평가 이미지를 사용하지 않는다.']
  text='\n'.join(lines)+'\n';tmp=DOC.with_suffix('.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(DOC)
def heartbeat():
 while not stopped.wait(5):
  try:report()
  except Exception:traceback.print_exc()
def job_update(key,**kwargs):
 with lockstate:STATE['jobs'].setdefault(key,{}).update(kwargs)
 report()
def model_from_config(c):
 return ClassConditionalUNet(image_size=c['image_size'],in_channels=c['in_channels'],num_classes=c['num_classes'],model_channels=c['model_channels'],channel_mult=tuple(c['channel_mult']),num_res_blocks=c['num_res_blocks'],attention_resolutions=tuple(c['attention_resolutions']),dropout=c['dropout'],activation_checkpoint_resolutions=tuple(c.get('activation_checkpoint_resolutions',())))
def load_teacher(parent_k,weights):
 st=torch.load(PARENTS[parent_k],map_location='cpu',weights_only=False);cfg=st['config'];m=model_from_config(cfg);sd=st['ema'] if weights=='ema' else st['model'] if weights=='model' else st['ema_tracks'][weights.split(':',1)[1]];m.load_state_dict(sd,strict=True);m=m.cuda().eval();assert sum(p.numel() for p in m.parameters())==48883107;return m,cfg

def metrics_for_model(model,cfg,folder,scale,steps,weights,job=None):
 global feature_net
 folder.mkdir(parents=True,exist_ok=True)
 if (folder/'metrics.json').exists():return read(folder/'metrics.json')
 images=folder/'generated';images.mkdir(exist_ok=True);schedule=Diffusion(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'],torch.device('cuda'));cats=torch.arange(151).repeat_interleave(20);counter=[0]
 def hook(*_):counter[0]+=1
 h=model.register_forward_hook(hook);was_training=model.training;model.eval();nfe=steps*(2 if scale not in (0,1) else 1)
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for bi,pos in enumerate(range(0,3020,16)):
    y=cats[pos:pos+16].cuda();before=counter[0];x,reported=schedule.sample(model,y,64,3,steps,scale,42+bi,sampler='ddim',clip_denoised=True);assert reported==nfe==counter[0]-before and torch.isfinite(x).all()
    for i,im in enumerate(x):
     idx=pos+i;to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(cats[idx]):04d}_{idx%20:02d}.png')
    if job and bi%40==0:job_update(job,generated_count=pos+len(y))
 finally:h.remove();model.train(was_training)
 if feature_net is None:feature_net=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
 feats=fid.get_files_features([str(p) for p in sorted(images.glob('*.png'))],model=feature_net,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False);np.save(folder/'features.npy',feats);mean=feats.mean(0);cov=np.cov(feats,rowvar=False)
 m={'official_fid':float(fid.frechet_distance(mean,cov,rm,rc)),'valid_fid':float(fid.frechet_distance(mean,cov,vm,vc)),**shared.common.pr(feats,ref),'cfg':scale,'sampling_steps':steps,'nfe':nfe,'weights':weights,'generated_count':3020,'official_reference_count':3020,'valid_reference_count':755,'seed_base':42,'batch_size':16,'precision':'bf16','parameters':sum(p.numel() for p in model.parameters()),'submission_eligible':False};write(folder/'metrics.json',m);print('EVAL',folder,m,flush=True);return m

def evaluation(key,k,weights='ema',cfg_scale=2.,steps=32):
 STATE['current_job']=key;job_update(key,status='running');model,cfg=load_teacher(k,weights)
 try:m=metrics_for_model(model,cfg,OUT/key,cfg_scale,steps,weights,key)
 finally:del model;torch.cuda.empty_cache()
 result={'id':key,'parent_k':k,'parent_checkpoint':str(PARENTS[k]),'weights':weights,'cfg':cfg_scale,'sampling_steps':steps,'metrics':m};write(OUT/key/'candidate.json',result);job_update(key,status='completed',metrics=m);return result
class CachedImages(Dataset):
 def __init__(self,paths,mapping):
  self.paths=paths;self.labels=torch.tensor([mapping[Path(x).parent.name] for x in paths]);self.data=torch.empty((len(paths),3,64,64),dtype=torch.uint8);resize=Resize((64,64))
  for i,line in enumerate(paths):
   with Image.open(DATA/line) as im:a=np.array(resize(im.convert('RGB')),dtype=np.uint8,copy=True)
   self.data[i]=torch.from_numpy(a).permute(2,0,1)
   if i%2000==0:STATE['data_cache_images']=i;report()
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):return (self.data[i].float()/255-.5)/.5,self.labels[i]
def get_cache():
 global cache
 if cache is None:
  mapping=read(DATA/'category_to_id.json');lines=(BASE/'data/development/seed42_holdout5/train.txt').read_text().splitlines();cache=CachedImages(lines,mapping)
 return cache

def diagnostic(key,raw,ema,cfg,track):
 folder=OUT/key;folder.mkdir(parents=True,exist_ok=True);p=folder/'timestep_diagnostics.json'
 if p.exists():return read(p)
 mapping=read(DATA/'category_to_id.json');train=get_cache();groups={c:[] for c in range(151)}
 for i,line in enumerate(train.paths):groups[int(train.labels[i])].append(i)
 rng=np.random.default_rng(8102);indices=np.concatenate([rng.choice(groups[c],5,replace=False) for c in range(151)]);held=(BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines();holdcache=CachedImages(held,mapping)
 splits={'train_subset755':([train[int(i)][0] for i in indices],train.labels[torch.tensor(indices)]),'holdout755':([holdcache[i][0] for i in range(755)],holdcache.labels)};schedule=Diffusion(1000,cfg['beta_start'],cfg['beta_end'],torch.device('cuda'));rows=[]
 for weight,network in [('raw',raw),(track,ema)]:
  prev=network.training;network.eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for split,(images,labels) in splits.items():
    for timepoint in range(50,1000,100):
     total=0.;n=0;generator=torch.Generator(device='cuda').manual_seed(8300+timepoint)
     for pos in range(0,755,32):
      x=torch.stack(images[pos:pos+32]).cuda();y=labels[pos:pos+32].cuda();t=torch.full_like(y,timepoint);a=schedule.alpha_bar[t,None,None,None];noise=torch.randn(x.shape,device='cuda',generator=generator);noisy=a.sqrt()*x+(1-a).sqrt()*noise;pred=network(noisy,t,y);error=(pred.float()-noise).square().flatten(1).mean(1);total+=float(error.sum());n+=len(y)
     snr=float(schedule.alpha_bar[timepoint]/(1-schedule.alpha_bar[timepoint]));rows.append({'weights':weight,'split':split,'timestep':timepoint,'log_snr':math.log(snr),'snr':snr,'min_snr_gamma5_weight':min(snr,5)/snr,'epsilon_mse':total/n,'images':n})
  network.train(prev)
 result={'rows':rows,'selection':'class-stratified fixed5/class train subset; all755holdout; fixed noise per timestep','interpretation':'MSE across timesteps has different difficulty; large value alone is not proof of bottleneck. Weighted objectives compared using common unweighted diagnostic.'};write(p,result)
 with lockstate:STATE.setdefault('diagnostic_files',[]).append(str(p.relative_to(OUT)))
 report();return result

def train_arm(key,kind):
 global champion
 STATE['current_job']=key;job_update(key,status='preparing');folder=OUT/key;folder.mkdir(parents=True,exist_ok=True);ds=get_cache();st=torch.load(PARENTS[selected['parent_k']],map_location='cpu',weights_only=False);cfg=st['config'].copy();cfg['data_root']=str(DATA);cfg['split_dir']=str(DATA);manifest=BASE/'data/development/seed42_holdout5/train.txt';assert st['train_paths_sha256']==hashlib.sha256('\n'.join(ds.paths).encode()).hexdigest();assert st['class_names']==[name for name,c in sorted(read(DATA/'category_to_id.json').items(),key=lambda x:x[1])];assert cfg['batch_size']==64 and cfg['gradient_accumulation']==1
 model=model_from_config(cfg).cuda();model.load_state_dict(st['model']);ema=copy.deepcopy(model).requires_grad_(False).eval();ema.load_state_dict(st['ema']);extra={k:copy.deepcopy(model).requires_grad_(False).eval() for k in st.get('ema_tracks',{})}
 for k,v in extra.items():v.load_state_dict(st['ema_tracks'][k])
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);optimizer.load_state_dict(st['optimizer']);assert sum(p.numel() for a in optimizer.param_groups for p in a['params'])==48883107
 track=ema if selected['weights']=='ema' else extra[selected['weights'].split(':',1)[1]];schedule=Diffusion(1000,cfg['beta_start'],cfg['beta_end'],torch.device('cuda'));initial_step=st['step'];epoch=st['epoch'];cursor=st['batch_in_epoch'];seen=st.get('images_seen',initial_step*64);updates=0
 configuration={'parent_checkpoint':str(PARENTS[selected['parent_k']]),'parent_id':'teacher'+str(initial_step),'hypothesis':'change timestep weighting only','objective':kind,'gamma':5 if kind=='min_snr' else None,'updates':10000,'evaluate_updates':[5000,10000],'learning_rate':cfg['learning_rate'],'batch_size':64,'microbatch_size':64,'gradient_accumulation':1,'class_dropout':cfg['class_dropout'],'primary_ema_decay':cfg['ema_decay'],'ema_tracks':list(extra),'selected_track':selected['weights'],'sampling_steps':selected['sampling_steps'],'cfg':selected['cfg'],'train_manifest':str(manifest),'train_manifest_sha256':st['train_paths_sha256'],'augmentation':'none','trainable_parameters':48883107,'all_optimized_parameters':48883107,'raw_config':cfg};write(folder/'config.json',configuration)
 # GPU infrastructure and gradient check; no optimizer update before restoring state/RNG.
 sample=torch.stack([ds[i][0] for i in range(4)]).cuda();labels=ds.labels[:4].cuda();t=torch.tensor([50,350,650,950],device='cuda');noise=torch.randn_like(sample);a=schedule.alpha_bar[t,None,None,None]
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(a.sqrt()*sample+(1-a).sqrt()*noise,t,labels);per=(pred.float()-noise).square().flatten(1).mean(1);w=torch.minimum((a.flatten()/(1-a.flatten())),torch.full_like(t,5,dtype=torch.float32))/(a.flatten()/(1-a.flatten()));probe=(per*(w if kind=='min_snr' else 1)).mean()
 probe.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));assert math.isfinite(norm) and norm>0;optimizer.zero_grad(set_to_none=True);write(folder/'preflight.json',{'parameters':48883107,'gradient_norm':norm,'loss_finite':bool(torch.isfinite(probe)),'parent_step':initial_step,'train_manifest_verified':True,'actual_lrs':[x['lr'] for x in optimizer.param_groups]})
 torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);torch.cuda.set_rng_state_all(st['cuda_rng']);del st
 model.train();start=time.time();torch.cuda.reset_peak_memory_stats();job_update(key,status='training',update=0);evals=[];best={'id':selected['id'],'metrics':selected['metrics'],'parent_checkpoint':str(PARENTS[selected['parent_k']]),'weights':selected['weights']}
 def save_training(path):
  shared.atom_torch(path,{'model':model.state_dict(),'ema':ema.state_dict(),'ema_tracks':{k:v.state_dict() for k,v in extra.items()},'optimizer':optimizer.state_dict(),'step':initial_step+updates,'local_updates':updates,'epoch':epoch,'batch_in_epoch':cursor,'images_seen':seen,'config':cfg,'experiment':configuration,'class_names':[name for name,c in sorted(read(DATA/'category_to_id.json').items(),key=lambda x:x[1])],'train_paths_sha256':configuration['train_manifest_sha256'],'scaler':{},'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all()})
 while updates<10000:
  loader=epoch_loader(ds,64,0,cfg['seed'],epoch)
  for batch_index,(images,labels) in enumerate(loader):
   if batch_index<cursor:continue
   x=images.cuda(non_blocking=True);y=drop_classes(labels.cuda(non_blocking=True),151,cfg['class_dropout']);t=torch.randint(1000,(len(y),),device='cuda');noise=torch.randn_like(x);a=schedule.alpha_bar[t,None,None,None];optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pred=model(a.sqrt()*x+(1-a).sqrt()*noise,t,y);per=(pred.float()-noise).square().flatten(1).mean(1);snr=a.flatten()/(1-a.flatten());weights=torch.minimum(snr,torch.full_like(snr,5))/snr if kind=='min_snr' else torch.ones_like(snr);loss=(weights*per).mean();unweighted=per.mean()
   assert torch.isfinite(loss);loss.backward();gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True));optimizer.step();update_ema(ema,model,cfg['ema_decay'])
   for k,v in extra.items():update_ema(v,model,float(k))
   updates+=1;seen+=len(y);cursor=batch_index+1;peak=torch.cuda.max_memory_allocated();assert peak<20*1024**3
   if updates==1 or updates%20==0:
    record={'update':updates,'step':initial_step+updates,'loss':float(loss.detach()),'unweighted_epsilon_mse':float(unweighted.detach()),'mean_weight':float(weights.mean()),'gradient_norm':gradnorm,'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.time()-start,'peak_vram_bytes':peak,'images_seen':seen}
    with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    job_update(key,update=updates,last_train=record);print(key,record,flush=True)
   if updates in [5000,10000]:
    # Store full training state before evaluation to avoid evaluation consuming resume RNG.
    save_training(folder/'latest_training.pt');cp=folder/f'update_{updates:06d}_teacher.pt';shared.atom_torch(cp,{'state_dict':track.state_dict(),'config':cfg,'step':initial_step+updates,'local_updates':updates,'weights':selected['weights'],'parameter_count':48883107,'submission_eligible':False,'experiment':configuration})
    metrics=metrics_for_model(track,cfg,folder/f'eval_{updates:06d}',selected['cfg'],selected['sampling_steps'],selected['weights'],key);ev={'update':updates,**metrics};evals.append(ev)
    with lockstate:STATE['jobs'][key]['evaluations']=evals.copy()
    candidate={'id':key+'_update'+str(updates),'checkpoint':str(cp),'metrics':metrics,'parent_k':selected['parent_k'],'weights':selected['weights'],'cfg':selected['cfg'],'sampling_steps':selected['sampling_steps'],'objective':kind}
    if metrics['official_fid']<best['metrics']['official_fid']:best=candidate
    if metrics['official_fid']<champion['metrics']['official_fid']:champion=candidate;write(OUT/'best_teacher_candidate.json',champion)
    diagnostic(key+f'/diag_{updates:06d}',model,track,cfg,selected['weights']);report();model.train()
   if updates==10000:break
  else:epoch+=1;cursor=0
 write(folder/'result.json',{'best':best,'evaluations':evals,'changed_factor':'timestep loss weighting only','checkpoint_raw_resume_parent':str(PARENTS[selected['parent_k']])});job_update(key,status='completed',metrics=best['metrics']);del model,ema,extra,track,optimizer;torch.cuda.empty_cache();return best

from augmentation_stage import run_augmentation_impl
def run_augmentation():
 return run_augmentation_impl(sys.modules[__name__])

def main():
 global selected,champion
 OUT.mkdir(exist_ok=True);q=(ROOT/'Experiment/teacher_improvement.lock').open('a');fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_lock=(ROOT/'Experiment/gt_gan.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'completed.json').exists(),'Teacher queue already completed'
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 for line in gpu.splitlines():
  if line.strip():assert int(line.split(',')[0])==os.getpid(),'Another GPU compute process is active'
 assert torch.cuda.is_available();torch.empty(1,device='cuda');torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 train=(BASE/'data/development/seed42_holdout5/train.txt').read_text().splitlines();held=(BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines();official=(DATA/'val_split.txt').read_text().splitlines();assert len(train)==16324 and len(held)==755 and len(official)==3020 and not set(train)&(set(held)|set(official));expected=hashlib.sha256('\n'.join(train).encode()).hexdigest();parents={}
 for k,p in PARENTS.items():
  d=torch.load(p,map_location='cpu',weights_only=False);assert d['step']==k*1000 and d['train_paths_sha256']==expected;parents[k]={'path':str(p),'sha256':sha(p),'step':d['step'],'train_manifest_hash':d['train_paths_sha256']};del d
 write(OUT/'preflight.json',{'gpu_processes_before_start':gpu,'train_manifest_hash':expected,'train_official_overlap':0,'train_valid_overlap':0,'parents':parents,'source_sha256':sha(Path(__file__))});write(OUT/'config.json',CONFIG);STATE.update({'status':'running','phase':'1_teacher_reproduction'});report();threading.Thread(target=heartbeat,daemon=True).start()
 candidates=[]
 for k in [150,200]:
  result=evaluation(f'1_reproduce_{k}k',k);candidates.append(result)
  existing=BASE/f'experiments/teacher_budget_noaug_fresh_200k/evaluation/step_{k*1000:06d}/ddim32_cfg2/official_metrics.json';old=read(existing);difference=result['metrics']['official_fid']-old['fid'];write(OUT/f'1_reproduce_{k}k/reproduction_delta.json',{'historical_official_fid':old['fid'],'remeasured_official_fid':result['metrics']['official_fid'],'difference':difference});assert abs(difference)<.1,'Teacher reproduction drift exceeds0.1; inspect implementation before training'
 base=min(candidates,key=lambda c:c['metrics']['official_fid']);STATE['phase']='1_CFG_EMA_sampler_selection';report()
 for scale in [1.5,2.5]:candidates.append(evaluation('1_cfg_'+str(scale).replace('.','_'),base['parent_k'],cfg_scale=scale))
 bestcfg=min(candidates,key=lambda c:c['metrics']['official_fid']);alternate=evaluation('1_ema0999',bestcfg['parent_k'],weights='ema:0.999',cfg_scale=bestcfg['cfg']);candidates.append(alternate);bestema=min(candidates,key=lambda c:c['metrics']['official_fid'])
 for steps in [50,64]:candidates.append(evaluation(f'1_ddim{steps}',bestema['parent_k'],weights=bestema['weights'],cfg_scale=bestema['cfg'],steps=steps))
 selected=min(candidates,key=lambda c:c['metrics']['official_fid']);champion=selected.copy();write(OUT/'phase1_selected.json',selected);write(OUT/'best_teacher_candidate.json',champion);run_augmentation();STATE['phase']='2_timestep_diagnostics';STATE['current_job']='2_initial_diagnostics';job_update('2_initial_diagnostics',status='running');raw,cfg=load_teacher(selected['parent_k'],'model');ema,cfg=load_teacher(selected['parent_k'],selected['weights']);diagnostic('2_initial_diagnostics',raw,ema,cfg,selected['weights']);del raw,ema;torch.cuda.empty_cache();job_update('2_initial_diagnostics',status='completed');STATE['phase']='2_objective_comparison';report()
 train_arm('2_uniform_mse','mse');train_arm('2_min_snr_gamma5','min_snr')
 STATE['phase']='completed';STATE['status']='completed';STATE['current_job']=None;write(OUT/'best_teacher_candidate.json',champion);write(OUT/'completed.json',STATE);report();stopped.set()
 # Source checkpoints remain immutable and student baselines are never changed.
 for k,p in PARENTS.items():assert sha(p)==parents[k]['sha256']
if __name__=='__main__':
 try:main()
 except BaseException:
  STATE.update({'status':'failed','error':traceback.format_exc()});OUT.mkdir(exist_ok=True);write(OUT/'failure.json',STATE);report();stopped.set();raise
