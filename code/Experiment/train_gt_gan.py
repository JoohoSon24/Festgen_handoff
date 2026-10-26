"""GT conditional GAN fine-tuning, NFE4 then NFE1; no DMD or teacher retraining."""
import os,sys,json,csv,time,hashlib,random,traceback,fcntl,copy,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from PIL import Image,ImageDraw
from torchvision.transforms import Compose,Resize,ToTensor,Normalize
from torchvision.transforms.functional import to_pil_image
import run_cfg as common
from model import Model
from distillation.core import coefficients,ddim_step
from cleanfid import fid,features
ROOT=common.ROOT;OUT=ROOT/'FID/gt_gan_20261019';DOC=ROOT/'Docs/Experiment';BASE=common.BASE;DATA=common.DATA
CKPTS={'n4':(common.CK,'few_nfe','d3c6d0cccc1b81d1709ed090abd68961197ef7417a7c6ae67724835bf7bfa596'), 'n1':(Path('/root/Diffusion/NFE-1-38FID/checkpoints/dmd_step_002500.ckpt'),'one_nfe','acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c')}
CONFIG={'method':'GT conditional hinge GAN + frozen original lowpass/output anchor','order':['n4','n1'],'steps':1000,'evaluate_steps':[250,500,1000],'d_warmup':100,'g_lr':1e-6,'d_lr':1e-4,'effective_batch':32,'microbatch':4,'gan_weight':0.1,'lowpass_anchor_weight':1.,'full_image_anchor_weight':0.1,'g_ema_decay':0.999,'r1_gamma':1.,'r1_interval':16,'translation_pixels':2,'seed':20261019,'selection':'minimum development valid_fid including unchanged baseline; report Official and precision/recall for frozen selected candidate','no_guaranteed_improvement':True}
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atom_torch(p,x):
 t=p.with_suffix(p.suffix+'.tmp');torch.save(x,t);t.replace(p)
class Discriminator(nn.Module):
 def __init__(self):
  super().__init__();blocks=[];cin=3
  for cout in [64,128,256,256]:
   blocks.extend([spectral_norm(nn.Conv2d(cin,cout,4,2,1)),nn.LeakyReLU(.2)]);cin=cout
  self.body=nn.Sequential(*blocks);self.out=spectral_norm(nn.Linear(256,1));self.embed=nn.Embedding(151,256);nn.init.normal_(self.embed.weight,0,.01)
 def forward(self,x,y):
  h=self.body(x).mean((2,3));return self.out(h).squeeze(1)+(h*self.embed(y)).sum(1)/16

def rollout(model,z,labels,grad=False):
 x=z
 for i,t in enumerate(model.sampling_times.tolist()):
  tt=torch.full_like(labels,t);nt=torch.full_like(labels,model.sampling_times[i+1].item() if i+1<len(model.sampling_times) else -1)
  a,s=coefficients(model.alpha_bar,tt);an,sn=coefficients(model.alpha_bar,nt)
  v=checkpoint(model.backbone,x,tt,labels,use_reentrant=False) if grad else model.backbone(x,tt,labels)
  x0=a*x-s*v.float();x=ddim_step(x,x0,a,s,an,sn)
 return x.clamp(-1,1)
def augment(x):
 # Same weak translation distribution on real and fake; no color/cutout.
 b,_,h,w=x.shape;p=CONFIG['translation_pixels'];xx=F.pad(x,(p,p,p,p),mode='replicate');offset=torch.randint(0,2*p+1,(b,2),device=x.device)
 return torch.stack([xx[i,:,int(offset[i,0]):int(offset[i,0])+h,int(offset[i,1]):int(offset[i,1])+w] for i in range(b)])
def edges(x):
 # Finite differences, diagnostic only, not an optimization loss.
 return ((x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()+ (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()).item()/2

def main():
 OUT.mkdir(exist_ok=True);lock=(ROOT/'Experiment/gt_gan.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert torch.cuda.is_available();torch.empty(1,device='cuda')
 active=torch.cuda.memory_allocated();assert active<512*1024**2
 torch.set_num_threads(4)
 torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);random.seed(CONFIG['seed'])
 mapping=json.loads((DATA/'category_to_id.json').read_text());trainlines=(BASE/'data/development/seed42_holdout5/train.txt').read_text().splitlines();hold=set((BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines());official=set((DATA/'val_split.txt').read_text().splitlines());assert not set(trainlines)&(hold|official)
 byclass={c:[] for c in range(151)}
 for line in trainlines:
  if line.strip():byclass[mapping[Path(line).parent.name]].append(DATA/line)
 assert all(byclass.values())
 transform=Compose([Resize((64,64)),ToTensor(),Normalize((.5,)*3,(.5,)*3)])
 # Cache GT uint8 tensors on CPU; only train split enters losses.
 realcache={}
 for c,paths in byclass.items():
  realcache[c]=[]
  for p in paths:
   with Image.open(p) as im:realcache[c].append(((transform(im.convert('RGB'))+1)*127.5).round().to(torch.uint8))
 def real_batch(labels):return torch.stack([realcache[c][random.randrange(len(realcache[c]))] for c in labels.tolist()]).cuda().float()/127.5-1
 old=ROOT/'FID/20261018_142153';ref=np.load(old/'reference_features.npy');valid=np.load(old/'valid_features.npy')
 fmap=None
 def extract(paths):
  nonlocal fmap
  if fmap is None:fmap=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
  return fid.get_files_features([str(p) for p in paths],model=fmap,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False)
 refmu=ref.mean(0);refcov=np.cov(ref,rowvar=False);vm=valid.mean(0);vc=np.cov(valid,rowvar=False)
 status={'status':'running','pid':os.getpid(),'config':CONFIG,'models':{k:{'status':'pending'} for k in CKPTS}}
 save(OUT/'status.json',status);save(OUT/'config.json',{**CONFIG,'train_count':len(trainlines),'train_manifest':str(BASE/'data/development/seed42_holdout5/train.txt'),'train_manifest_sha256':digest(BASE/'data/development/seed42_holdout5/train.txt'),'official_reference':str(BASE/'results/reference'),'valid_reference_count':755,'generated_count':3020,'evaluation_seed':'42+batch index; batch16 bf16','augmentation':'translation +/-2px only, applied to D real/fake; class-uniform train sampling'})
 def evaluate(m,folder,official_score=False):
  folder.mkdir(exist_ok=True);images=folder/'generated';images.mkdir(exist_ok=True);m.eval();gl=torch.arange(151).repeat_interleave(20);calls=[0]
  def hook(*_):calls[0]+=1
  handle=m.backbone.register_forward_hook(hook)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for bi,pos in enumerate(range(0,3020,16)):
    y=gl[pos:pos+16].cuda();z=torch.randn((len(y),3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42+bi));before=calls[0];x=rollout(m,z,y);assert calls[0]-before==len(m.sampling_times)
    for i,im in enumerate(x):
     idx=pos+i;to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(gl[idx]):04d}_{idx%20:02d}.png')
  handle.remove();g=extract(sorted(images.glob('*.png')));np.save(folder/'features.npy',g);mu=g.mean(0);cov=np.cov(g,rowvar=False)
  result={'valid_fid':float(fid.frechet_distance(mu,cov,vm,vc)),'nfe':len(m.sampling_times),'sample_count':3020}
  if official_score:result.update({'official_fid':float(fid.frechet_distance(mu,cov,refmu,refcov)),**common.pr(g,ref)})
  save(folder/'metrics.json',result);return result
 report=[]
 for name,(ck,mode,sha) in CKPTS.items():
  assert digest(ck)==sha;folder=OUT/name;folder.mkdir(exist_ok=True);state=status['models'][name];state['status']='preflight';save(OUT/'status.json',status)
  g=Model.load_checkpoint(str(ck),mode,device='cuda');anchor=Model.load_checkpoint(str(ck),mode,device='cuda').requires_grad_(False).eval();ema=copy.deepcopy(g).requires_grad_(False).eval();d=Discriminator().cuda()
  gp=list(g.parameters());dp=list(d.parameters());counts={'generator':sum(p.numel() for p in gp),'discriminator':sum(p.numel() for p in dp),'jointly_optimized':sum(p.numel() for p in gp+dp),'inference':g.count_parameters(),'frozen_anchor':anchor.count_parameters(),'ema':'stop-gradient weight average, no separate optimizer'};assert counts['jointly_optimized']<=100000000 and counts['inference']<=100000000;save(folder/'parameter_audit.json',counts)
  og=torch.optim.AdamW(gp,lr=CONFIG['g_lr'],betas=(0.,.99),weight_decay=0);od=torch.optim.AdamW(dp,lr=CONFIG['d_lr'],betas=(0.,.99),weight_decay=0)
  assert sum(p.numel() for a in og.param_groups+od.param_groups for p in a['params'])==counts['jointly_optimized']
  # Prove differentiable rollout matches released sampler and sends gradients to G.
  y=torch.tensor([0,47],device='cuda');z=torch.randn((2,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=rollout(g,z,y);b=g.sample(z.shape,device='cuda',category=y,seed=42)
  assert torch.allclose(a,b,atol=1e-5),float((a-b).abs().max())
  with torch.autocast('cuda',dtype=torch.bfloat16):fake=rollout(g,z,y,True);probe=-d(fake,y).mean()
  probe.backward();gradnorm=sum(p.grad.float().square().sum().item() for p in gp if p.grad is not None)**.5;assert math.isfinite(gradnorm) and gradnorm>0;og.zero_grad(set_to_none=True)
  save(folder/'preflight.json',{'sampler_max_difference':float((a-b).abs().max()),'generator_probe_gradient_norm':gradnorm,'nfe':len(g.sampling_times),'checkpoint_sha256':sha,'train_only_verified':True})
  # Paired frozen baseline for the exact current implementation and edge comparison.
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):baseline_edge=edges(anchor.sample((16,3,64,64),device='cuda',category=torch.arange(16,device='cuda'),seed=42))
  basepath=old/'baseline_n4/metrics.json' if name=='n4' else ROOT/'FID/sequential_20261019/n1_baseline/metrics.json';baseline=json.loads(basepath.read_text());best={'step':0,**baseline};best_ck=ck;save(folder/'baseline.json',baseline)
  state.update({'status':'training','step':0,'best_valid_fid':best['valid_fid'],'parameter_counts':counts});save(OUT/'status.json',status)
  log=(folder/'train.jsonl').open('a',encoding='utf-8');start=time.time();torch.cuda.reset_peak_memory_stats()
  def d_step(step,warm=False):
   d.train().requires_grad_(True);od.zero_grad(set_to_none=True);total=0.
   for micro in range(CONFIG['effective_batch']//CONFIG['microbatch']):
    yy=torch.randint(151,(CONFIG['microbatch'],),device='cuda');zz=torch.randn((len(yy),3,64,64),device='cuda');real=real_batch(yy)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ff=rollout(g,zz,yy)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=(F.relu(1-d(augment(real),yy)).mean()+F.relu(1+d(augment(ff),yy)).mean())
    (loss/(CONFIG['effective_batch']//CONFIG['microbatch'])).backward();total+=float(loss.detach())
   if not warm and step%CONFIG['r1_interval']==0:
    yy=torch.randint(151,(CONFIG['microbatch'],),device='cuda');rr=real_batch(yy).requires_grad_(True);pred=d(rr,yy);gr=torch.autograd.grad(pred.sum(),rr,create_graph=True)[0];reg=gr.square().flatten(1).sum(1).mean()*CONFIG['r1_gamma']*.5*CONFIG['r1_interval'];reg.backward()
   torch.nn.utils.clip_grad_norm_(dp,10,error_if_nonfinite=True);od.step();return total/(CONFIG['effective_batch']//CONFIG['microbatch'])
  for warm in range(CONFIG['d_warmup']):
   d_step(warm,True)
   if warm%25==0:print(name,'D warmup',warm,flush=True)
  for step in range(1,CONFIG['steps']+1):
   dl=d_step(step);d.eval().requires_grad_(False);g.train();og.zero_grad(set_to_none=True);gs=ans=0.
   for micro in range(CONFIG['effective_batch']//CONFIG['microbatch']):
    yy=torch.randint(151,(CONFIG['microbatch'],),device='cuda');zz=torch.randn((len(yy),3,64,64),device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):target=rollout(anchor,zz,yy)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     ff=rollout(g,zz,yy,True);adv=-d(augment(ff),yy).mean();anch=F.mse_loss(F.avg_pool2d(ff,4),F.avg_pool2d(target,4))+CONFIG['full_image_anchor_weight']*F.l1_loss(ff,target);loss=CONFIG['gan_weight']*min(1.,step/100)*adv+CONFIG['lowpass_anchor_weight']*anch
    assert torch.isfinite(loss);(loss/(CONFIG['effective_batch']//CONFIG['microbatch'])).backward();gs+=float(adv.detach());ans+=float(anch.detach())
   gn=float(torch.nn.utils.clip_grad_norm_(gp,1,error_if_nonfinite=True));og.step()
   with torch.no_grad():
    for ep,p in zip(ema.parameters(),g.parameters()):ep.lerp_(p,1-CONFIG['g_ema_decay'])
   peak=torch.cuda.max_memory_allocated();assert peak<20*1024**3
   if step==1 or step%10==0:
    record={'step':step,'d_hinge':dl,'g_adv':gs/(CONFIG['effective_batch']//CONFIG['microbatch']),'anchor':ans/(CONFIG['effective_batch']//CONFIG['microbatch']),'g_grad_norm':gn,'elapsed_seconds':time.time()-start,'peak_vram_bytes':peak};log.write(json.dumps(record)+'\n');log.flush();state.update(record);save(OUT/'status.json',status);print(name,record,flush=True)
   if step in CONFIG['evaluate_steps']:
    g.eval();cp=folder/f'step_{step:06d}.ckpt';atom_torch(cp,{'state_dict':ema.state_dict(),'step':step,'config':CONFIG,'source_sha256':sha,'parameter_count':g.count_parameters(),'nfe':len(g.sampling_times)})
    ev=evaluate(ema,folder/f'eval_{step:06d}');print(name,'evaluation',step,ev,flush=True)
    if ev['valid_fid']<best['valid_fid']:best={'step':step,**ev};best_ck=cp
    state['best_valid_fid']=best['valid_fid'];state['best_step']=best['step'];save(OUT/'status.json',status)
    # Resume state includes RNG, both optimizers, D, current G, EMA.
    atom_torch(folder/'latest_training.pt',{'generator':g.state_dict(),'ema':ema.state_dict(),'discriminator':d.state_dict(),'g_optimizer':og.state_dict(),'d_optimizer':od.state_dict(),'step':step,'config':CONFIG,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'source_sha256':sha})
  log.close();selected=Model.load_checkpoint(str(best_ck),mode,device='cuda');final=evaluate(selected,folder/'selected_evaluation',True);save(folder/'selected.json',{'selected_checkpoint':str(best_ck),'selected_step':best['step'],'original_checkpoint':str(ck),'selection':'development valid_fid including step0','baseline':baseline,'selected_metrics':final,'official_delta':final['official_fid']-baseline['official_fid'],'status':'improved' if final['official_fid']<baseline['official_fid'] else 'no_official_improvement','original_sha256_after':digest(ck)})
  assert digest(ck)==sha
  # Static matched-noise overview, real samples shown separately (not paired GT).
  preview=Image.new('RGB',(16*80,3*104+48),'white');draw=ImageDraw.Draw(preview);draw.text((5,5),name+' GT / original / selected; GT not paired to noise',(0,0,0))
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   yy=torch.arange(16,device='cuda');zz=torch.randn((16,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));aa=rollout(anchor,zz,yy);bb=rollout(selected,zz,yy);rr=real_batch(yy)
  for ri,(batch,title) in enumerate([(rr,'GT train'),(aa,'Original'),(bb,'Selected')]):
   draw.text((5,30+ri*104),title,(0,0,0))
   for i,im in enumerate(batch):preview.paste(to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).resize((76,76),Image.Resampling.NEAREST),(i*80,48+ri*104))
  preview.save(folder/'comparison_grid.png');
  # Matched original/selected center crops (2x zoom); GT row is unpaired.
  crops=Image.new('RGB',(16*80,3*104+48),'white');cd=ImageDraw.Draw(crops);cd.text((5,5),name+' center32 crops; nearest zoom, NOT an edge score',(0,0,0))
  for ri,(batch,title) in enumerate([(rr,'GT train'),(aa,'Original'),(bb,'Selected')]):
   cd.text((5,30+ri*104),title,(0,0,0))
   for i,im in enumerate(batch):crops.paste(to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).crop((16,16,48,48)).resize((76,76),Image.Resampling.NEAREST),(i*80,48+ri*104))
  crops.save(folder/'center_crop_grid.png');final['diagnostic_mean_abs_gradient_original']=edges(aa);final['diagnostic_mean_abs_gradient_selected']=edges(bb);save(folder/'edge_diagnostics.json',{'original':edges(aa),'selected':edges(bb),'warning':'first16classes, finite-difference diagnostic; not proof of perceptual sharpness or fidelity'});state.update({'status':'completed','selected_step':best['step'],'selected_checkpoint':str(best_ck),'metrics':final});report.append({'model':name,'baseline':baseline,'selected':final,'step':best['step']});save(OUT/'status.json',status)
  del g,anchor,ema,d,selected,og,od;torch.cuda.empty_cache()
 status['status']='completed';save(OUT/'status.json',status)
 lines=['# GT 기반 GAN fine-tuning 결과','','NFE4 → NFE1 순으로 학습. DMD/fake-score 없음. Development Valid FID로 원본 포함 후보를 선택하고 선택된 모델의 Official FID를 측정했다. 개선을 보장하지 않으며 원본은 보존한다.','','| 모델 | 선택 step | 기준 Official | 선택 Official | 기준 Valid | 선택 Valid | Precision | Recall |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in report:
  b=r['baseline'];s=r['selected'];lines.append(f"| {r['model']} | {r['step']} | {b['official_fid']:.4f} | {s['official_fid']:.4f} | {b['valid_fid']:.4f} | {s['valid_fid']:.4f} | {s['precision_k3']:.4f} | {s['recall_k3']:.4f} |")
 lines+=['','단일 training seed의1000 update pilot 결과. Selected step0이면 fine-tuned 후보가 development 기준 개선하지 못해 원본을 유지했다는 뜻이다. 경계 강도만을 품질 향상으로 해석하지 않는다. 원본/선택/GT grid는 FID/gt_gan_20261019/<model>/comparison_grid.png를 확인한다.'];(DOC/'GT_GAN_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);save(OUT/'failure.json',{'status':'failed','traceback':traceback.format_exc()});raise
