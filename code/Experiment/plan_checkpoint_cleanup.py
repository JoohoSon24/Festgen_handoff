"""Only confirmed failing NFE1/4 checkpoints outside protected reproducibility roots."""
import os,re,json,collections
from pathlib import Path
import torch
R=Path('/root/Diffusion 3');old=json.loads((R/'docs/STORAGE_CHECKPOINT_INVENTORY.json').read_text());index={}
for row in old:
 p=Path(row['path'])
 if p.exists():s=p.stat();index[(s.st_dev,s.st_ino)]=row
protected_paths=[Path('/root/Diffusion/NFE-4-33FID/checkpoints/few_nfe.ckpt'),Path('/root/Diffusion/NFE-1-38FID/checkpoints/dmd_step_002500.ckpt'),Path('/root/Diffusion 2/FID/gt_gan_20261019/n4/step_001000.ckpt'),Path('/root/Diffusion 2/FID/gt_gan_20261019/n4/latest_training.pt')]
protected={(p.stat().st_dev,p.stat().st_ino) for p in protected_paths}
groups=collections.defaultdict(list)
for root in ['/root/Diffusion','/root/Diffusion 2']:
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d not in ['.venv','.git','__pycache__'] and not (Path(base)/d).is_symlink()]
  for name in files:
   p=Path(base)/name
   if p.suffix.lower() not in ['.pt','.pth','.ckpt','.safetensors'] or p.is_symlink():continue
   s=p.stat();groups[(s.st_dev,s.st_ino)].append(p)
candidates=[];held=[]
for inode,paths in groups.items():
 p=paths[0]
 if inode in protected:held.append({'path':str(p),'reason':'protected baseline/reproducibility ancestor'});continue
 try:st=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 except Exception as e:held.append({'path':str(p),'reason':'metadata load unavailable'});continue
 if not isinstance(st,dict):continue
 stage=st.get('stage');n=st.get('nfe');times=st.get('sampling_times');sd=st.get('state_dict',{})
 if times is None and isinstance(sd,dict):times=sd.get('sampling_times')
 if isinstance(times,torch.Tensor):n=len(times)
 if n is None and stage in ['n1','n4']:n=int(stage[1:])
 # GT GAN full resume stores the sampling grid in generator state.
 if n is None and isinstance(st.get('generator'),dict):
  times=st['generator'].get('sampling_times')
  if isinstance(times,torch.Tensor):n=len(times)
 if n not in [1,4]:del st;continue
 step=st.get('step',st.get('update'));ms=[]
 for m in index.get(inode,{}).get('official_measurements',[]):
  c=m.get('conditions',{});mn=c.get('nfe');seed=c.get('seed',c.get('seed_base',42));same_nfe=mn==n
  if same_nfe:ms.append({'fid':m['official_fid'],'source':m['source'],'method':'direct checkpoint Official reference'})
 # Exact update-specific evaluation beside full/milestone checkpoint; do not substitute a different update.
 if isinstance(step,int):
  for parent in paths:
   folder=parent.parent
   for sub in folder.iterdir():
    if not sub.is_dir() or not (sub.name.startswith('eval_') or sub.name.startswith('official_step_')):continue
    match=re.search(r'(\d{3,6})',sub.name)
    if not match or int(match.group())!=step:continue
    for filename in ['official_metrics.json','metrics.json']:
     metric=sub/filename
     if not metric.exists():continue
     try:m=json.loads(metric.read_text())
     except (ValueError,OSError):continue
     fid=m.get('official_fid')
     if fid is None and (m.get('reference_split')=='official' or filename=='official_metrics.json'):fid=m.get('fid')
     if isinstance(fid,(int,float)) and m.get('nfe',n)==n:ms.append({'fid':fid,'source':str(metric),'method':f'exact step{step} EMA evaluation'})
 # Export metadata identifies the source stage exactly; use only explicit measured source inode.
 source=st.get('source_checkpoint')
 if isinstance(source,str) and Path(source).exists():
  ss=Path(source).stat()
  for m in index.get((ss.st_dev,ss.st_ino),{}).get('official_measurements',[]):
   if m.get('conditions',{}).get('nfe')==n:ms.append({'fid':m['official_fid'],'source':m['source'],'method':'export explicit source checkpoint'})
 del st
 if not ms:held.append({'path':str(p),'nfe':n,'step':step,'reason':'Official FID not verified for this update'});continue
 best=min(m['fid'] for m in ms);threshold={1:39.,4:34.}[n]
 if best<threshold:held.append({'path':str(p),'nfe':n,'step':step,'best_official_fid':best,'reason':'at least one verified measurement below cutoff'});continue
 s=p.stat();candidates.append({'paths':[str(x) for x in paths],'inode':list(inode),'allocated_bytes':s.st_blocks*512,'logical_bytes':s.st_size,'nlink':s.st_nlink,'nfe':n,'step':step,'best_verified_official_fid':best,'evidence':ms})
candidates.sort(key=lambda x:x['allocated_bytes'],reverse=True)
plan={'status':'planned','user_authorization':'2026-10-20 remove old NFE4>=34 / NFE1>=39, protect current experiments and NFE4 FID32 reproduction','protected_roots':['/root/Diffusion 3'],'protected_checkpoint_paths':[str(x) for x in protected_paths],'candidates':candidates,'held':held,'unique_allocated_bytes':sum(x['allocated_bytes'] for x in candidates)}
(R/'docs/CHECKPOINT_CLEANUP_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
print('candidate_count',len(candidates),'GiB',plan['unique_allocated_bytes']/1024**3)
for c in candidates:print(f"{c['allocated_bytes']/1024**3:.3f}GiB NFE{c['nfe']} FID{c['best_verified_official_fid']:.6f} step{c['step']} links{len(c['paths'])}/{c['nlink']} {c['paths']}")
