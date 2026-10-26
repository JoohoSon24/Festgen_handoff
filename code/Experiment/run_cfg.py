import os,sys,json,hashlib,time,csv,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Experiment/source'))
import torch,numpy as np
from PIL import Image
from model import Model
from distillation.core import coefficients,ddim_step
from cleanfid import fid,features
CK=Path('/root/Diffusion/NFE-4-33FID/checkpoints/few_nfe.ckpt')
DATA=Path('/root/Diffusion/code/workspace_distillation/data/pokemon-generation-one-22k')
BASE=Path('/root/Diffusion/code/workspace_distillation')
def write(p,x):
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def dist(a,b):return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)
def radii(a):
 return np.concatenate([np.partition(np.where(np.arange(len(a))[None,:]==np.arange(i,min(i+128,len(a)))[:,None],np.inf,dist(a[i:i+128],a)),2,axis=1)[:,2] for i in range(0,len(a),128)])
def cover(q,c,r):return float(np.concatenate([(dist(q[i:i+128],c)<=r).any(1) for i in range(0,len(q),128)]).mean())
def pr(g,r):return {'precision_k3':cover(g,r,radii(r)),'recall_k3':cover(r,g,radii(g))}
def main():
 assert torch.cuda.is_available(),'CUDA unavailable; run in GPU-capable host context'
 torch.empty(1,device='cuda')
 digest=hashlib.sha256(CK.read_bytes()).hexdigest()
 assert digest=='d3c6d0cccc1b81d1709ed090abd68961197ef7417a7c6ae67724835bf7bfa596'
 m=Model.load_checkpoint(str(CK),'few_nfe',device='cuda');params=m.count_parameters();assert params<=100000000
 trials=[('baseline_n4',None,[999,741,483,226]),('two_step_cfg1',1.,[999,483]),('two_step_cfg1_5',1.5,[999,483]),('two_step_cfg2',2.,[999,483]),('two_step_cfg2_5',2.5,[999,483])]
 stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime());out=ROOT/'FID'/stamp;out.mkdir()
 write(ROOT/'FID/latest_run.json',{'run':str(out),'pid':os.getpid(),'status':'running'})
 mapping=json.loads((DATA/'category_to_id.json').read_text())
 lines=(DATA/'val_split.txt').read_text().splitlines();refpaths=sorted((BASE/'results/reference').glob('*.png'));assert len(refpaths)==len(lines)==3020
 ref_labels=np.array([mapping[Path(x).parent.name] for x in lines])
 holdpaths=[DATA/x for x in (BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines() if x.strip()];assert len(holdpaths)==755
 write(out/'protocol.json',{'checkpoint':str(CK),'sha256':digest,'parameters':params,'seed':'42 + batch_index','batch':16,'precision':'bf16','reference':str(BASE/'results/reference'),'generated_count':3020,'samples_per_class':20,'trials':trials,'selection':'Official FID; repeated Official selection is not independent test','warning':'student absorbed CFG2; null branch and 2-step path are experimental','class_pr_warning':'20 samples/class; high variance exploratory estimates'})
 model_f=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
 def extract(paths):return fid.get_files_features([str(p) for p in paths],model=model_f,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=True)
 # Reference export file names must follow evaluator sequential validation order.
 assert [p.name for p in refpaths]==[f'{i:04d}.png' for i in range(3020)]
 r=extract(refpaths);np.save(out/'reference_features.npy',r)
 # Export holdout with identical resize/normalization/PNG convention.
 from torchvision.transforms import Compose,Resize,ToTensor,Normalize
 from torchvision.transforms.functional import to_pil_image
 transform=Compose([Resize((64,64)),ToTensor(),Normalize((.5,)*3,(.5,)*3)])
 hd=out/'valid_reference';hd.mkdir()
 for i,p in enumerate(holdpaths):
  with Image.open(p) as im:x=transform(im.convert('RGB'))
  to_pil_image(((x+1)/2).clamp(0,1)).save(hd/f'{i:04d}.png')
 h=extract(sorted(hd.glob('*.png')));np.save(out/'valid_features.npy',h)
 results=[]
 for name,scale,times in trials:
  folder=out/name;folder.mkdir();images=folder/'generated';images.mkdir();start=time.time();calls=[0]
  def hook(*args):calls[0]+=1
  handle=m.backbone.register_forward_hook(hook);torch.cuda.reset_peak_memory_stats()
  categories=torch.arange(151).repeat_interleave(20)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for batch_i,pos in enumerate(range(0,3020,16)):
    labels=categories[pos:pos+16].cuda();gen=torch.Generator(device='cuda').manual_seed(42+batch_i)
    x=torch.randn((len(labels),3,64,64),device='cuda',generator=gen);before=calls[0]
    for i,t in enumerate(times):
     tt=torch.full_like(labels,t);nt=torch.full_like(labels,times[i+1] if i+1<len(times) else -1)
     a,s=coefficients(m.alpha_bar,tt);an,sn=coefficients(m.alpha_bar,nt)
     if scale is None:v=m.backbone(x,tt,labels).float()
     else:
      u=m.backbone(x,tt,torch.full_like(labels,151)).float();c=m.backbone(x,tt,labels).float();v=u+scale*(c-u)
     x0=a*x-s*v;x=ddim_step(x,x0,a,s,an,sn)
    assert calls[0]-before==4
    assert torch.isfinite(x).all()
    for j,img in enumerate(x.clamp(-1,1)):
     index=pos+j;to_pil_image(((img.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(categories[index]):04d}_{index%20:02d}.png')
    if batch_i%20==0:print(name,pos,'/3020',flush=True)
  handle.remove();g=extract(sorted(images.glob('*.png')));np.save(folder/'generated_features.npy',g)
  score=lambda a:float(fid.frechet_distance(g.mean(0),np.cov(g,rowvar=False),a.mean(0),np.cov(a,rowvar=False)))
  metric={'experiment':name,'official_fid':score(r),'valid_fid':score(h),**pr(g,r),'mean_feature_distance_squared':float(((g.mean(0)-r.mean(0))**2).sum()),'parameters':params,'nfe':4,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-start}
  metric['covariance_term']=metric['official_fid']-metric['mean_feature_distance_squared']
  assert metric['peak_vram_bytes']<20*1024**3
  with (folder/'class_metrics.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['class_id','generated_count','reference_count','precision_k3','recall_k3','mean_feature_distance_squared','generated_variance_trace','reference_variance_trace']);w.writeheader()
   for c in range(151):
    a=g[np.array([int(p.stem.split('_')[0]) for p in sorted(images.glob('*.png'))])==c];b=r[ref_labels==c];assert len(a)==len(b)==20
    w.writerow({'class_id':c,'generated_count':len(a),'reference_count':len(b),**pr(a,b),'mean_feature_distance_squared':float(((a.mean(0)-b.mean(0))**2).sum()),'generated_variance_trace':float(a.var(0).sum()),'reference_variance_trace':float(b.var(0).sum())})
  write(folder/'metrics.json',metric);results.append(metric);write(out/'results.json',results);print(json.dumps(metric),flush=True)
 write(ROOT/'FID/latest_run.json',{'run':str(out),'status':'completed','pid':os.getpid()})
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'FID/failure.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
