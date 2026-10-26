"""Explicit, read-only checkpoint reproduction; run separately when GPU is free."""
import os,json,fcntl,hashlib
from pathlib import Path
import torch,numpy as np
from torchvision.transforms.functional import to_pil_image
from cleanfid import fid,features
import train_gt_gan as source
from model import Model
ROOT=Path('/root/Diffusion 3');CK=Path('/root/Diffusion 2/FID/gt_gan_20261019/n4/step_001000.ckpt')
def main():
 with (ROOT/'Experiment/gt_gan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert source.digest(CK)=='e99cb3a050c21084acb04c6d0ae80d73b225307cd232f6a8322efbedc977445d'
  assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  out=ROOT/'FID/reproduce_nfe4_fid32';out.mkdir(exist_ok=True);images=out/'generated';images.mkdir(exist_ok=True);model=Model.load_checkpoint(str(CK),'few_nfe',device='cuda');assert model.count_parameters()==48883107;cats=torch.arange(151).repeat_interleave(20);calls=[0];handle=model.backbone.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for bi,pos in enumerate(range(0,3020,16)):
    y=cats[pos:pos+16].cuda();before=calls[0];x=model.sample((len(y),3,64,64),category=y,seed=42+bi);assert calls[0]-before==4 and torch.isfinite(x).all()
    for i,im in enumerate(x):idx=pos+i;to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(cats[idx]):04d}_{idx%20:02d}.png')
  handle.remove();net=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False);fs=fid.get_files_features([str(p) for p in sorted(images.glob('*.png'))],model=net,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False);np.save(out/'features.npy',fs);ref=np.load(ROOT/'FID/20261018_142153/reference_features.npy');val=np.load(ROOT/'FID/20261018_142153/valid_features.npy');mu=fs.mean(0);cov=np.cov(fs,rowvar=False)
  metrics={'official_fid':float(fid.frechet_distance(mu,cov,ref.mean(0),np.cov(ref,rowvar=False))),'valid_fid':float(fid.frechet_distance(mu,cov,val.mean(0),np.cov(val,rowvar=False))),**source.common.pr(fs,ref),'checkpoint':str(CK),'sha256':source.digest(CK),'nfe':4,'parameters':model.count_parameters(),'seed_base':42,'batch':16,'precision':'bf16','generated_count':3020,'expected_official_fid':32.664720239619726};metrics['official_difference']=metrics['official_fid']-metrics['expected_official_fid'];(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics);assert abs(metrics['official_difference'])<.02
if __name__=='__main__':main()
