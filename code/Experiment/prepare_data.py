import json,random,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path('/root/Diffusion 3');BASE=Path('/root/Diffusion/code/workspace_distillation');SRC=BASE/'data/pokemon-generation-one-22k';DEST=ROOT/'Data';rng=random.Random(20261019)
def main():
 lines=(BASE/'data/development/seed42_holdout5/train.txt').read_text().splitlines(); excluded=set((BASE/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines())|set((SRC/'val_split.txt').read_text().splitlines());assert not set(lines)&excluded
 groups={}
 for p in lines:groups.setdefault(Path(p).parent.name,[]).append(p)
 target=max(map(len,groups.values()));manifest=[];records=[]
 for name,paths in sorted(groups.items()):
  folder=DEST/name;folder.mkdir(parents=True,exist_ok=True)
  for i,src in enumerate(paths):
   dst=folder/f'original_{i:04d}{Path(src).suffix}';
   if not dst.exists() or dst.stat().st_size!=(SRC/src).stat().st_size:shutil.copy2(SRC/src,dst)
   manifest.append(str(dst.relative_to(DEST)));records.append({'output':manifest[-1],'source':src,'flip':False,'angle':0,'original':True})
  order=paths.copy();rng.shuffle(order)
  for i in range(target-len(paths)):
   src=order[i%len(order)];flip=bool(rng.randrange(2));angle=rng.uniform(-10,10)
   with Image.open(SRC/src) as im:
    im=im.convert('RGB');im=ImageOps.mirror(im) if flip else im
    # Reflect-pad before rotation; crop returns original size without introduced solid corners.
    import numpy as np
    import math
    a=np.asarray(im);w,h=im.size;pad=max(2,math.ceil(max(w,h)*.12));padded=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),mode='reflect'));rot=padded.rotate(angle,resample=Image.Resampling.BICUBIC);im=rot.crop((pad,pad,pad+w,pad+h));dst=folder/f'aug_{i:04d}.png';im.save(dst)
   manifest.append(str(dst.relative_to(DEST)));records.append({'output':manifest[-1],'source':src,'flip':flip,'angle':angle,'original':False})
 (DEST/'train.txt').write_text('\n'.join(manifest)+'\n');shutil.copy2(SRC/'category_to_id.json',DEST/'category_to_id.json');(DEST/'provenance.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));summary={'classes':len(groups),'target_per_class':target,'original_images':len(lines),'augmented_images':len(manifest)-len(lines),'total_images':len(manifest),'seed':20261019,'manifest_sha256':hashlib.sha256('\n'.join(manifest).encode()).hexdigest(),'source_manifest_sha256':hashlib.sha256('\n'.join(lines).encode()).hexdigest(),'train_excluded_path_overlap':0,'augmentation':'native-resolution horizontal flip p=.5 and rotation uniform[-10,10], reflected border, bicubic; no color jitter'};(DEST/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  (DEST/'failure.txt').write_text(traceback.format_exc());raise
