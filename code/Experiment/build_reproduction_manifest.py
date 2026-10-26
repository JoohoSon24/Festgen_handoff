import json,hashlib,os,time
from pathlib import Path
R=Path('/root/Diffusion 3');D=Path('/root/Diffusion 2');BASE=Path('/root/Diffusion/code/workspace_distillation');data=BASE/'data/pokemon-generation-one-22k';folder=R/'docs/reproducibility/NFE4_FID32'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
paths=[Path('/root/Diffusion/NFE-4-33FID/checkpoints/few_nfe.ckpt'),Path('/root/Diffusion/NFE-1-38FID/checkpoints/dmd_step_002500.ckpt'),D/'FID/gt_gan_20261019/n4/step_001000.ckpt',D/'FID/gt_gan_20261019/n4/latest_training.pt',D/'Experiment/train_gt_gan.py',D/'Experiment/run_cfg.py',D/'FID/BASELINES.json',D/'FID/GAN_OFFICIAL_SUMMARY.json',D/'Docs/BASELINES.md',D/'Docs/GAN_OFFICIAL_FINAL_REPORT.md',R/'Experiment/reproduce_nfe4_fid32.py',BASE/'data/development/seed42_holdout5/train.txt',BASE/'data/development/seed42_holdout5/holdout.txt',data/'category_to_id.json',data/'val_split.txt']
paths+=list((D/'Experiment/source').rglob('*.py'))
for top in [D/'FID/gt_gan_20261019/n4/eval_001000',D/'FID/gt_gan_20261019/n4/selected_evaluation',R/'FID/20261018_142153']:
 paths +=[p for p in top.rglob('*') if p.is_file() and p.suffix in ['.json','.npy']]
for p in (BASE/'environment').glob('*'):
 if p.is_file() and p.stat().st_size<2_000_000:paths.append(p)
records=[]
for p in dict.fromkeys(paths):
 if p.is_file():records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
# Dataset content checksums support reproduction without making another image copy.
lines=set()
for manifest in [BASE/'data/development/seed42_holdout5/train.txt',BASE/'data/development/seed42_holdout5/holdout.txt',data/'val_split.txt']:lines.update(manifest.read_text().splitlines())
image_records=[]
for line in sorted(lines):
 p=data/line;image_records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
(folder/'DATASET_SHA256.json').write_text(json.dumps(image_records,ensure_ascii=False,indent=2));manifest={'scope':'checkpoint inference + GAN continuation from preserved original student; original teacher-from-scratch lineage incomplete','created_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'protected_files':records,'dataset_image_count':len(image_records),'python':str(BASE/'.venv/bin/python'),'checkpoint_baselines':{'nfe4_official_fid':32.664720239619726,'nfe1_official_fid':38.605125777247}}
(folder/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('preserved source/checkpoint files',len(records),'dataset images',len(image_records))
