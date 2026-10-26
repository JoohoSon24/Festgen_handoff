"""User-authorized cleanup of regenerable, completed augmentation input copies only."""
from pathlib import Path
import json,os,shutil,hashlib,time
r=Path('/root/Diffusion 3');data=r/'Data';base=Path('/root/Diffusion/code/workspace_distillation/data/pokemon-generation-one-22k')
status=json.loads((r/'FID/teacher_improvement_20261019/status.json').read_text());assert status['status']=='completed'
config=json.loads((r/'Experiment/configs/exp-20261020-student90-cfg25-review.json').read_text());assert not Path(config['training']['train_manifest']).resolve().is_relative_to(data.resolve())
records=json.loads((data/'provenance.json').read_text());assert len(records)==26878
candidates=[]
for row in records:
 p=data/row['output'];source=base/row['source']
 assert p.resolve().is_relative_to(data.resolve()) and not p.is_symlink() and source.is_file()
 assert p.name.startswith(('original_','aug_')) and p.suffix.lower() in ['.png','.jpg','.jpeg','.webp','.bmp','.gif']
 if p.exists():candidates.append({'path':str(p),'source':str(source),'bytes':p.stat().st_size,'allocated':p.stat().st_blocks*512})
manifest={'authorized_by':'2026-10-20 user asked storage cleanup and minibatch increase','scope':'only regenerable augmentation input copies; no checkpoints or original train data','files':candidates,'allocated_bytes':sum(x['allocated'] for x in candidates),'before_free':shutil.disk_usage(r).free,'status':'planned'}
p=r/'docs/STORAGE_CLEANUP_MANIFEST.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for row in candidates:Path(row['path']).unlink()
manifest.update(status='completed',after_free=shutil.disk_usage(r).free,completed_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()));p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(data/'IMAGE_COPIES_REMOVED.json').write_text(json.dumps({'state':'images removed; metadata retained','recreate':'Experiment/prepare_data.py','manifest':str(p)},indent=2));print({k:v for k,v in manifest.items() if k!='files'})
