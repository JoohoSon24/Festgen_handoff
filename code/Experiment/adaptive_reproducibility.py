"""Local source snapshots, checkpoint manifests, and idempotent decision records."""
import json,hashlib,shutil,time,os,sys,importlib.metadata,subprocess
from pathlib import Path
import torch

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def weight_hash(model):
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):
  tensor=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def log_decision(path,decision):
 existing=[]
 if path.exists():
  for line in path.read_text().splitlines():
   try:existing.append(json.loads(line))
   except ValueError:pass
 if any(row.get('decision_id')==decision['decision_id'] for row in existing):return
 row={**decision,'recorded_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())}
 with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def capture_revision(root,out,config_path,script_path):
 paths=[Path(script_path),Path(config_path)]
 for name in ['adaptive_budget_policy.py','adaptive_reproducibility.py','student90_model.py','teacher_core.py','train_gt_gan.py','run_cfg.py']:paths.append(root/'Experiment'/name)
 paths+=sorted((root/'Experiment/source').rglob('*.py'));hashes={str(p):file_hash(p) for p in paths};revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();dest=out/'provenance/revisions'/revision[:16];dest.mkdir(parents=True,exist_ok=True)
 for p in paths:
  relative=p.relative_to(root/'Experiment') if p.is_relative_to(root/'Experiment') else Path(p.name);target=dest/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(p,target)
 package_versions={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions() if dist.metadata.get('Name')};environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'packages':package_versions}
 environment['deterministic_algorithms']=torch.are_deterministic_algorithms_enabled()
 environment['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
 (dest/'files_sha256.json').write_text(json.dumps(hashes,indent=2));(dest/'environment.json').write_text(json.dumps(environment,indent=2));data_hashes=root/'docs/reproducibility/NFE4_FID32/DATASET_SHA256.json'
 references={'dataset_content_manifest':str(data_hashes),'dataset_manifest_sha256':file_hash(data_hashes) if data_hashes.exists() else None,'references':{}}
 for name in ['reference_features.npy','valid_features.npy']:
  p=root/'FID/20261018_142153'/name;references['references'][str(p)]=file_hash(p)
 (dest/'dataset_and_reference.json').write_text(json.dumps(references,indent=2));return {'revision':revision,'path':str(dest),'source_hashes':hashes}

def register_checkpoint(folder,path,role,update,parent,config_sha,policy_sha,revision):
 record={'role':role,'update':update,'path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size,'parent':str(parent),'config_sha256':config_sha,'policy_sha256':policy_sha,'source_revision':revision,'created_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())};target=folder/'checkpoint_manifest.json';data=json.loads(target.read_text()) if target.exists() else {};data[role]=record;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(target)
 with (folder/'checkpoint_history.jsonl').open('a') as history:history.write(json.dumps(record,ensure_ascii=False)+'\n');history.flush();os.fsync(history.fileno())
 return record
