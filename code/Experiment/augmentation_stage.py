"""Isolated offline-augmentation stage; original checkpoint and comparison arms stay intact."""
import inspect,hashlib,time,copy
from pathlib import Path

def run_augmentation_impl(m):
 m.STATE['phase']='5_offline_augmentation';m.STATE['current_job']='5_prepare_data';m.job_update('5_prepare_data',status='waiting')
 while not (m.ROOT/'Data/summary.json').exists():
  failure=m.ROOT/'Data/failure.txt'
  if failure.exists():raise RuntimeError(failure.read_text())
  time.sleep(5)
 summary=m.read(m.ROOT/'Data/summary.json');assert summary['total_images']==26878
 m.job_update('5_prepare_data',status='completed',summary=summary)
 original_data=m.DATA;original_get_cache=m.get_cache;original_cache=m.cache;original_diagnostic=m.diagnostic
 m.DATA=m.ROOT/'Data';m.cache=None
 def aug_cache():
  if m.cache is None:m.cache=m.CachedImages((m.DATA/'train.txt').read_text().splitlines(),m.read(m.DATA/'category_to_id.json'))
  return m.cache
 m.get_cache=aug_cache;m.diagnostic=lambda *args,**kwargs:None
 src=inspect.getsource(m.train_arm).replace('def train_arm(', 'def train_augmented(')
 src=src.replace('10000','30000').replace('[5000,30000]','[5000,10000,20000,30000]')
 src=src.replace("manifest=BASE/'data/development/seed42_holdout5/train.txt'","manifest=DATA/'train.txt'")
 src=src.replace("assert st['train_paths_sha256']==hashlib.sha256('\\n'.join(ds.paths).encode()).hexdigest();", "parent_manifest_hash=st['train_paths_sha256'];")
 src=src.replace("'train_manifest_sha256':st['train_paths_sha256']", "'train_manifest_sha256':hashlib.sha256('\\n'.join(ds.paths).encode()).hexdigest(),'parent_train_manifest_sha256':parent_manifest_hash")
 src=src.replace("'augmentation':'none'", "'augmentation':'offline balanced178/class, flip and rotation+-10, no color transform'")
 src=src.replace("optimizer.load_state_dict(st['optimizer']);", "optimizer.load_state_dict(st['optimizer']);\n for group in optimizer.param_groups:group['lr']=5e-5\n ")
 src=src.replace("cfg['data_root']=str(DATA);", "cfg['learning_rate']=5e-5;cfg['data_root']=str(DATA);")
 src=src.replace("cursor=st['batch_in_epoch'];", "cursor=0;")
 src=src.replace("'hypothesis':'change timestep weighting only'", "'hypothesis':'class balancing + offline augmentation + conservative LR; exploratory combined intervention'")
 src=src.replace("save_training(folder/'latest_training.pt')", "save_training(folder/f'update_{updates:06d}_training.pt')")
 src=src.replace("'changed_factor':'timestep loss weighting only'", "'changed_factor':'class balancing, offline spatial augmentation, and reduced LR'")
 compile(src,'train_augmented.py','exec');assert 'updates in [5000,10000,20000,30000]' in src
 (m.ROOT/'Experiment/train_augmented.py').write_text(src);exec(compile(src,'train_augmented.py','exec'),m.__dict__)
 try:
  m.write(m.OUT/'5_augmentation_protocol.json',{'parent':str(m.PARENTS[m.selected['parent_k']]),'parent_sha256':m.sha(m.PARENTS[m.selected['parent_k']]),'data':summary,'updates':30000,'lr':5e-5,'objective':'mse','evaluations':[5000,10000,20000,30000]})
  result=m.train_augmented('5_augmented_continuation','mse');m.write(m.OUT/'5_augmentation_result.json',result)
 finally:
  m.DATA=original_data;m.get_cache=original_get_cache;m.cache=original_cache;m.diagnostic=original_diagnostic
