"""Build an isolated adaptive runner without touching the active configuration/process."""
from pathlib import Path
import json,copy,ast
R=Path('/root/Diffusion 3');OLD=R/'Experiment/configs/exp-20261020-student90-cfg25-restart17.json';NEW=R/'Experiment/configs/exp-20261021-student90-adaptive.json';c=json.loads(OLD.read_text());assert c['total_distillation_updates']==612000
base={'guidance32':200000,'n16':30000,'n8':30000,'n4':40000,'n2':50000,'n1':75000};caps={'guidance32':250000,'n16':45000,'n8':45000,'n4':60000,'n2':75000,'n1':112500};gaps={'guidance32':2.,'n16':1.,'n8':1.5,'n4':2.,'n2':3.,'n1':4.}
for spec in c['stages']:spec['updates']=base[spec['name']]
c['total_distillation_updates']=425000;c['budget_revision']={'policy':'rebalance toward CFG transfer; adaptive bounded extensions','authorized_by':'2026-10-21 user approved200K/30K/30K/40K/50K/75K and automatic FID review/extensions with reproducible storage'}
c['adaptive_policy']={'version':'adaptive-v1','selection_metric':'official_fid','trend_intervals':2,'min_meaningful_fid_improvement':.2,'max_gap_recovery_extensions':2,'decision_time':'at base budget and each extension boundary','max_total_updates':sum(caps.values()),'stages':{name:{'base_updates':base[name],'max_updates':caps[name],'extension_updates':10000 if name=='guidance32' else 5000,'target_parent_gap':gaps[name]} for name in base}}
c['storage']={'keep_all_existing_results':True,'retain_each_stage':'selected EMA, full best raw/EMA/optimizer/RNG, metrics/features/preview, source+environment snapshot','retain_active_latest':True,'prune_only':'previous newly-generated latest.pt after next stage latest is durable; never best_resume.pt','legacy_best':'previous120K best has EMA only; do not fabricate optimizer; preserve transition full checkpoint','minimum_free_gib_before_write':3,'on_insufficient_space':'stop preserving all committed checkpoints'}
NEW.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
s=(R/'Experiment/student90_restart17.py').read_text();s=s.replace('exp-20261020-student90-cfg25-restart17.json','exp-20261021-student90-adaptive.json').replace('Experiment/student90_restart17.log','Experiment/student90_adaptive.log').replace('증류612K update 후','기본 증류425K(자동 연장 상한587.5K) update 후');s=s.replace('from student90_model import Student90,export','from student90_model import Student90,export\nfrom adaptive_budget_policy import decide\nimport adaptive_reproducibility as repro');needle='def write(p,x):';pos=s.index(needle);s=s[:pos]+"POLICY=C['adaptive_policy'];POLICY_SHA=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest();REVISION=None\n\n"+s[pos:]
start=s.index('def resume_config_allowed(');end=s.index('def save_selected(',start)
function='''def resume_config_allowed(saved_sha):
 current_sha=shared.sha(CONFIG_PATH)
 if saved_sha==current_sha:return True
 migration=read(OUT/'adaptive_migration.json',{})
 if saved_sha!=migration.get('old_sha256') or current_sha!=migration.get('new_sha256'):return False
 snapshot=Path(migration['old_config_snapshot']);assert shared.sha(snapshot)==saved_sha
 def normalized(value):
  value=copy.deepcopy(value)
  for key in ['total_distillation_updates','budget_revision','adaptive_policy','storage']:value.pop(key,None)
  for stage in value['stages']:stage.pop('updates',None)
  return value
 assert normalized(read(snapshot))==normalized(C),'Training settings changed outside authorized budget/policy/storage migration'
 assert C['total_distillation_updates']==425000 and sum(x['updates'] for x in C['stages'])==425000
 assert POLICY['max_total_updates']==587500
 return True

'''
s=s[:start]+function+s[end:]
# Wrap evaluations with a content fingerprint, preventing reuse for a different model after a restart.
s=s.replace('def evaluate_v(', 'def _evaluate_v_original(',1);pos=s.index('def target_for(')
wrapper='''def evaluate_v(backbone,times,clip,folder,retain_images=False,student_cfg=None):
 digest=repro.weight_hash(backbone);cached=read(folder/'metrics.json')
 if cached and cached.get('weights_sha256'):
  assert cached['weights_sha256']==digest,'Evaluation artifact belongs to different weights'
  assert cached['nfe']==len(times) and cached['clip_denoised']==clip
 elif cached:
  # Preserve legacy metrics and regenerate an explicitly fingerprinted evaluation if revisited.
  write(folder/'metrics_legacy_before_fingerprint.json',cached);(folder/'metrics.json').unlink()
 metrics=_evaluate_v_original(backbone,times,clip,folder,retain_images,student_cfg)
 metrics.update(weights_sha256=digest,policy_sha256=POLICY_SHA,source_revision=REVISION['revision'] if REVISION else None)
 write(folder/'metrics.json',metrics);return metrics

'''
s=s[:pos]+wrapper+s[pos:]
start=s.index('def train_stage(');end=s.index('def export_and_verify(',start)
train='''def train_stage(spec,parent,micro):
 name=spec['name'];folder=OUT/name;folder.mkdir(exist_ok=True);result_path=folder/'result.json'
 if result_path.exists():return read(result_path)
 S['current_stage']=name;S['phase']='distillation';seed();teacher,parent_times,teacher_clip=load_teacher(parent,name=='guidance32');guided=name=='guidance32';times=torch.tensor(C['nested_sampling_grids'][str(spec['sampling_steps'])],device='cuda');model=shared.model_from_config(student_config).cuda().train()
 if not guided:model.load_state_dict(teacher.state_dict(),strict=True)
 ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);assert sum(p.numel() for group in optimizer.param_groups for p in group['params'])==89577673;ds=get_dataset();updates=epoch=cursor=seen=0;elapsed=0.;evals=[];best=None;latest=folder/'latest.pt';selected=folder/'selected.pt';target_budget=spec['updates'];decisions=[]
 parent_metrics=read(OUT/'teacher32_anchor/metrics.json') if guided else read(Path(parent).parent/'result.json')['best']['metrics'];parent_hash=shared.sha(Path(parent))
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=False);assert st['parent']==str(parent) and st['stage']==name and resume_config_allowed(st['config_sha256']);model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);optimizer.load_state_dict(st['optimizer']);updates=st['update'];epoch=st['epoch'];cursor=st['cursor'];seen=st['images_seen'];elapsed=st['training_seconds'];best=st['best'];evals=st['evaluations'];target_budget=max(spec['updates'],st.get('adaptive_target_budget',spec['updates']));decisions=st.get('adaptive_decisions',[]);restore_rng(st['rng']);del st
 if not guided and not evals:
  initial=evaluate_v(ema,times,False,folder/'eval_000000');evals=[{'update':0,**initial}];best={'update':0,'metrics':initial};save_selected(selected,ema,times,False,name,0,initial)
 log_event(name,status='training',update=updates,budget=target_budget,base_budget=spec['updates'],max_budget=POLICY['stages'][name]['max_updates'],evaluations=evals,best_metrics=(best or {}).get('metrics',{}));write(folder/'config.json',{'spec':spec,'parent':str(parent),'parent_sha256':parent_hash,'microbatch':micro,'effective_batch':64,'student_config':student_config,'global_config_sha256':shared.sha(CONFIG_PATH),'adaptive_policy':POLICY['stages'][name],'policy_sha256':POLICY_SHA,'source_revision':REVISION})
 def save_resume(destination=latest,role='latest'):
  save_torch(destination,{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'update':updates,'epoch':epoch,'cursor':cursor,'images_seen':seen,'training_seconds':elapsed,'rng':rng_state(),'parent':str(parent),'parent_sha256':parent_hash,'stage':name,'config_sha256':shared.sha(CONFIG_PATH),'global_config':C,'policy_sha256':POLICY_SHA,'adaptive_target_budget':target_budget,'adaptive_decisions':decisions,'source_revision':REVISION,'best':best,'evaluations':evals,'student_config':student_config,'sampling_times':times.cpu(),'clip_denoised':guided,'train_manifest_sha256':C['training']['train_manifest_sha256']})
  if role!='latest':repro.register_checkpoint(folder,destination,role,updates,parent,shared.sha(CONFIG_PATH),POLICY_SHA,REVISION['revision'])
  if role=='latest':
   previous=OUT/'active_resume.json';old=read(previous,{})
   if old.get('path') and old['path']!=str(latest):
    old_path=Path(old['path'])
    if old_path.parent.parent==OUT and old_path.name=='latest.pt':old_path.unlink(missing_ok=True)
   write(previous,{'path':str(latest),'stage':name,'update':updates,'policy_sha256':POLICY_SHA})
 def assess():
  nonlocal target_budget
  decision=decide(name,updates,evals,parent_metrics['official_fid'],POLICY,decisions);decisions.append(decision);repro.log_decision(OUT/'adaptive_decisions.jsonl',decision);write(folder/'adaptive_state.json',{'target_budget':decision['next_budget'],'decisions':decisions,'best':best});target_budget=decision['next_budget'];log_event(name,adaptive_decision=decision,budget=target_budget)
  return decision
 # The migration boundary gets an exact evaluation and full state, rather than labelling the legacy120K EMA as a full optimizer snapshot.
 if not (folder/'adaptive_start.json').exists():
  metrics=evaluate_v(ema,times,guided,folder/f'eval_{updates:06d}');
  if not any(item['update']==updates for item in evals):evals.append({'update':updates,**metrics})
  improved=best is None or metrics['official_fid']<best['metrics']['official_fid']
  if improved:best={'update':updates,'metrics':metrics};save_selected(selected,ema,times,guided,name,updates,metrics);save_resume(folder/'best_resume.pt','best_resume')
  elif updates==0:save_resume(folder/'best_resume.pt','best_resume')
  repro.register_checkpoint(folder,selected,'best_ema',best['update'],parent,shared.sha(CONFIG_PATH),POLICY_SHA,REVISION['revision']);save_resume();write(folder/'adaptive_start.json',{'update':updates,'best':best,'legacy_best_full_state_available':(folder/'best_resume.pt').exists(),'policy_sha256':POLICY_SHA});log_event(name,evaluations=evals,best_metrics=best['metrics'])
 while updates<target_budget:
  loader=epoch_loader(ds,64,0,42,epoch)
  for batch_index,(images,labels) in enumerate(loader):
   if batch_index<cursor:continue
   start=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);total_loss=0.;batch=len(labels)
   for pos in range(0,batch,micro):
    x0=images[pos:pos+micro].cuda();y=labels[pos:pos+micro].cuda();idx=torch.randint(len(times),(len(y),),device='cuda');t=times[idx];a,s=coefficients(alpha,t);x=a*x0+s*torch.randn_like(x0);top=t==times[0];x[top]=torch.randn_like(x[top]);target=target_for(teacher,x,t,y,guided,parent_times,idx,teacher_clip)
    with amp():pred=model(x,t,y);loss=velocity_loss(pred,target,a,s)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation loss')
    (loss*len(y)/batch).backward();total_loss+=float(loss.detach())*len(y)/batch
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));optimizer.step();update_ema(ema,model,.999);torch.cuda.synchronize();elapsed+=time.monotonic()-start;updates+=1;cursor=batch_index+1;seen+=batch;peak=torch.cuda.max_memory_allocated();assert peak<20_000_000_000
   if updates==1 or updates%20==0:
    rec={'update':updates,'loss':total_loss,'gradient_norm':norm,'lr':2e-4,'training_seconds':elapsed,'images_seen':seen,'actual_batch_size':batch,'microbatch':micro,'activation_checkpoint_resolutions':student_config['activation_checkpoint_resolutions'],'peak_vram_bytes':peak,'target_budget':target_budget}
    with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\\n')
    log_event(name,update=updates,last_log=rec);print(name,rec,flush=True)
   due=updates%spec['evaluation_interval']==0 or updates==target_budget;improved=False
   if due:
    log_event(name,status='evaluating');metrics=evaluate_v(ema,times,guided,folder/f'eval_{updates:06d}');evals.append({'update':updates,**metrics});improved=best is None or metrics['official_fid']<best['metrics']['official_fid']
    if improved:best={'update':updates,'metrics':metrics};save_selected(selected,ema,times,guided,name,updates,metrics)
    if updates>=target_budget:assess()
    if improved:save_resume(folder/'best_resume.pt','best_resume');repro.register_checkpoint(folder,selected,'best_ema',best['update'],parent,shared.sha(CONFIG_PATH),POLICY_SHA,REVISION['revision'])
    log_event(name,status='training',evaluations=evals,best_metrics=best['metrics'],budget=target_budget)
   if updates%500==0 or due:save_resume()
   if updates>=target_budget:break
  else:epoch+=1;cursor=0
 result={'stage':name,'selected_checkpoint':str(selected),'selected_full_resume':str(folder/'best_resume.pt') if (folder/'best_resume.pt').exists() else None,'best':best,'evaluations':evals,'training_updates':updates,'base_updates':spec['updates'],'max_updates':POLICY['stages'][name]['max_updates'],'adaptive_decisions':decisions,'parent_metrics':parent_metrics,'policy_sha256':POLICY_SHA,'source_revision':REVISION};write(result_path,result);log_event(name,status='completed',best_metrics=best['metrics']);del model,ema,teacher,optimizer;gc.collect();torch.cuda.empty_cache();return result

'''
s=s[:start]+train+s[end:]
# Retain a complete winning GAN state as well as the final selected inference model.
gstart=s.index('def gan_finetune(');gend=s.index('def main():',gstart);gan=s[gstart:gend];a=gan.index(' def save_resume():');b=gan.index(' def d_step(',a)
replacement=''' def save_resume(destination=latest):
  save_torch(destination,{'g':g.state_dict(),'ema':ema.state_dict(),'d':d.state_dict(),'og':og.state_dict(),'od':od.state_dict(),'update':steps,'warm':warm,'training_seconds':elapsed,'best':best,'evaluations':evals,'rng':rng_state(),'source':source,'config_sha256':shared.sha(CONFIG_PATH),'global_config':C,'source_revision':REVISION,'policy_sha256':POLICY_SHA})
  if destination!=latest:repro.register_checkpoint(folder,destination,'best_resume',steps,source,shared.sha(CONFIG_PATH),POLICY_SHA,REVISION['revision'])
  if destination==latest:
   previous=OUT/'active_resume.json';old=read(previous,{})
   if old.get('path') and old['path']!=str(latest):
    p=Path(old['path'])
    if p.parent.parent==OUT and p.name=='latest.pt':p.unlink(missing_ok=True)
   write(previous,{'path':str(latest)})
'''
gan=gan[:a]+replacement+gan[b:];gan=gan.replace("best={'update':steps,'metrics':metrics,'checkpoint':str(selected)}", "best={'update':steps,'metrics':metrics,'checkpoint':str(selected)};save_resume(folder/'best_resume.pt')");s=s[:gstart]+gan+s[gend:]
s=s.replace('global alpha,student_config\n OUT.mkdir', 'global alpha,student_config,REVISION\n OUT.mkdir');needle="S.update(status='running',phase='numeric_checks');";replacement="REVISION=repro.capture_revision(ROOT,OUT,CONFIG_PATH,Path(__file__));write(OUT/'active_source_revision.json',REVISION);"+needle;s=s.replace(needle,replacement)
# Show the most recent automatic decision beside the live training log.
needle="if S.get('error'):lines+=";s=s.replace(needle,"if j.get('adaptive_decision'):lines+=['','## 최근 자동 판단','','```json',json.dumps(j['adaptive_decision'],indent=2,ensure_ascii=False),'```']\n  "+needle)
ast.parse(s);(R/'Experiment/student90_adaptive.py').write_text(s);print('Adaptive runner compiled; base',c['total_distillation_updates'],'max',c['adaptive_policy']['max_total_updates'])
