"""Upgrade at an atomically committed resume checkpoint; never reset trained state."""
import os,time,json,copy,hashlib,signal,subprocess,fcntl,shutil,sys
from pathlib import Path
R=Path('/root/Diffusion 3');OUT=R/'FID/student90_cfg25_restart17';OLD=R/'Experiment/configs/exp-20261020-student90-cfg25-restart17.json';NEW=R/'Experiment/configs/exp-20261021-student90-adaptive.json';SCRIPT=R/'Experiment/student90_adaptive.py'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def normalized(c):
 c=copy.deepcopy(c)
 for key in ['total_distillation_updates','budget_revision','adaptive_policy','storage']:c.pop(key,None)
 for stage in c['stages']:stage.pop('updates',None)
 return c
old=read(OLD);new=read(NEW);assert old['total_distillation_updates']==612000 and new['total_distillation_updates']==425000 and new['adaptive_policy']['max_total_updates']==587500;assert normalized(old)==normalized(new)
s=read(OUT/'status.json');assert s['status']=='running' and s['current_stage']=='guidance32';resume=Path(read(OUT/'active_resume.json')['path']);initial=resume.stat().st_mtime_ns;write(OUT/'adaptive_upgrade_request.json',{'status':'waiting_for_atomic_checkpoint','base_budget':425000,'max_budget':587500})
for _ in range(440):
 if resume.stat().st_mtime_ns!=initial:break
 state=read(OUT/'status.json')
 if state['status']!='running':raise RuntimeError('Trainer stopped before upgrade')
 time.sleep(.5)
else:raise RuntimeError('No atomic resume replacement before220sec')
state=read(OUT/'status.json');pid=state['pid'];args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True);assert '/root/Diffusion 3/Experiment/student90_restart17.py' in args;os.kill(pid,signal.SIGTERM);state['status']='paused_for_adaptive_upgrade';write(OUT/'status.json',state)
with (R/'Experiment/gt_gan.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 provenance=OUT/'provenance';provenance.mkdir(exist_ok=True);snapshot=provenance/'config_before_adaptive.json';snapshot.write_bytes(OLD.read_bytes());legacy=provenance/'legacy_source';legacy.mkdir(exist_ok=True)
 for name in ['student90_restart17.py','student90_model.py','teacher_core.py','train_gt_gan.py','run_cfg.py']:shutil.copy2(R/'Experiment'/name,legacy/name)
 transition=provenance/'transition_resume.pt';assert not transition.exists();os.link(resume,transition)
 selected=OUT/'guidance32/selected.pt';shutil.copy2(selected,provenance/'legacy_selected_ema.pt')
 record={'old_sha256':sha(snapshot),'new_sha256':sha(NEW),'old_config_snapshot':str(snapshot),'transition_resume':str(transition),'transition_resume_sha256':sha(transition),'legacy_best_ema':str(provenance/'legacy_selected_ema.pt'),'legacy_best_ema_sha256':sha(provenance/'legacy_selected_ema.pt'),'stage_at_upgrade':state['current_stage'],'observed_update_at_pause':state['stages'][state['current_stage']]['update'],'old_total':612000,'new_base_total':425000,'new_max_total':587500,'state_preserved':['raw_model','ema','optimizer','epoch_cursor','images_seen','torch_rng','cuda_rng','python_rng','numpy_rng'],'legacy_best_optimizer':'not available for historical120K best; full transition state preserved instead','authorized_by':'User approved proposed budgets plus periodic automatic FID review and reproducible saving','created_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())};write(OUT/'adaptive_migration.json',record)
 active=read(R/'FID/ACTIVE_STUDENT_EXPERIMENT.json');active.update(script=str(SCRIPT),config=str(NEW),total_distillation_updates=425000,max_distillation_updates=587500,mode='adaptive_fid_review',latest_budget_document=str(R/'docs/STUDENT90_ADAPTIVE_PROTOCOL.md'));write(R/'FID/ACTIVE_STUDENT_EXPERIMENT.json',active);write(OUT/'adaptive_upgrade_request.json',{'status':'configured_resuming','migration':record});print('Adaptive upgrade ready; transition full state protected; exec trainer',flush=True)
handle=open(R/'Experiment/student90_adaptive.log','a');os.dup2(handle.fileno(),1);os.dup2(handle.fileno(),2);os.execv('/root/Diffusion/code/workspace_distillation/.venv/bin/python',['/root/Diffusion/code/workspace_distillation/.venv/bin/python','-X','utf8','-u',str(SCRIPT)])
