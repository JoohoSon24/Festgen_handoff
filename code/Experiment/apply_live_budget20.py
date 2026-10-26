"""Watch atomic resume replacement, pause safely, apply budget-only +20%, and exec trainer."""
import os,json,time,copy,hashlib,signal,subprocess,fcntl,ast,sys
from pathlib import Path
R=Path('/root/Diffusion 3');OUT=R/'FID/student90_cfg25_restart17';CONFIG=R/'Experiment/configs/exp-20261020-student90-cfg25-restart17.json';SOURCE=R/'Experiment/student90_restart17.py';H=OUT/'budget_history';H.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def normalize(obj):
 obj=copy.deepcopy(obj);obj.pop('total_distillation_updates',None);obj.pop('budget_revision',None)
 for stage in obj['stages']:stage.pop('updates',None)
 return obj
old=read(CONFIG);assert old['total_distillation_updates']==510000;state=read(OUT/'status.json');assert state['status']=='running' and state['current_stage']=='guidance32';resume=Path(read(OUT/'active_resume.json')['path']);assert resume==OUT/'guidance32/latest.pt' and resume.is_file();initial_stamp=resume.stat().st_mtime_ns
write(OUT/'budget_extension_request.json',{'status':'waiting_for_next_atomic_resume','factor':1.2,'current_total':510000,'new_total':612000,'teacher_optimizer_rng_preserved':True,'requested_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())})
# Do not import or deserialize torch while waiting: mtime changes only after atomic checkpoint replacement.
for _ in range(400):
 if resume.stat().st_mtime_ns!=initial_stamp:break
 state=read(OUT/'status.json')
 if state['status']!='running':raise RuntimeError('Trainer stopped while waiting for checkpoint')
 time.sleep(.5)
else:raise RuntimeError('No new atomic checkpoint within200sec')
state=read(OUT/'status.json');pid=state['pid'];args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True);assert '/root/Diffusion 3/Experiment/student90_restart17.py' in args;os.kill(pid,signal.SIGTERM);state['status']='paused_for_budget_extension20';write(OUT/'status.json',state)
# Acquire the same GPU lock to wait until the old process has fully released it.
with (R/'Experiment/gt_gan.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime());snapshot=H/f'config_before_20pct_{stamp}.json';snapshot.write_bytes(CONFIG.read_bytes());old_sha=sha(snapshot);new=copy.deepcopy(old)
 for stage in new['stages']:
  assert stage['updates']*6%5==0;stage['updates']=stage['updates']*6//5
 new['total_distillation_updates']=sum(x['updates'] for x in new['stages']);assert new['total_distillation_updates']==612000;new['budget_revision']={'factor_over_previous':1.2,'previous_total':510000,'new_total':612000,'factor_over_initial300k':2.04,'authorized_by':'2026-10-21 user: 현재 예산에서20% 추가. Continue from current state; no reset.','gan_budget_unchanged':True};assert normalize(old)==normalize(new)
 write(CONFIG,new);new_sha=sha(CONFIG);migration={'old_sha256':old_sha,'new_sha256':new_sha,'old_config_snapshot':str(snapshot),'factor_numerator':6,'factor_denominator':5,'stage':state['current_stage'],'observed_update_at_pause':state['stages'][state['current_stage']]['update'],'resume_checkpoint':str(resume),'resume_file_unmodified':True,'old_budgets':{s['name']:s['updates'] for s in old['stages']},'new_budgets':{s['name']:s['updates'] for s in new['stages']},'authorized_by':new['budget_revision']['authorized_by']};path=OUT/'budget_migrations.json';records=read(path) if path.exists() else [];records.append(migration);write(path,records)
 source=SOURCE.read_text();function='''def resume_config_allowed(saved_sha):
 current_sha=shared.sha(CONFIG_PATH)
 if saved_sha==current_sha:return True
 for migration in read(OUT/'budget_migrations.json',[]):
  if saved_sha!=migration['old_sha256'] or current_sha!=migration['new_sha256']:continue
  snapshot=Path(migration['old_config_snapshot'])
  assert shared.sha(snapshot)==saved_sha
  previous=read(snapshot)
  def normalize(value):
   value=copy.deepcopy(value);value.pop('total_distillation_updates',None);value.pop('budget_revision',None)
   for stage in value['stages']:stage.pop('updates',None)
   return value
  assert normalize(previous)==normalize(C),'Budget migration changed other settings'
  num=migration['factor_numerator'];den=migration['factor_denominator']
  assert all(n['updates']*den==o['updates']*num for o,n in zip(previous['stages'],C['stages']))
  assert C['total_distillation_updates']==sum(stage['updates'] for stage in C['stages'])
  return True
 return False

'''
 assert 'def resume_config_allowed(' not in source
 pos=source.index('def save_selected(');source=source[:pos]+function+source[pos:];source=source.replace("st['config_sha256']==shared.sha(CONFIG_PATH)","resume_config_allowed(st['config_sha256'])");source=source.replace('증류510K update 후','증류612K update 후');ast.parse(source);SOURCE.write_text(source)
 request=read(OUT/'budget_extension_request.json');request.update(status='configured_resuming',migration=migration);write(OUT/'budget_extension_request.json',request)
 lines=['# 현재 증류 예산20% 추가 — 상태 유지','', '2026-10-21 사용자 요청: 현재510K 예산에서20% 추가. 총612K update이며최초300K의2.04배다. 재초기화하지않고현재checkpoint의model/EMA/optimizer/epoch cursor/RNG에서이어학습한다.','', '| 단계 | 이전 | 변경 |','|---|---:|---:|']
 for previous,current in zip(old['stages'],new['stages']):lines.append(f"| {current['name']} | {previous['updates']:,} | {current['updates']:,} |")
 lines+=['| 합계 |510,000|612,000|','','모델·Teacher CFG2.5·objective·LR2e-4·유효batch64·EMA·sampling grid는그대로다. GAN fine-tune은NFE1/NFE4 각1000 update를유지한다. 이전config SHA와원본설정을보존하고budget 외설정동일성을재개시검증한다. 기존checkpoint본문을변경하지않는다.','', '진행문서:[STUDENT90_PROGRESS.md](STUDENT90_PROGRESS.md). 상세변경기록은FID/student90_cfg25_restart17/budget_migrations.json이다.']
 (R/'docs/STUDENT90_BUDGET20.md').write_text('\n'.join(lines)+'\n');print('Budget updated to612000; resume checkpoint untouched; exec trainer',flush=True)
# No optimization model was created by this controller; checkpoint state remains unchanged.
handle=open(R/'Experiment/student90_restart17.log','a');os.dup2(handle.fileno(),1);os.dup2(handle.fileno(),2);os.execv('/root/Diffusion/code/workspace_distillation/.venv/bin/python',['/root/Diffusion/code/workspace_distillation/.venv/bin/python','-X','utf8','-u',str(SOURCE)])
