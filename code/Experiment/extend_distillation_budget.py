"""Authorized +25% budget migration; preserve weights/optimizer/RNG and strict non-budget equality."""
import json,hashlib,time,copy,ast
from pathlib import Path
R=Path('/root/Diffusion 3');OUT=R/'FID/student90_cfg25';P=R/'Experiment/configs/exp-20261020-student90-cfg25-review.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def normalized(c):
 c=copy.deepcopy(c);c.pop('total_distillation_updates',None);c.pop('budget_revision',None)
 for stage in c['stages']:stage.pop('updates',None)
 return c
for _ in range(240):
 status=json.loads((OUT/'status.json').read_text())
 if status['status']=='paused_for_budget_extension':break
 if status['status']=='failed':raise RuntimeError('Trainer failed before migration')
 time.sleep(1)
else:raise RuntimeError('Waiting for explicit pause timed out')
old=json.loads(P.read_text());assert old['total_distillation_updates']==300000;new=copy.deepcopy(old)
for stage in new['stages']:stage['updates']=stage['updates']*5//4
new['total_distillation_updates']=sum(stage['updates'] for stage in new['stages']);assert new['total_distillation_updates']==375000;new['budget_revision']={'factor':1.25,'authorized_by':'User asked each distillation stage about1/4 longer, 2026-10-20','gan_budget_unchanged':True};assert normalized(old)==normalized(new)
history=OUT/'budget_history';history.mkdir(exist_ok=True);snapshot=history/'config_before_25pct.json';snapshot.write_bytes(P.read_bytes());old_hash=sha(snapshot);write(P,new);new_hash=sha(P);oldbudgets={s['name']:s['updates'] for s in old['stages']};newbudgets={s['name']:s['updates'] for s in new['stages']}
record={'old_sha256':old_hash,'new_sha256':new_hash,'old_config_snapshot':str(snapshot),'old_stage_budgets':oldbudgets,'new_stage_budgets':newbudgets,'old_total':300000,'new_total':375000,'protected_invariants':'model,teacher,objective,LR,batch,RNG,optimizer and GAN budgets unchanged','checkpoint_stage':status['current_stage'],'checkpoint_update_at_pause':status['stages'][status['current_stage']]['update'],'authorized_by':new['budget_revision']['authorized_by']};migrations=OUT/'budget_migrations.json';existing=json.loads(migrations.read_text()) if migrations.exists() else [];existing.append(record);write(migrations,existing)
source=R/'Experiment/student90_pipeline.py';s=source.read_text()
function='''def resume_config_allowed(saved_sha):
 current_sha=shared.sha(CONFIG_PATH)
 if saved_sha==current_sha:return True
 for migration in read(OUT/'budget_migrations.json',[]):
  if migration['old_sha256']!=saved_sha or migration['new_sha256']!=current_sha:continue
  snapshot=Path(migration['old_config_snapshot'])
  assert shared.sha(snapshot)==saved_sha
  previous=read(snapshot);current=copy.deepcopy(C)
  def normalize(value):
   value=copy.deepcopy(value);value.pop('total_distillation_updates',None);value.pop('budget_revision',None)
   for stage in value['stages']:stage.pop('updates',None)
   return value
  assert normalize(previous)==normalize(current),'Non-budget setting changed in budget migration'
  assert [x['name'] for x in previous['stages']]==[x['name'] for x in current['stages']]
  assert all(new['updates']==old['updates']*5//4 for old,new in zip(previous['stages'],current['stages']))
  assert current['total_distillation_updates']==sum(x['updates'] for x in current['stages'])
  return True
 return False

'''
needle='def save_selected(';idx=s.index(needle);s=s[:idx]+function+s[idx:];s=s.replace("st['config_sha256']==shared.sha(CONFIG_PATH)","resume_config_allowed(st['config_sha256'])");s=s.replace('증류300K update 후','증류375K update 후');source.write_text(s);ast.parse(s)
lines=['# 증류 학습량25% 연장','', '2026-10-20 사용자의 요청에 따라 현재 CFG 내재화와 모든 progressive 단계의 optimizer update를1.25배로 늘렸다. 기존 가중치·EMA·optimizer·RNG와 최선 평가 이력을 보존해 이어 학습한다.','', '| 단계 | 이전 update | 변경 update |','|---|---:|---:|']
for name,oldbudget in oldbudgets.items():lines.append(f'| {name} | {oldbudget:,} | {newbudgets[name]:,} |')
lines+=['| 합계 |300,000|375,000|','','LR2e-4 constant, 유효 batch64, microbatch64, EMA0.999, Teacher CFG2.5, loss·sampling grid·증강 미사용 조건은 그대로다. GAN fine-tune은NFE1/NFE4 각1K를 유지한다.','', '기존 config를 budget_history에 보관하고 checkpoint의 이전 config SHA를 기록했다. 재개 시 budget을 제외한 모든 설정이 일치하는지 검사한다. checkpoint 본문을 수정하거나 초기화하지 않는다. 다음 resume 저장부터 새 config SHA를 사용한다.','', '기존 75K/30K 등의 문서는 당시 설정이며 이 문서가 학습량을 대체한다. 평가 간격은 기존대로guidance10K, progressive5K 및각단계최종update다. 추가되는순수학습시간은현재실측기준약8시간이며평가포함약8~10시간 증가로예상한다.']
(R/'docs/STUDENT90_BUDGET_EXTENSION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
