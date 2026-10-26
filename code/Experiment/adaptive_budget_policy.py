"""Deterministic, bounded decisions based on already measured paired-seed Official FID."""
import math,hashlib,json

def decide(stage,update,evaluations,parent_fid,policy,history=()):
 cfg=policy['stages'][stage];base=int(cfg['base_updates']);cap=int(cfg['max_updates']);interval=int(cfg['extension_updates']);rows=sorted(evaluations,key=lambda x:x['update']);assert rows and all(math.isfinite(float(x['official_fid'])) for x in rows)
 best=min(float(x['official_fid']) for x in rows);cutoff=update-policy['trend_intervals']*interval;earlier=[float(x['official_fid']) for x in rows if x['update']<=cutoff];gain=min(earlier)-best if earlier else 0.;gap=best-float(parent_fid);recovery_count=sum(x.get('reason')=='gap_recovery' for x in history);target_met=gap<=cfg['target_parent_gap'];next_budget=update
 if update>=cap:action='finish';reason='max_budget_reached'
 elif gain>=policy['min_meaningful_fid_improvement']:action='extend';reason='still_improving';next_budget=min(cap,update+interval)
 elif not target_met and recovery_count<policy['max_gap_recovery_extensions']:action='extend';reason='gap_recovery';next_budget=min(cap,update+interval)
 else:action='finish';reason='plateau_target_met' if target_met else 'plateau_target_unmet'
 payload={'policy_version':policy['version'],'stage':stage,'update':update,'base_updates':base,'max_updates':cap,'action':action,'reason':reason,'next_budget':next_budget,'parent_official_fid':float(parent_fid),'best_official_fid':best,'parent_gap':gap,'target_parent_gap':cfg['target_parent_gap'],'target_met':target_met,'recent_best_fid_improvement':gain,'trend_start_update':cutoff,'recovery_extensions_used':recovery_count,'selection':'best EMA among measured updates; no automatic replacement of fixed baselines'}
 payload['decision_id']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();return payload
