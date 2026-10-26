from adaptive_budget_policy import decide
P={'version':'test','trend_intervals':2,'min_meaningful_fid_improvement':.2,'max_gap_recovery_extensions':2,'stages':{'n4':{'base_updates':40000,'max_updates':60000,'extension_updates':5000,'target_parent_gap':2}}}
def row(step,fid):return {'update':step,'official_fid':fid}
a=[row(30000,32),row(35000,31.7),row(40000,31.4)];assert decide('n4',40000,a,30,P)['reason']=='still_improving'
a=[row(30000,31),row(35000,31.02),row(40000,31.01)];assert decide('n4',40000,a,30,P)['reason']=='plateau_target_met'
a=[row(30000,34),row(35000,34.01),row(40000,34.02)];d=decide('n4',40000,a,30,P);assert d['reason']=='gap_recovery';assert d['next_budget']==45000
assert decide('n4',40000,a,30,P,[d,d])['reason']=='plateau_target_unmet'
assert decide('n4',60000,a,30,P)['reason']=='max_budget_reached'
assert decide('n4',40000,a,30,P)==decide('n4',40000,a,30,P)
print('Adaptive policy checks pass: improvement, plateau, recovery, cap, deterministic replay')
