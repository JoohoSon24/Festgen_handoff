def train_augmented(key,kind):
 global champion
 STATE['current_job']=key;job_update(key,status='preparing');folder=OUT/key;folder.mkdir(parents=True,exist_ok=True);ds=get_cache();st=torch.load(PARENTS[selected['parent_k']],map_location='cpu',weights_only=False);cfg=st['config'].copy();cfg['learning_rate']=5e-5;cfg['data_root']=str(DATA);cfg['split_dir']=str(DATA);manifest=DATA/'train.txt';parent_manifest_hash=st['train_paths_sha256'];assert st['class_names']==[name for name,c in sorted(read(DATA/'category_to_id.json').items(),key=lambda x:x[1])];assert cfg['batch_size']==64 and cfg['gradient_accumulation']==1
 model=model_from_config(cfg).cuda();model.load_state_dict(st['model']);ema=copy.deepcopy(model).requires_grad_(False).eval();ema.load_state_dict(st['ema']);extra={k:copy.deepcopy(model).requires_grad_(False).eval() for k in st.get('ema_tracks',{})}
 for k,v in extra.items():v.load_state_dict(st['ema_tracks'][k])
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);optimizer.load_state_dict(st['optimizer']);
 for group in optimizer.param_groups:group['lr']=5e-5
 assert sum(p.numel() for a in optimizer.param_groups for p in a['params'])==48883107
 track=ema if selected['weights']=='ema' else extra[selected['weights'].split(':',1)[1]];schedule=Diffusion(1000,cfg['beta_start'],cfg['beta_end'],torch.device('cuda'));initial_step=st['step'];epoch=st['epoch'];cursor=0;seen=st.get('images_seen',initial_step*64);updates=0
 configuration={'parent_checkpoint':str(PARENTS[selected['parent_k']]),'parent_id':'teacher'+str(initial_step),'hypothesis':'class balancing + offline augmentation + conservative LR; exploratory combined intervention','objective':kind,'gamma':5 if kind=='min_snr' else None,'updates':30000,'evaluate_updates':[5000,10000,20000,30000],'learning_rate':cfg['learning_rate'],'batch_size':64,'microbatch_size':64,'gradient_accumulation':1,'class_dropout':cfg['class_dropout'],'primary_ema_decay':cfg['ema_decay'],'ema_tracks':list(extra),'selected_track':selected['weights'],'sampling_steps':selected['sampling_steps'],'cfg':selected['cfg'],'train_manifest':str(manifest),'train_manifest_sha256':hashlib.sha256('\n'.join(ds.paths).encode()).hexdigest(),'parent_train_manifest_sha256':parent_manifest_hash,'augmentation':'offline balanced178/class, flip and rotation+-10, no color transform','trainable_parameters':48883107,'all_optimized_parameters':48883107,'raw_config':cfg};write(folder/'config.json',configuration)
 # GPU infrastructure and gradient check; no optimizer update before restoring state/RNG.
 sample=torch.stack([ds[i][0] for i in range(4)]).cuda();labels=ds.labels[:4].cuda();t=torch.tensor([50,350,650,950],device='cuda');noise=torch.randn_like(sample);a=schedule.alpha_bar[t,None,None,None]
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(a.sqrt()*sample+(1-a).sqrt()*noise,t,labels);per=(pred.float()-noise).square().flatten(1).mean(1);w=torch.minimum((a.flatten()/(1-a.flatten())),torch.full_like(t,5,dtype=torch.float32))/(a.flatten()/(1-a.flatten()));probe=(per*(w if kind=='min_snr' else 1)).mean()
 probe.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));assert math.isfinite(norm) and norm>0;optimizer.zero_grad(set_to_none=True);write(folder/'preflight.json',{'parameters':48883107,'gradient_norm':norm,'loss_finite':bool(torch.isfinite(probe)),'parent_step':initial_step,'train_manifest_verified':True,'actual_lrs':[x['lr'] for x in optimizer.param_groups]})
 torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);torch.cuda.set_rng_state_all(st['cuda_rng']);del st
 model.train();start=time.time();torch.cuda.reset_peak_memory_stats();job_update(key,status='training',update=0);evals=[];best={'id':selected['id'],'metrics':selected['metrics'],'parent_checkpoint':str(PARENTS[selected['parent_k']]),'weights':selected['weights']}
 def save_training(path):
  shared.atom_torch(path,{'model':model.state_dict(),'ema':ema.state_dict(),'ema_tracks':{k:v.state_dict() for k,v in extra.items()},'optimizer':optimizer.state_dict(),'step':initial_step+updates,'local_updates':updates,'epoch':epoch,'batch_in_epoch':cursor,'images_seen':seen,'config':cfg,'experiment':configuration,'class_names':[name for name,c in sorted(read(DATA/'category_to_id.json').items(),key=lambda x:x[1])],'train_paths_sha256':configuration['train_manifest_sha256'],'scaler':{},'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all()})
 while updates<30000:
  loader=epoch_loader(ds,64,0,cfg['seed'],epoch)
  for batch_index,(images,labels) in enumerate(loader):
   if batch_index<cursor:continue
   x=images.cuda(non_blocking=True);y=drop_classes(labels.cuda(non_blocking=True),151,cfg['class_dropout']);t=torch.randint(1000,(len(y),),device='cuda');noise=torch.randn_like(x);a=schedule.alpha_bar[t,None,None,None];optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pred=model(a.sqrt()*x+(1-a).sqrt()*noise,t,y);per=(pred.float()-noise).square().flatten(1).mean(1);snr=a.flatten()/(1-a.flatten());weights=torch.minimum(snr,torch.full_like(snr,5))/snr if kind=='min_snr' else torch.ones_like(snr);loss=(weights*per).mean();unweighted=per.mean()
   assert torch.isfinite(loss);loss.backward();gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True));optimizer.step();update_ema(ema,model,cfg['ema_decay'])
   for k,v in extra.items():update_ema(v,model,float(k))
   updates+=1;seen+=len(y);cursor=batch_index+1;peak=torch.cuda.max_memory_allocated();assert peak<20*1024**3
   if updates==1 or updates%20==0:
    record={'update':updates,'step':initial_step+updates,'loss':float(loss.detach()),'unweighted_epsilon_mse':float(unweighted.detach()),'mean_weight':float(weights.mean()),'gradient_norm':gradnorm,'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.time()-start,'peak_vram_bytes':peak,'images_seen':seen}
    with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    job_update(key,update=updates,last_train=record);print(key,record,flush=True)
   if updates in [5000,10000,20000,30000]:
    # Store full training state before evaluation to avoid evaluation consuming resume RNG.
    save_training(folder/f'update_{updates:06d}_training.pt');cp=folder/f'update_{updates:06d}_teacher.pt';shared.atom_torch(cp,{'state_dict':track.state_dict(),'config':cfg,'step':initial_step+updates,'local_updates':updates,'weights':selected['weights'],'parameter_count':48883107,'submission_eligible':False,'experiment':configuration})
    metrics=metrics_for_model(track,cfg,folder/f'eval_{updates:06d}',selected['cfg'],selected['sampling_steps'],selected['weights'],key);ev={'update':updates,**metrics};evals.append(ev)
    with lockstate:STATE['jobs'][key]['evaluations']=evals.copy()
    candidate={'id':key+'_update'+str(updates),'checkpoint':str(cp),'metrics':metrics,'parent_k':selected['parent_k'],'weights':selected['weights'],'cfg':selected['cfg'],'sampling_steps':selected['sampling_steps'],'objective':kind}
    if metrics['official_fid']<best['metrics']['official_fid']:best=candidate
    if metrics['official_fid']<champion['metrics']['official_fid']:champion=candidate;write(OUT/'best_teacher_candidate.json',champion)
    diagnostic(key+f'/diag_{updates:06d}',model,track,cfg,selected['weights']);report();model.train()
   if updates==30000:break
  else:epoch+=1;cursor=0
 write(folder/'result.json',{'best':best,'evaluations':evals,'changed_factor':'class balancing, offline spatial augmentation, and reduced LR','checkpoint_raw_resume_parent':str(PARENTS[selected['parent_k']])});job_update(key,status='completed',metrics=best['metrics']);del model,ema,extra,track,optimizer;torch.cuda.empty_cache();return best
