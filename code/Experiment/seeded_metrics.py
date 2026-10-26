def seeded_metrics(model,cfg,folder,scale,steps,weights,job=None,seed_base=42):
 global feature_net
 folder.mkdir(parents=True,exist_ok=True)
 if (folder/'metrics.json').exists():return read(folder/'metrics.json')
 images=folder/'generated';images.mkdir(exist_ok=True);schedule=Diffusion(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'],torch.device('cuda'));cats=torch.arange(151).repeat_interleave(20);counter=[0]
 def hook(*_):counter[0]+=1
 h=model.register_forward_hook(hook);was_training=model.training;model.eval();nfe=steps*(2 if scale not in (0,1) else 1)
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for bi,pos in enumerate(range(0,3020,16)):
    y=cats[pos:pos+16].cuda();before=counter[0];x,reported=schedule.sample(model,y,64,3,steps,scale,seed_base+bi,sampler='ddim',clip_denoised=True);assert reported==nfe==counter[0]-before and torch.isfinite(x).all()
    for i,im in enumerate(x):
     idx=pos+i;to_pil_image(((im.float().cpu()+1)/2).clamp(0,1)).save(images/f'{int(cats[idx]):04d}_{idx%20:02d}.png')
    if job and bi%40==0:job_update(job,generated_count=pos+len(y))
 finally:h.remove();model.train(was_training)
 if feature_net is None:feature_net=features.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
 feats=fid.get_files_features([str(p) for p in sorted(images.glob('*.png'))],model=feature_net,num_workers=0,batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False);np.save(folder/'features.npy',feats);mean=feats.mean(0);cov=np.cov(feats,rowvar=False)
 m={'official_fid':float(fid.frechet_distance(mean,cov,rm,rc)),'valid_fid':float(fid.frechet_distance(mean,cov,vm,vc)),**shared.common.pr(feats,ref),'cfg':scale,'sampling_steps':steps,'nfe':nfe,'weights':weights,'generated_count':3020,'official_reference_count':3020,'valid_reference_count':755,'seed_base':seed_base,'batch_size':16,'precision':'bf16','parameters':sum(p.numel() for p in model.parameters()),'submission_eligible':False};write(folder/'metrics.json',m);print('EVAL',folder,m,flush=True);return m
