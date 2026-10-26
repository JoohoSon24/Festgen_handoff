"""90M-compatible derived model; protected base Model and original repo stay unchanged."""
import contextlib
import torch
import teacher_core as shared
from model import Model as ProtectedModel
from teacher.diffusion import Diffusion
from distillation.core import coefficients,ddim_step,velocity_x0

class Student90(ProtectedModel):
 def __init__(self,config,times,clip=False):
  super().__init__(shared.model_from_config(config));self.config=config.copy();self.clip_denoised=bool(clip)
  self.register_buffer('alpha_bar',Diffusion(1000,config['beta_start'],config['beta_end'],torch.device('cpu')).alpha_bar)
  self.register_buffer('sampling_times',torch.as_tensor(times,dtype=torch.long).clone());self.register_buffer('sampling_seed',torch.tensor(42));self._batch_index=0
 def rollout(self,z,labels,grad=False):
  # Network checkpointing is enabled within ClassConditionalUNet.run_block while training.
  x=z
  for i,time in enumerate(self.sampling_times.tolist()):
   t=torch.full_like(labels,time);nt=torch.full_like(labels,int(self.sampling_times[i+1]) if i+1<len(self.sampling_times) else -1)
   a,s=coefficients(self.alpha_bar,t);an,sn=coefficients(self.alpha_bar,nt);x0=velocity_x0(self.backbone,x,t,labels,self.alpha_bar)
   if self.clip_denoised:x0=x0.clamp(-1,1)
   x=ddim_step(x,x0,a,s,an,sn)
  return x.clamp(-1,1)
 @torch.inference_mode()
 def sample(self,shape,*,device='cuda',category=None,**kwargs):
  assert tuple(shape[1:])==(3,64,64) and category is not None and len(category)==shape[0]
  seed=int(kwargs.get('seed',int(self.sampling_seed)+self._batch_index))
  if 'seed' not in kwargs:self._batch_index+=1
  labels=category.to(device=device,dtype=torch.long);assert bool(((labels>=0)&(labels<151)).all())
  self.to(device).eval();z=torch.randn(shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
  amp=torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else contextlib.nullcontext()
  with amp:return self.rollout(z,labels)
 @classmethod
 def load_checkpoint(cls,path,evaluate_mode=None,device='cpu',**kwargs):
  st=torch.load(path,map_location='cpu',weights_only=False);model=cls(st['student_config'],st['sampling_times'],st['clip_denoised']);model.load_state_dict(st['state_dict'],strict=True)
  if evaluate_mode:assert len(model.sampling_times)=={'one_nfe':1,'few_nfe':4}[evaluate_mode]
  assert model.count_parameters()==89577673
  return model.to(device).eval()

def export(path,backbone,config,times,clip,metadata):
 model=Student90(config,times,clip);model.backbone.load_state_dict({k:v.detach().cpu() for k,v in backbone.state_dict().items()});model.eval()
 shared.shared.atom_torch(path,{'state_dict':model.state_dict(),'student_config':config,'sampling_times':torch.as_tensor(times).cpu(),'clip_denoised':clip,'parameter_count':model.count_parameters(),'prediction_type':'v','fixed_guidance_scale':2.5,**metadata})
 return path
