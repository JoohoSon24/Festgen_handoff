"""Read-only disk/FID inventory. Never interprets an unlabelled development fid as official."""
import os,json,collections
from pathlib import Path
ROOT=Path('/root/Diffusion 3');rows=json.loads((ROOT/'docs/STORAGE_CHECKPOINT_INVENTORY.json').read_text());byinode={};measurements=collections.defaultdict(list)
for r in rows:
 p=Path(r['path']);s=p.stat();byinode[(s.st_dev,s.st_ino)]=r
keys=['checkpoint','teacher_checkpoint','model_checkpoint','selected_checkpoint']
def ident(p):
 try:
  s=Path(p).stat();return (s.st_dev,s.st_ino)
 except (OSError,TypeError,ValueError):return None
def scan(obj,path,inherited=None):
 if isinstance(obj,dict):
  refs=[obj[k] for k in keys if isinstance(obj.get(k),str)] or inherited or []
  fid=obj.get('official_fid');official=obj.get('reference_split')=='official' or path.name=='official_metrics.json'
  if fid is None and official:fid=obj.get('fid')
  if isinstance(fid,(int,float)):
   meta={k:obj[k] for k in ['seed','seed_base','cfg','guidance_scale','steps','sampling_steps','nfe','weights','checkpoint_step','step','update'] if k in obj};entry={'official_fid':fid,'source':str(path),'conditions':meta}
   for ref in refs:
    key=ident(ref)
    if key in byinode and entry not in measurements[key]:measurements[key].append(entry)
  for field,v in obj.items():
   if isinstance(v,(list,dict)):scan(v,path,refs if field in ['metrics','selected_metrics'] else None)
 elif isinstance(obj,list):
  for v in obj:scan(v,path,inherited)
for root in ['/root/Diffusion','/root/Diffusion 2','/root/Diffusion 3']:
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d not in ['.venv','.git','__pycache__','Data','node_modules'] and not (Path(base)/d).is_symlink()]
  for name in files:
   if not name.endswith('.json') or name.startswith('STORAGE_CHECKPOINT'):continue
   p=Path(base)/name
   try:
    if p.stat().st_size>4_000_000:continue
    scan(json.loads(p.read_text()),p)
   except (OSError,ValueError,UnicodeError):pass
# Exact current experiment associations where full resume contains the same EMA as the evaluated export.
def attach(p,metric,note):
 key=ident(p)
 if key not in byinode or not Path(metric).exists():return
 m=json.loads(Path(metric).read_text());fid=m.get('official_fid')
 if fid is not None:measurements[key].append({'official_fid':fid,'source':str(metric),'conditions':{'note':note,'cfg':m.get('cfg'),'nfe':m.get('nfe')}})
b=ROOT/'FID/teacher_improvement_20261019'
for n in [5000,10000,20000,30000]:attach(b/'5_augmented_continuation'/f'update_{n:06d}_training.pt',b/'5_augmented_continuation'/f'eval_{n:06d}/metrics.json',f'full resume 내 EMA, 추가{n} update')
for arm in ['2_uniform_mse','2_min_snr_gamma5']:attach(b/arm/'latest_training.pt',b/arm/'eval_010000/metrics.json','full resume 내 EMA, 추가10K update')
for r in rows:
 p=Path(r['path']);s=p.stat();ms=measurements[(s.st_dev,s.st_ino)];r['official_measurements']=sorted(ms,key=lambda x:x['official_fid'])
 # Repeated summary references do not make independent measurements.
 dedup={}
 for m in r['official_measurements']:dedup.setdefault((round(m['official_fid'],7),json.dumps(m['conditions'],sort_keys=True)),m)
 r['official_measurements']=list(dedup.values())
(ROOT/'docs/STORAGE_CHECKPOINT_INVENTORY.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
lines=['# 저장 공간·체크포인트·Official FID 목록','', '검사 시점: 2026-10-20. 용량은 할당된 디스크 블록 기준 GiB(2^30 bytes). hardlink는 inode당 한 번만 합산했다. hardlink 하나를 삭제해도 다른 링크가 남으면 공간이 회수되지 않는다. 아래 FID는 파일 내 평가된 EMA/조건에 대응하며 raw weight의 점수로 해석하지 않는다. 같은 checkpoint도 CFG·NFE·seed에 따라 점수가 달라 모든 관측 조건은 JSON에 보존했다. 미확인은 FID가 나쁘다는 뜻이 아니다.','', '## 큰 폴더 순서','', '| 폴더 | 대략 GiB | 설명 |','|---|---:|---|','| /root/Diffusion |39.83|기존 원본 프로젝트·Teacher·다른 계열 실험·환경|','| /root/Diffusion 3 |12.70|이번 Teacher/90M 실험 및 증강 복사본|','| /root/Diffusion 2 |10.21|이전 Student GAN·후속 실험|','| /root/GAN |8.12|별도 GAN 프로젝트·실행환경|','| /root/.vscode-server |3.92|편집기 서버|','| /root/.codex |3.54|세션·플러그인·캐시|','','## 체크포인트 용량 내림차순','', '| 순위 | 할당 GiB | 파일 | Official FID 관측 | hardlink 수 |','|---:|---:|---|---|---:|']
for i,r in enumerate(rows,1):
 ms=r['official_measurements'];desc='미확인 — 해당 파일과 직접 연결된 Official 지표 없음'
 if ms:
  m=ms[0];c=m['conditions'];cond=', '.join(f'{k}={v}' for k,v in c.items() if v is not None);desc=f"{m['official_fid']:.6f} ({cond or '상세 JSON 참조'})"
  if len(ms)>1:desc+=f'; 다른 관측 {len(ms)-1}건'
 lines.append(f"| {i} | {r['allocated']/1024**3:.3f} | `{r['path']}` | {desc} | {r['hardlinks']} |")
lines+=['','## 정리 범위','', '체크포인트·원본 dataset·학습 환경·다른 프로젝트는 삭제하지 않는다. 완료된 증강 실험의 Data 하위 복사 이미지 26,878장만 정리 대상으로 삼는다. 원본 train 이미지·변환 코드·manifest·provenance·실험 checkpoint와 FID는 보존한다. 증강 입력은 필요하면 동일 seed로 재생성할 수 있다. 증강 실험의 추가5K/10K/20K/30K Official FID는22.650870/23.454423/23.770582/24.058454이다. 이 데이터 폴더 자체의 FID는 정의하지 않는다.','', '디스크 정리는 GPU VRAM을 늘리지 않는다. microbatch 증대는 별도의 GPU 처리량·메모리 검사로 결정한다.']
(ROOT/'docs/STORAGE_CHECKPOINT_REPORT.md').write_text('\n'.join(lines)+'\n');print('Report saved; checkpoints',len(rows),'mapped',sum(bool(r['official_measurements']) for r in rows))
for r in rows[:12]:print(round(r['allocated']/1024**3,3),r['path'],[round(m['official_fid'],6) for m in r['official_measurements'][:3]])
