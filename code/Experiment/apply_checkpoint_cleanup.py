"""Execute explicit user thresholds after verifying protected reproducibility files."""
import json,hashlib,shutil,time,os
from pathlib import Path
R=Path('/root/Diffusion 3');plan_path=R/'docs/CHECKPOINT_CLEANUP_PLAN.json';plan=json.loads(plan_path.read_text());bundle=R/'docs/reproducibility/NFE4_FID32/MANIFEST.json';manifest=json.loads(bundle.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
protected_paths={Path(x['path']).resolve() for x in manifest['protected_files']};protected_inodes={(p.stat().st_dev,p.stat().st_ino) for p in protected_paths}
for record in manifest['protected_files']:assert Path(record['path']).is_file() and sha(Path(record['path']))==record['sha256'],record['path']
plan.update(status='validated',before_free_bytes=shutil.disk_usage(R).free,protected_manifest=str(bundle),deleted_paths=[]);plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
for c in plan['candidates']:
 n=c['nfe'];assert n in [1,4] and c['best_verified_official_fid']>={1:39.,4:34.}[n];assert tuple(c['inode']) not in protected_inodes
 for path in c['paths']:
  p=Path(path);assert not p.resolve().is_relative_to(R.resolve());assert p.resolve() not in protected_paths;assert p.suffix in ['.pt','.ckpt','.pth','.safetensors'];assert not p.is_symlink();s=p.stat();assert (s.st_dev,s.st_ino)==tuple(c['inode'])
 c['sha256_before_delete']=sha(Path(c['paths'][0]));c['deleted']=False
 # Record the exact path, hash, and metrics before irreversible cleanup.
 plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 for path in c['paths']:Path(path).unlink();plan['deleted_paths'].append(path)
 c['deleted']=True;plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
for record in manifest['protected_files']:assert Path(record['path']).is_file() and sha(Path(record['path']))==record['sha256'],record['path']
plan.update(status='completed',after_free_bytes=shutil.disk_usage(R).free,completed_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()));plan['free_bytes_gained']=plan['after_free_bytes']-plan['before_free_bytes'];plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
lines=['# 재현성을 보존한 체크포인트 정리 결과','',f"완료: {plan['completed_utc']}",'',f"고유 checkpoint {len(plan['candidates'])}개, 경로 {len(plan['deleted_paths'])}개 삭제. 할당 블록 합계 {plan['unique_allocated_bytes']/1024**3:.3f}GiB. 파일시스템 실제 여유 변화 {plan['free_bytes_gained']/1024**3:.3f}GiB(동시 학습의 checkpoint 쓰기에 따라 다를 수 있음).",'', '## 보존','', '- 현재 Diffusion3 실험 전체.', '- NFE4 FID32.664720 EMA1000 + 원본33.051769 anchor + G/EMA/D/optimizer/RNG resume.', '- NFE1 원본38.605126.', '- 재현 코드·설정·split·원본 이미지·환경·기존 평가 reference/특징·보고서.', '- 기준보다 좋은 관측이 하나라도 있거나 동일 update의Official FID가 확인되지 않은 checkpoint.','', '[NFE4 FID32 재현성 묶음](reproducibility/NFE4_FID32/README.md) · [상세 삭제·hash·FID 근거](CHECKPOINT_CLEANUP_PLAN.json)','', '| 고유 파일 | GiB | NFE | 확인된 최저 Official FID |','|---|---:|---:|---:|']
for c in plan['candidates']:lines.append(f"| `{c['paths'][0]}` | {c['allocated_bytes']/1024**3:.3f} | {c['nfe']} | {c['best_verified_official_fid']:.6f} |")
lines+=['','초기Teacher부터원본33.05Student까지의전체학습계보는일부누락되어있다. 이번보존은저장가중치의추론평가와원본Student에서GAN1000구간재현을포함하며누락된최초Teacher까지복구했다고주장하지않는다.']
(R/'docs/CHECKPOINT_CLEANUP_RESULT.md').write_text('\n'.join(lines)+'\n');print('deleted unique checkpoints',len(plan['candidates']),'paths',len(plan['deleted_paths']),'GiB gained',plan['free_bytes_gained']/1024**3,'protected files verified',len(manifest['protected_files']))
