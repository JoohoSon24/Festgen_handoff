# Teacher seed 재현 및 CFG 탐색

갱신: 2026-10-19 17:34:44 UTC

상태: **completed** / 단계: completed / 작업: None

증강 실험과 Min-SNR 실험은 원본에서 독립적으로 출발했다. 현재 최선 Min-SNR은 무증강 train으로 학습했다.

| 작업 | 상태 | 생성 완료 | Official FID ↓ | Valid FID ↓ | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| original_seed42_cfg2.5 | reused | 3020 | 22.505840 | 42.406386 | 0.754636 | 0.558278 |
| min_snr_seed42_cfg2.5 | reused | 3020 | 22.205939 | 42.065375 | 0.752649 | 0.563245 |
| original_seed1042_cfg2.5 | completed | 3020 | 22.577962 | 42.349602 | 0.755629 | 0.545364 |
| min_snr_seed1042_cfg2.5 | completed | 3020 | 22.288333 | 42.129443 | 0.747020 | 0.567550 |
| original_seed2042_cfg2.5 | completed | 3020 | 22.584749 | 42.263247 | 0.750993 | 0.566225 |
| min_snr_seed2042_cfg2.5 | completed | 3020 | 22.435029 | 42.080465 | 0.736755 | 0.600993 |
| min_snr_seed42_cfg2 | completed | 3020 | 23.958015 | 41.723655 | 0.719536 | 0.606291 |
| min_snr_seed42_cfg3 | completed | 3020 | 22.111313 | 43.047002 | 0.763245 | 0.539073 |
| min_snr_seed42_cfg3.5 | completed | 3020 | 22.672854 | 44.155103 | 0.765563 | 0.528146 |

## 동일 seed의 원본 대비 개선

| seed base | 원본 FID | Min-SNR FID | 원본−Min-SNR (양수=개선) |
|---:|---:|---:|---:|
| 42 | 22.505840 | 22.205939 | 0.299902 |
| 1042 | 22.577962 | 22.288333 | 0.289629 |
| 2042 | 22.584749 | 22.435029 | 0.149720 |

평균 paired 개선: 0.246417; 개선 seed 3/3. 소수 생성 seed 비교이므로 통계적 유의성을 주장하지 않는다.

현재 CFG 최선: 3.0, Official FID 22.111313

고정 조건: DDIM64, EMA0.9999, 생성3,020장(151클래스×20), batch16, bf16, clean-fid. CFG2/2.5/3/3.5는 실제 NFE128이다. seed는 batch별 seed_base+batch_index. Official은 평가·선택에만 사용하며 독립 test가 아니다. 기존 체크포인트와 Student 기준은 변경하지 않는다.

[실험 설계](Exp.md) · [상태 JSON](../FID/teacher_seed_cfg/status.json)
