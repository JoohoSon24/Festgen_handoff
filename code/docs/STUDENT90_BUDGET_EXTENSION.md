# N8 → N4 → N2 추가 버짓 실험

## 계획

2026-10-23 요청에 따라 N8부터 학습 예산을 늘려 순차 재증류한다. N8은 Official FID best인 20K의 durable `best_resume.pt`에서 optimizer와 RNG 상태를 보존해 45K까지 이어 학습한다. N4는 60K, N2는 75K까지 새로 증류한다. 원본 실행 폴더는 보존하고 별도 결과 폴더 `FID/student90_n8_n4_n2_budget_extension_20261023`에 기록한다.

각 단계에서 다음 단계의 부모는 마지막 step이 아니라, 해당 단계에서 측정한 **최저 Official FID의 EMA 체크포인트**로 고른다. 따라서 체인은 `기존 N16 best → 연장 N8 best → 새 N4 best → 새 N2 best`다. FID는 동일한 고정 3,020장 Official 평가 절차를 사용하며, 이 반복 평가 결과를 독립 테스트 점수로 취급하지 않는다.

## 예산

| 단계 | 기준 체크포인트 | 총 예산 | 동작 |
|---|---|---:|---|
| N8 | 기존 N8 best Official FID 28.7501 @20K의 전체 학습 상태 | 45K | optimizer/RNG를 이어 25K 추가 |
| N4 | 연장 N8의 best Official FID EMA | 60K | 새 부모에서 처음부터 증류 |
| N2 | 새 N4의 best Official FID EMA | 75K | 새 부모에서 처음부터 증류 |

기존 N4/N2 단계의 진행 상태를 새 부모에서 이어 붙이지 않는다. 부모 모델이 바뀌므로 각각 새 증류를 시작한다. FID 기록과 각 단계에서 선택된 최고 Official FID 체크포인트는 유지하고, 그 외 대형 학습 체크포인트는 SHA-256 정리 기록을 남긴 뒤 제거한다.

## 진행 결과

| 단계 | 상태 | best Official FID | 마지막 평가 |
|---|---|---:|---:|
| N8 | 45K 완료 | **28.750052 @20K** | 29.264393 @45K |
| N4 | 60K 완료 | **34.328461 @55K** | 34.503511 @60K |
| N2 | 75K 완료 | **43.252675 @70K** | 43.470067 @75K |
| N1 | 112.5K 완료 | **65.905712 @110K** | 66.053101 @112.5K |
| N4 GT-GAN | 첫 1K 완료; 추가 1K 실행 예정 | **33.398881 @1K** | 후속구간 종료 시 +1,250/1,500/1,750/2,000 평가 |
| N1 DMD | N4 추가 1K 완료 후 대기열 실행 | — | 이전 DMD2500 recipe; 2.5K 후 Valid FID 개선 시 최대 10K |

N8 45K 완료 결과의 선택 EMA가 N4 시작점이었고, N4 60K 완료 결과의 선택 EMA가 N2 시작점이었다. 이전 실행 대비 N4 best Official FID는 34.559095에서 34.328461로, N2는 기존 부분 실행의 44.147968에서 43.252675로 낮아졌다. N8 best는 28.750063에서 28.750052로 사실상 유지됐다. 단계별 `selected.pt`와 모든 FID metric/log를 보존했다. 비선택 `latest.pt`, `best_resume.pt`, 중복 export는 SHA-256을 `storage_cleanup.jsonl`에 기록한 뒤 정리했다.

N1은 제가 N8→N4→N2 요청 범위를 좁게 해석해 누락했던 다음 단계다. N1은 별도 폴더 `FID/student90_n1_budget_extension_20261025/`에서 N2의 최고 Official FID **43.252675 @70K** EMA를 부모로 삼아 112.5K까지 학습했다.

### N1 결과

| 항목 | 결과 |
|---|---:|
| 학습 예산 | 112,500 update |
| 시작점 | N2 best EMA, Official FID 43.252675 @70K |
| 선택 EMA | **110,000 update** |
| 최고 Official FID / Valid FID | **65.905712 / 72.846169** |
| 최종 112,500 update Official FID / Valid FID | 66.053101 / 72.937930 |
| 파라미터 | 89,577,673 |
| precision / recall (k=3), best 시점 | 0.434437 / 0.377483 |

110K EMA가 이 run의 best Official FID여서 `n1/selected.pt`에 보존했다. 최종 112.5K 평가값이 조금 나빠져도 선택 체크포인트는 best EMA를 유지한다. 모든 평가와 학습 로그는 `n1_result.json`, `n1/eval_*/metrics.json`, `n1/train.jsonl`에 있다. Official 집합의 반복 평가로 체크포인트를 선택했으므로 이 best 값은 독립 테스트 점수가 아니다.

후속 N4는 기존 N4 best EMA(55K, Official FID 34.328461)를 시작점으로 이전의 **GT conditional hinge GAN + frozen anchor** recipe를 1K update 적용했다. Official FID는 34.328428에서 33.398881로 개선됐다. 요청에 따라 저장된 1K optimizer·판별기 상태를 복원해 같은 recipe를 추가 1K(총 2K까지) 이어서 실행하며 1,250/1,500/1,750/2,000에서 평가한다. 이 recipe에는 DMD/fake-score가 없다. 첫 orchestration은 N1 종료 후 N4 출력 폴더가 없어 중단됐으며, N1을 재실행하지 않고 별도 복구 스크립트 [`run_n4_gtgan_after_n1.py`](../Experiment/run_n4_gtgan_after_n1.py)로 첫 1K를 완료했다. 추가 구간은 [`continue_n4_gtgan_then_dmd.py`](../Experiment/continue_n4_gtgan_then_dmd.py)에서 복구 체크포인트를 읽는다. 실시간 상태는 [진행 문서](STUDENT90_PROGRESS.md)에 기록한다.

N4 종료 후 N1의 현재 90M EMA를 초기 생성기로 삼아 DMD fine-tune을 진행하도록 대기열과 별도 실행기를 준비했다. DMD recipe, 90M 구조 적용, 예산, Valid-FID 선택, 38대 과거 결과의 근거 한계는 [N1 DMD 실험 계획](N1_DMD_AFTER_N4.md)에 기록했다.

## 재현 및 상태

실행 스크립트: [`run_student90_n8_n4_n2_budget_extension.py`](../Experiment/run_student90_n8_n4_n2_budget_extension.py)

상태·체크포인트·평가 기록은 `FID/student90_n8_n4_n2_budget_extension_20261023/`에 저장한다. 단계별 `config.json`에 실제 부모 경로와 SHA-256이 들어가며, `chain.json`에는 선택된 best Official FID 부모가 기록된다. 원본 N2 run은 36.7K에서 중단됐고 원본 체크포인트는 보존했다. N8은 best Official FID full-resume 체크포인트에서 이어 45K를 완료했다. N4는 60K 예산으로 재증류 중이며, 직전 중단 시 1K였다.

## 90M Teacher를 새로 학습하는 대안의 시간

기존 무증강 Teacher는 48,883,107개 파라미터다. 실제 기록상 100K 학습은 16,725초(4.65시간), 다음 50K는 8,359초(2.32시간), 그다음 50K는 8,403초(2.33시간)였다. 따라서 현재 49M 구조를 300K까지 늘리는 것은 총 약 14시간 규모로 추정된다.

다만 버짓 증가는 파라미터 shape를 바꾸지 않으므로 49M Teacher를 89.58M Student에 strict initialization할 수 있게 만들지는 않는다. 직접 초기화하려면 Teacher부터 Student와 같은 `model_channels=130`, UNet 구조·class embedding·출력 parameterization을 사용해야 한다. 현재 Student는 89,577,673개이며 `v` prediction을 쓴다.

90M Student N16에서 관측된 학습 속도는 30K에 12,887초, 약 0.43초/update다. 90M Teacher는 frozen target 계산이 빠져 Student 증류보다 조금 빠를 가능성이 있지만, 장치·학습 구현 차이를 감안해 0.31–0.43초/update로 잡으면 200K는 약 17–24시간, 300K는 약 26–36시간이다. 별도 평가·초기화 검증 시간은 추가된다. 이는 측정값이 아니라 현재 49M Teacher와 90M Student 속도를 이용한 추정이다.

직접 Teacher 초기화 경로를 택하면 최소 200K, 비교 기준을 확보하려면 300K 안팎의 90M Teacher 학습이 합리적인 실험 예산이다. 현재 N8→N4→N2 추가 버짓 run은 별도로 시작해 원본 결과를 보존한다.
