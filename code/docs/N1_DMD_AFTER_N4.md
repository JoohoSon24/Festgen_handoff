# N1 DMD 증류 계획 — N4 완료 후 실행

## 대기 조건과 실행 순서

N4 GTGAN 첫 1,000 update가 완료됐으며, 요청에 따라 그 best EMA에서 optimizer·판별기 상태를 복원해 **추가 1,000 update**(총 update 2,000까지) 진행한다. N1 DMD는 추가 N4 구간의 완료 마커 `FID/student90_n1_budget_extension_20261025/n4_extension_completed.json`가 생성된 뒤 시작한다. N4 연장이 실패하면 자동 실행을 멈추고 복구를 기다린다. N1의 시작점은 이번 증류 run에서 선택한 90M N1 EMA이고, endpoint target은 N4 연장 결과 중 기존 선택 규칙상 best 체크포인트다.

대기열 상태: [`queue_status.json`](../FID/student90_n1_dmd_after_n4_20261026/queue_status.json) · [run 기록](../FID/student90_n1_dmd_after_n4_20261026/) · [실시간 N1 진행](STUDENT90_PROGRESS.md)

## 이전 “FID 38대 DMD” 방식

복원된 DMD 방법은 기존 1-NFE 생성기를 유지하면서, 고정된 real score model과 학습되는 fake score model의 차이로 distribution-matching gradient를 만들고, 같은 latent와 class에서 나온 4-NFE endpoint를 약하게 회귀하는 방식이다. 구현은 canonical paper의 정확한 재현이라고 주장하지 않으며 VP/v-prediction에 맞춘 변형이다.

재현 가능한 설정은 200 fake-score warmup, noise timestep 50–950, G AdamW 2e-5, fake-score AdamW 2e-4, effective batch 64 (microbatch 16 × 4), gradient clip 1, G EMA 0.999, DMD CFG 2, endpoint MSE weight 0.1이다. 이전 run은 2,500 / 5,000 / 10,000 update를 비교했고, 가장 좋은 DMD 후보가 2,500 update였다. 이 레시피를 현재 90M 모델에 적용하되, 구조와 prediction type 차이는 아래처럼 명시해 보정한다.

## 현재 90M 모델에 적용할 구성

| 역할 | 체크포인트 / 구조 | 사용 |
|---|---|---|
| 초기 생성기 G | `student90_n1_budget_extension_20261025/n1/selected.pt`, 89,577,673 params, v-prediction, NFE1 | 학습 대상 |
| endpoint teacher | N4 GTGAN 완료 결과의 best 90M EMA, v-prediction, NFE4 | 같은 noise/class의 4-step endpoint target |
| real score | Diffusion 3 Min-SNR teacher EMA, 48,883,107 params, epsilon prediction | freeze; CFG 2 |
| fake score | real-score 가중치에서 복사한 48,883,107-param epsilon model | 생성 이미지로 매 update 학습 |

G와 N4 endpoint는 현재 90M 계열과 동일한 v architecture다. Score network는 기존 DMD와 동일하게 epsilon prediction을 유지한다. N1의 90M v 출력은 VP 계수로 x0 이미지로 변환한 뒤 score matching에 넣는다. Official/Valid 이미지와 특징은 training loss에 사용하지 않는다.

## 실행 예산과 체크포인트 선택

먼저 fake-score 200 update warmup 후 G 2,500 update를 수행한다. Valid FID가 변하지 않은 N1 baseline보다 낮으면 5,000까지, 5,000 결과가 2,500보다 낮으면 10,000까지 이어간다. baseline과 후보 중 **최저 Valid FID** checkpoint를 선택하고 Official FID·precision·recall은 결과 보고에만 쓴다. Official FID가 선택 기준에 들어가지 않도록 한다.

재개 가능한 전체 상태는 100 G update마다 저장한다. FID는 baseline, 2,500, 조건부 5,000, 조건부 10,000에서 측정한다. best EMA export와 원본 N1은 보존한다. 설정·부모 SHA-256·학습 로그·metric JSON은 `FID/student90_n1_dmd_after_n4_20261026/`에 둔다.

## 결과 해석의 한계

기존 38.605 Official FID는 과거 NFE1 DMD2500 체크포인트에 연결된 보고값이며, 해당 수치의 일부 원본 기록은 완전하게 복원되지 않았다. 별도 보존된 재현 실행에서 DMD2500은 개발 reference FID 65.386270에서 55.725149로 개선됐지만, 5,000과 10,000 update에서는 각각 58.622004와 60.444141로 악화됐다. 따라서 38대 재현이나 90M N1 개선을 보장하지 않는다. 현재 run은 Valid 분할을 이용해 후보를 고르고 Official 결과를 따로 보고한다.

실행기: [`continue_n4_gtgan_then_dmd.py`](../Experiment/continue_n4_gtgan_then_dmd.py), [`run_n1_dmd_after_n4.py`](../Experiment/run_n1_dmd_after_n4.py) · 대기열: [`queue_n1_dmd_after_n4.py`](../Experiment/queue_n1_dmd_after_n4.py)
