# Diffusion 3 — Teacher 개선 실험 1~5

이 작업 공간은 기존 프로젝트의 코드·dataset·split·원본 checkpoint를 수정하지 않는다. 모든 신규 코드, 증강 이미지, 학습 checkpoint 및 평가 결과는 `/root/Diffusion 3` 아래에 저장한다. 기존에 완료한 1번 평가 결과는 이곳으로 이관하여 재사용한다. 이전 결과 경로에는 호환 링크만 남긴다. Git 원격 생성·업로드는 하지 않는다.

## 실행 순서

**1번 → 5번 → 2번 → 3번 → 4번** 순으로 GPU 작업을 자동 실행한다. 증강 이미지 준비는 CPU에서 먼저 진행하며, 5번 학습은 데이터 준비가 정상 완료된 뒤 시작한다. 오류가 발생하면 후속 학습을 중단하고 오류를 기록한다.

| 기존 번호 | 실험 | 순서 | update |
|---|---|---:|---:|
| 1 | Teacher150K/200K 재현 및 CFG·EMA·샘플러 선택 | 1 | 재학습 없음 |
| 5 | 클래스 균형·증강 데이터로 Teacher 이어 학습 | 2 | 30K |
| 2 | timestep별 loss 진단 | 3 | 학습 없음 |
| 3 | 기존 uniform epsilon MSE 이어 학습 | 4 | 10K |
| 4 | Min-SNR gamma5 이어 학습 | 5 | 10K |

5번, 3번, 4번은 **1번에서 선택한 동일 원본 full Teacher checkpoint에서 독립적으로 출발**한다. 실행 순서만 변경하며 5번에서 학습한 가중치를 3·4번으로 연쇄 전달하지 않는다. 3·4번이 동일 데이터·초기화·optimizer/RNG 조건을 사용하는 objective 비교가 되도록 유지하기 위함이다. 5번의 개선 후보도 전체 최선 후보 선정에 포함한다.

## 1번 — 기존 Teacher 재현 및 추론 조건 선택

### 구조

48,883,107개 파라미터의 기존 class-conditional UNet Teacher150K와200K를 읽는다. 원본 EMA0.9999, DDIM32, CFG2로 각각3,020장을 새로 생성해 과거 Official FID와 차이0.1 미만인지 확인한다. 두 checkpoint 중 더 좋은 것을 기준으로 CFG1.5/2/2.5를 비교한다. 최선 CFG에서 EMA0.999와0.9999를 비교하고, 선택한 조건으로 DDIM50·64를 추가 평가한다. 조건 탐색은 이 순서의 제한된 탐색이며 모든 조합의 전수 조사는 아니다.

CFG는 cond/uncond 두 forward를 사용하므로 DDIM32/50/64는 실제 NFE64/100/128이다. 모델 hook으로 호출 수를 검증한다. 이후 학습 후보도 동일한 선택 CFG·sampling steps로 평가한다.

### 의의

재학습 없이 더 좋은 생성 target을 확보하고 재현 가능한 기준을 만든다. 체크포인트나 추론 설정 선택 효과를 objective·데이터 증강 효과와 구분한다. 모델 크기는 바꾸지 않는다.

### 이관 시 확인된 결과

- 150K, EMA0.9999, DDIM32, CFG2: Official FID25.009206.
- 200K, EMA0.9999, DDIM32, CFG2:26.286256.
- 150K, EMA0.9999, DDIM32, CFG2.5:23.383282.
- 150K, EMA0.999, DDIM32, CFG2.5:24.686788.
- 150K, EMA0.9999, DDIM50, CFG2.5:22.749484.
- DDIM64는 이관 후 미완료 평가부터 다시 생성한다. 완료된 다른 평가는 재사용한다.

## 5번 — 클래스 균형과 공간 증강을 적용한 이어 학습

### 구조

개발 train16,324장만 읽고151개 클래스를 각178장으로 맞춘다. 원본16,324장을 별도 데이터 폴더에 복사하고 부족분10,554장을 실제 이미지 파일로 생성해 총26,878장으로 구성한다. source 경로·반전 여부·회전 각도를 provenance.json에 저장한다. holdout755장과Official3,020장은 학습 원본 후보에서 제외한다.

부족한 클래스의 원본을 반복 선택해 좌우 반전 확률0.5, 회전 각도 uniform[-10°,10°]를 적용한다. 원본 해상도에서 반사 패딩 후 bicubic 회전·원래 크기 crop을 수행하고, 학습 시64×64로 resize한다. 색 변환·상하 반전은 적용하지 않는다. 색 변환이 없어도 보간에 따른 픽셀 값 변화는 발생한다. 증강은 독립된 실제 표본을 새로 수집한 것과 다르다.

1번의 선택 parent raw·optimizer·EMA를 복원하고 기존 epsilon MSE로30K update를 학습한다. 유효 batch64, LR5e-5, 기존 class dropout·EMA·gradient clip을 유지한다. 데이터 manifest가 바뀌므로 epoch cursor는0으로 초기화한다. 데이터가 균형 잡혀 있어 일반적인 shuffled epoch loader로 클래스 노출량을 맞춘다. 5K/10K/20K/30K마다 full training checkpoint와 선택 EMA, 생성 이미지, FID·precision·recall을 저장한다. 약71회분 데이터 노출에 해당한다.

### 의의와 해석 범위

희소 클래스의 학습 노출량과 자세 다양성을 늘려 기존 Teacher를 개선할 수 있는지 확인한다. 처음부터 다시 학습할 비용을 피하면서 기존 표현을 수정한다. 클래스 균형·증강·보수적 LR이 함께 바뀌므로 개선을 증강 단독 효과로 해석하지 않는다. 기존 분석에서 데이터 수와 class FID의 연관이 약했으므로 개선은 사전에 보장되지 않는다. 회전 보간이 경계를 흐릴 가능성도 있다.

5번의 각 checkpoint와 1번 원본을 동일 조건으로 비교하고, 최종 30K가 악화하면 이전 best를 보존한다. 다음3·4번의 출발점은5번 가중치가 아닌1번 원본이다.

## 2번 — timestep별 loss 진단

### 구조

1번 선택 Teacher의 raw와 선택 EMA에 대해 학습 데이터에서 클래스별5장씩 고정 추출한755장, 전체 개발 holdout755장을 사용한다. timestep50/150/.../950의10개 구간을 동일한 고정 noise로 평가하고 log-SNR·unweighted epsilon MSE·Min-SNR gamma5 가중치를 기록한다. optimizer update는 수행하지 않는다. 여기서 진단하는 모델은1번 원본이며,5번 증강 모델의 timestep 진단으로 표시하지 않는다.

### 의의

Teacher가 어느 노이즈 구간에서 오류가 큰지, train과holdout의 차이가 어떤지 확인한다. 3·4번 objective 비교의 해석 근거를 제공한다. timestep별 문제 난도가 달라 MSE가 큰 구간을 곧바로 학습 병목이라고 단정하지 않는다.

## 3번 — 기존 epsilon MSE 대조군

### 구조

1번에서 선택한 원본 full checkpoint의 raw·optimizer·EMA·RNG·원래 데이터 순서를 복원한다. 원래 개발 train16,324장, 무증강, batch64, LR2e-4를 유지해10K update 이어 학습한다. timestep을 균등 추출하고 per-sample epsilon MSE를 평균한다. 5K/10K에서 동일 생성·평가 및 timestep 진단을 수행한다.

### 의의

증강이나 objective 변경 없이 추가 학습만 해도 개선되는지 확인하는 대조군이다.4번과 동일 초기 조건을 사용해 loss weighting 효과를 분리한다.5번과는 LR·데이터가 달라 증강만의 엄밀한 단일 요인 비교가 아니다.

## 4번 — Min-SNR gamma5 objective 비교

### 구조

3번과 동일 원본 parent·데이터·optimizer·RNG에서 독립적으로 출발한다. epsilon predictor와uniform timestep 추출을 유지하며 per-sample MSE에 `min(SNR,5)/SNR` 가중치를 적용하는 부분만 바꾼다. batch64, LR2e-4,10K update,5K/10K 평가를3번과 동일하게 유지한다. weighted loss와 공통 unweighted epsilon MSE를 함께 기록한다.

### 의의

노이즈 구간 사이의 학습 배분 변경이 기존 Teacher의 생성 분포를 개선하는지 검증한다.3번과4번의 weighted loss 절대값은 직접 비교하지 않고 공통 진단과FID·precision·recall로 판단한다.

## 공통 평가·예산·보존

- 생성3,020장, 클래스별20장, seed42+batch index, batch16, bf16, 동일 clean-fid clean Inception2048 특징을 사용한다.
- Official reference3,020장과개발 holdout755장의 FID를 구분한다. Official을loss/gradient에 사용하지 않지만 반복 선택에 사용하므로 독립 test는 아니다.
- Teacher 학습 파라미터48,883,107개로100M 이하이며, peak allocated VRAM20GiB 미만을 확인한다. frozen EMA는별도 optimizer로학습하지 않는다.
- 기존 Student 기준 NFE4 FID32.664720, NFE1 FID38.605126은유지한다. Teacher의높은NFE 평가와 Student NFE1/4를동일compute 조건으로비교하지 않는다.
- 실패는FID/teacher_improvement_20261019/failure.json과docs/Progress.md에기록한다. 재부팅 자동복구 서비스는설치하지 않았다.

## 파일 위치

| 항목 | 경로 |
|---|---|
| 실행 코드 | Experiment/teacher_core.py |
| 증강 학습 구현 | Experiment/augmentation_stage.py 및 자동생성train_augmented.py |
| 증강 이미지 생성 | Experiment/prepare_data.py |
| 실제 증강 데이터와 계보 | Data/ |
| 평가·checkpoint·상태 | FID/teacher_improvement_20261019/ |
| 학습 로그 | Experiment/teacher_queue.log |
| 데이터 준비 로그 | Experiment/data_preparation.log |
| 실시간 결과 문서 | docs/Progress.md |

## 90M Student 후속 계획

90M Student 증류는 이번 1~5번 실행에 포함하지 않는다. 기존에 제안한 89,577,673개 구조의 초기 예산은 CFG 내재화·32-step 경로 학습 75K, 32→16 30K, 16→8 30K, 8→4 40K, 4→2 50K, 2→1 75K로 총 300K update다.

고정 Teacher를 사용하는 progressive distillation 기준이며, 90M Student + 기존 49M trainable fake-score DMD는 공동 100M 예산을 넘는다. 최종 Teacher와 32-step target을 검증한 뒤 Student 구현·메모리·prediction parameterization을 정해야 한다.

증강 이미지 확인: [원본 위 / 증강 아래 미리보기](augmentation_preview.jpg).

## 후속 실험 A — 생성 seed 재현 및 Min-SNR CFG 탐색

위 1~5번은 완료되었다. 최종 Min-SNR gamma5 추가10K 모델은 무증강 개발 train16,324장으로 학습했으며, 증강5번 모델에서 이어받지 않았다. DDIM64/CFG2.5에서 원본 Official22.505840 → Min-SNR22.205939, Valid42.406386 →42.065375이다. 따라서 이 개선은 증강 효과의 증거가 아니다. 증강 실험은5K22.650870에서30K24.058454로 악화했다.

이번 자동 대기열은 다음 두 단계이며 재학습하지 않는다.

1. 원본150K EMA0.9999와 Min-SNR 추가10K EMA0.9999를 추가 seed_base1042/2042에서 각각 생성·평가한다. DDIM64·CFG2.5·batch16·bf16·3,020장 생성 및 참조를 고정한다. 동일 seed끼리 paired FID 차이를 구하고, 기존 seed42를 포함한3개 seed의 평균 차이와 개선 방향을 기록한다. 생성 seed 변화만 확인하며 학습 seed 재현이나 통계적 유의성을 주장하지 않는다.
2. Min-SNR 모델을 seed42에서 CFG2/2.5/3/3.5로 비교한다.2.5는 기존 정확히 동일 조건의 결과를 재사용한다. 나머지3개 조건을 새로 평가한다. 모두 실제 NFE128이며 precision·recall·Official/Valid FID를 함께 기록한다. CFG 최선은 이 고정 seed·제한된 후보군에서의 최선이며 다른 seed에서 추가 검증 전 일반적 우월성을 단정하지 않는다.

새로운 전체 생성 평가는7회(4회 seed 비교 +3회 CFG)다. DDIM64 기존 실측을 바탕으로 대략40~70분 예상이며 GPU·feature 계산 시간에 따라 달라진다. 결과는 [TEACHER_SEED_CFG_PROGRESS.md](TEACHER_SEED_CFG_PROGRESS.md)에5초 주기로 갱신하고, `FID/teacher_seed_cfg`에 이미지·특징·지표·설정·체크포인트 hash를 보존한다. 기존 체크포인트와 Student 기준은 변경하지 않는다. Official 반복 선택은 독립 test가 아니다.

## 후속 실험 B — CFG3.0의 추가 seed 검증

Min-SNR 추가10K Teacher를 고정하고, seed1042/2042에서 CFG3.0의3,020장 생성·평가를 각각 새로 수행한다. 동일 seed의 기존 CFG2.5 결과 및 seed42의CFG2.5/3.0 결과를 재사용한다. DDIM64/NFE128, EMA0.9999, batch16, bf16, 동일 reference를 유지한다. 세 seed의 paired FID 차이·평균 차이·개선 방향과precision/recall을 비교한다. 재학습하거나 Student 기준을 변경하지 않는다.

코드: Experiment/verify_cfg_seeds.py. 결과: FID/teacher_cfg_seed_verification. 실시간 문서: [TEACHER_CFG_SEED_VERIFICATION.md](TEACHER_CFG_SEED_VERIFICATION.md).

## 승인 후 실행 — 90M Student 증류 및 NFE1·NFE4 fine-tune

사용자가 2026-10-20 설정을 승인했다. 기존의 ‘Student 미실행’ 표기는 당시 상태이며 현재는 [승인 설정](STUDENT90_SETTINGS_REVIEW.md)에 따라 별도90M 실행기를 시작했다. [실행 구조](STUDENT90_RUN.md)와[실시간 진행](STUDENT90_PROGRESS.md)을 따른다. GPU smoke 및 quality gate 실패 시 후속 학습은 중단한다. 기존 Teacher 실험1~5와후속seed/CFG 검증 결과는 그대로 보존한다.


## 현재 실행: Student 초기화 및1.7배 예산

사용자가이전90M 실행을초기화하고최초예산의1.7배로재시작하도록요청했다. 총510K 증류후NFE1/NFE4 GT-GAN 각1K를수행한다. [새설정과재시작범위](STUDENT90_RESTART17.md)가이전300K 및25%연장계획을대체한다. Teacher와NFE4 FID32 재현자료는유지한다.


## 현재 예산20% 연장 —612K

2026-10-21 사용자 지시로 현재510K에서20% 추가해 총612K update로 변경했다. 초기화하지 않고 현재 model·EMA·optimizer·epoch cursor·RNG를 보존해 재개한다. 상세 단계 예산과 config 검증 방식은 [STUDENT90_BUDGET20.md](STUDENT90_BUDGET20.md)를 따른다. GAN fine-tune은 각1K를 유지한다.


## 현재 실행: 기본425K + 성능 기반 자동 연장

사용자가 승인한내재화200K/후속30K·30K·40K·50K·75K를기본으로하고, 평가결과에따라상한587.5K 안에서자동연장한다. 매단계best EMA와그시점의raw/optimizer/RNG를함께보존하고자동판정과source/data/environment hash를기록한다. [현재프로토콜](STUDENT90_ADAPTIVE_PROTOCOL.md)이이전고정612K 예산과활성latest만보관하던방식을대체한다.
