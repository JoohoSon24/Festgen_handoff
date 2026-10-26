# Diffusion 3 재개용 handoff

생성 시점: 2026-10-26 UTC. 이 디렉터리는 teacher, progressive distillation, GTGAN student 단계의 best checkpoint와 N4 GTGAN 이어 학습 상태, 재개 코드/설정을 모아 둡니다. `manifest.json`에 각 체크포인트 원본 경로·크기·SHA256이 있습니다. 체크포인트는 같은 파일시스템의 hardlink라 추가 디스크를 거의 쓰지 않습니다.

## 현재 실행 상태

- N4 GTGAN extension은 기준 1,000 update에서 이어서 총 2,000 update까지 진행 중입니다. handoff 생성 시 train log 최신 update는 1,700이며, 선택점은 Valid FID 기준입니다. `resume/n4_gtgan_extension_latest.pt`는 생성 시점의 latest snapshot이고, `...best_optimizer.pt`는 update 1,250의 best resume입니다.
- N1 DMD queue는 N4 completion marker를 기다립니다. 2026-10-26 06:03 UTC의 첫 시도는 Teacher 체크포인트 키 `ema`를 찾지 못해 실패한 기록이 있습니다. 재개 전 `Experiment/run_n1_dmd_after_n4.py`의 teacher loader가 `update_010000_teacher.pt` 형식과 맞는지 수정하고 preflight 해야 합니다. 이 handoff는 DMD가 완료됐다고 주장하지 않습니다.
- `checkpoints/student/n4_gtgan_extension_best.ckpt`는 EMA 생성 가중치(best checkpoint)이며 optimizer resume 파일과 역할이 다릅니다.

## 단계별 계보 및 결과

| 단계 | best 가중치 | best Official FID | 구성 요약 |
|---|---|---:|---|
| Teacher Min-SNR | `checkpoints/teacher/min_snr_best.pt` | 22.20594 | 기존 150K teacher에서 Min-SNR gamma=5로 10K 이어 학습, EMA; CFG 2.5, DDIM64 평가 |
| Student guidance/internalization | `checkpoints/distillation/guidance32_best.pt` | 27.12573 | 90M student, guidance 32 단계 best (230K) |
| N16 | `checkpoints/distillation/n16_best.pt` | 27.30594 | CFG 2.5 progressive distillation, 10K/30K |
| N8 | `checkpoints/distillation/n8_best.pt` | 28.75005 | N16에서 N8, 20K/45K |
| N4 | `checkpoints/distillation/n4_best.pt` | 34.32846 | N8에서 N4, 55K/60K |
| N2 | `checkpoints/distillation/n2_best.pt` | 43.25267 | N4에서 N2, 70K/75K |
| N1 | `checkpoints/distillation/n1_best.pt` | 65.90571 | N2에서 N1, 110K/112.5K |
| N4 GTGAN 1K | `checkpoints/student/n4_gtgan_1k_best.ckpt` | 33.39888 | train-GT conditional hinge GAN + frozen low-pass/output anchor, 1K |
| N4 GTGAN extension | `checkpoints/student/n4_gtgan_extension_best.ckpt` | extension 로그 확인 | 위 GAN 설정을 optimizer/discriminator state와 이어서 총 2K까지 진행 |

Official FID는 결과 기록용이고 checkpoint 선택은 각 실험의 development Valid FID 기준입니다. teacher의 초기 scratch 학습 전체 계보는 완전하지 않습니다. 증강 continuation은 이전 best보다 나빠 채택하지 않았습니다.

## 모델 / 학습 설정

- Student: 89,577,673 parameters, width 130, RGB 64x64, 151 classes, v-prediction. 전체 기반 설정은 `results/configs/exp-20261021-student90-adaptive.json`.
- Progressive distillation: N16 → N8 → N4 → N2 → N1. 각 단계 입력은 바로 이전 단계의 best EMA checkpoint입니다. 상세한 설정/예산 변경 내역은 `code/docs/Exp.md`, `code/docs/STUDENT90_BUDGET_EXTENSION.md`, `code/docs/STUDENT90_PROGRESS.md`를 확인하세요.
- GAN: N4 GTGAN은 effective batch 32, microbatch 4, G LR 1e-6, D LR 1e-4, GAN weight 0.1, low-pass anchor 1.0, full-image L1 anchor 0.1, EMA 0.999, R1 gamma 1 / interval 16, 2px translation augmentation입니다. 설정은 포함된 `Experiment/train_gt_gan.py` 안에 기록되어 있습니다.
- 예정된 N1 DMD-inspired fine-tune: generator LR 2e-5, fake-score LR 2e-4, CFG 2, effective batch 64 (microbatch 16), fake-score warmup 200, timestep 50–950, N4 endpoint MSE weight 0.1. 2,500 screening 후 Valid FID가 개선될 때만 5K/10K로 연장합니다. 코드는 `Experiment/run_n1_dmd_after_n4.py`; 위 teacher loader 문제부터 해결해야 합니다.

## 프로젝트 아키텍처와 선택 이유

### 전체 파이프라인

```text
64×64 클래스 조건 이미지 데이터
        │
        ▼
48.9M Teacher (epsilon 예측, 1,000 diffusion timestep)
        │  CFG 2.5로 조건부 분포를 정하고 Min-SNR gamma=5로 이어 학습
        ▼
89.6M Student (v 예측, 동일한 클래스 조건)
        │  고정 guidance 경로를 먼저 학습한 뒤 timestep grid를 단계적으로 축소
        ├── N16 → N8 → N4 → N2 → N1
        │
        ├── N4 GTGAN: 실제 이미지 판별 신호 + 원본 N4 출력 보존 anchor
        └── N1 DMD-inspired: teacher/fake score 차이 + N4 endpoint 보존 (대기 중)
```

Teacher와 student는 같은 64×64 RGB 이미지 및 151개 클래스 공간을 사용하지만 예측 파라미터화와 크기는 다릅니다. Teacher는 epsilon을 예측하는 48,883,107 parameter UNet이고, student는 velocity(v)를 예측하는 89,577,673 parameter UNet입니다. Student backbone은 width 130, 채널 배수 `[1,2,3,4]`, 레벨별 residual block 2개, 16×16 해상도 attention으로 구성됩니다. 시간 embedding과 학습된 class embedding에 timestep 및 class를 넣고, null class는 151번 label입니다.

### Teacher → Student 증류

- **Teacher를 고정**해 증류 target이 학습 도중 움직이지 않게 합니다. Teacher CFG 2.5의 출력 분포를 student가 따라가도록 하여, 별도의 unconditional 호출 없이 조건부 모델 하나로 해당 guidance 동작을 내도록 합니다. 따라서 inference 비용에 CFG용 두 번째 network 호출이 추가되지 않습니다.
- **90M student를 사용**한 이유는 NFE를 줄일 때 step 수 감소로 생기는 오차를 더 큰 단일 네트워크의 표현 용량으로 보완하려는 것입니다. 초기화는 새 backbone에서 시작했고 teacher 파라미터를 직접 복사하지 않았습니다. 구조와 prediction type이 달라 단순한 weight 복사는 맞지 않습니다.
- **velocity 예측과 truncated-SNR 가중 MSE**를 사용합니다. 설정상 손실은 `max(alpha², sigma²) × MSE(v)`이며, noise level에 따라 학습 신호가 한 구간에 과도하게 몰리는 것을 줄입니다.
- **중첩된 timestep grid로 progressive distillation**합니다. 32-step guidance/internalization에서 시작해 N16→N8→N4→N2→N1 순으로 절반씩 sampling step을 줄이고, 각 단계는 직전 단계의 best EMA를 parent로 받습니다. 한 번에 1-step을 학습시키는 것보다 큰 step 수의 동작을 다음 단계 target으로 전달할 수 있습니다. NFE별 grid는 설정 파일의 `nested_sampling_grids`에 기록되어 있습니다.
- **EMA 가중치를 평가·전달**합니다. 평가 시 checkpoint를 무작위 최신 update 대신 EMA 후보로 고르고, 다음 distillation 단계에도 해당 단계의 best EMA를 전달합니다. 이 선택은 update 간 parameter 변동을 완화하려는 목적입니다.

### Student 사후 개선

- **N4 GTGAN**은 실제 이미지와 생성 이미지를 비교하는 클래스 조건 hinge discriminator를 둡니다. GAN 신호만 세게 적용하면 원래 N4의 색·구도·분포가 바뀔 수 있어, 고정된 원본 student 출력을 target으로 두고 저주파 MSE와 전체 이미지 L1 anchor를 함께 둡니다. 목적은 GT의 선명한 경계 정보를 보태면서 기존 생성 분포의 급격한 이탈을 억제하는 것입니다. R1 penalty와 작은 translation 증강은 discriminator 학습의 불안정을 줄이기 위한 설정입니다.
- **DMD-inspired N1**은 frozen real-score Teacher와 학습되는 fake-score 모델의 score 차이로 student에 분포 이동 방향을 줍니다. score 차이는 noisy sample의 timestep/noise 추정에 민감할 수 있으므로, N4 출력에 대한 matched-noise endpoint 회귀를 보조 loss로 두어 이미 확보된 N4 품질을 기준점으로 유지합니다. 이는 정통 DMD 구현과 완전히 같다고 간주하지 않고, 현재 코드의 score-matching fine-tune으로 기록합니다. 현재는 실행 전 Teacher checkpoint loader 형식 문제를 해결해야 합니다.

### 데이터와 평가 경계

학습은 `seed42_holdout5/train.txt`의 16,324개 이미지로 하고 holdout 개발 split은 checkpoint 선택 및 Valid FID에 사용합니다. **Official FID는 최종 보고용 지표**이고, 실험 중 선택은 개발 Valid FID 기준입니다. 이 분리를 둔 이유는 Official 평가 결과를 반복적으로 들여다보며 checkpoint를 선택하는 평가 누수를 피하기 위해서입니다. 데이터 증강 continuation은 기존 teacher best를 넘지 못해 현재 teacher target으로 채택하지 않았습니다. 클래스별 또는 seed별 작은 평가 표본의 FID는 변동이 클 수 있으므로, 낮은 표본 수의 진단 결과만으로 학습 처방을 결정하지 않습니다.

각 구성의 실행 순서, hyperparameter 변경 이력, stage별 상세 결과는 `code/docs/Exp.md`, `code/docs/STUDENT90_BUDGET_EXTENSION.md`, `code/docs/STUDENT90_PROGRESS.md`에 있습니다. 설정 파일은 `results/configs/exp-20261021-student90-adaptive.json`입니다.

## 재개 방법

1. 프로젝트 환경의 Python/CUDA 및 데이터 경로를 준비합니다. 기준 venv는 `/root/Diffusion/code/workspace_distillation/.venv/bin/python`입니다.
2. 이미지와 split manifest는 `/root/Diffusion/code/workspace_distillation/data/development/seed42_holdout5/` 및 `/root/Diffusion/code/workspace_distillation/data/`에 있습니다. 데이터는 이 handoff에 복제하지 않았습니다. 다른 머신으로 옮길 때는 해당 데이터와 패키지 전체를 함께 전달해야 합니다.
3. 각 runner의 상단 경로와 config hash 검사를 확인한 뒤, 현재 run의 checkpoint와 RNG/optimizer state에서 재개합니다. GTGAN continuation에는 `resume/n4_gtgan_extension_latest.pt`가, 직전 best update 재개에는 `resume/n4_gtgan_extension_best_optimizer.pt`가 있습니다.
4. DMD는 `n4_extension_completed.json` marker 생성과 teacher checkpoint loader 수정 이후 재개합니다. stage best EMA/weights와 full optimizer resume은 서로 대체할 수 없으므로 목적에 맞는 파일을 선택하세요.

## 포함 파일

- `checkpoints/`: teacher/student/distillation stage best weights
- `resume/`: N4 GTGAN latest 및 best optimizer/discriminator state
- `code/Experiment/`: 실험 runner와 모델/유틸 코드 사본
- `code/docs/`, `results/`: 프로토콜, 설정, 결과와 진행 기록 사본
- `manifest.json`: 체크포인트 출처 및 SHA256
