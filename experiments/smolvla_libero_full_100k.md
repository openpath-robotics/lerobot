# SmolVLA + LIBERO Full 실험 기록

## 1. 실험 개요

- **목표**: SmolVLA(450M)를 LIBERO 풀 데이터셋으로 학습하고 4개 suite 전체에서 eval
- **GPU**: NVIDIA RTX 5080 (16GB VRAM)
- **학습 기간**: 2026-03-10 ~ 2026-03-11
- **학습 소요 시간**: ~20시간 (100k steps)

---

## 2. 모델 정보

- **SmolVLA** (450M params total)
  - VLM backbone: SmolVLM2-500M-Video-Instruct (SigLIP vision encoder + SmolLM2 decoder)
  - Action expert: Flow Matching Transformer (~100M params)
  - 학습 가능 파라미터: 99,880,992 (100M) — action expert + state_proj만 학습
  - 전체 파라미터: 450,046,176 (450M)
  - `train_expert_only=true`, `freeze_vision_encoder=true`
  - 추론 VRAM: ~0.86GB, 학습 VRAM (batch=16): ~11-12GB

---

## 3. 데이터셋: HuggingFaceVLA/libero (풀)

| 항목 | 값 |
|------|------|
| 총 task 수 | 40 |
| 총 에피소드 | 1,693 |
| 총 프레임 | 273,465 |
| 용량 | 33GB |
| FPS | 10 |
| 로봇 | Franka Panda (시뮬레이션) |

### Features
| Feature | Shape | 설명 |
|---------|-------|------|
| `observation.images.image` | (256, 256, 3) | agentview 카메라 |
| `observation.images.image2` | (256, 256, 3) | eye-in-hand 카메라 |
| `observation.state` | (8,) | eef_pos(3) + eef_orientation(3) + gripper(2) |
| `action` | (7,) | eef(6) + gripper(1), relative control |

### 4개 Suite (각 10 task)
- **libero_spatial**: 같은 물체(black bowl)를 다른 위치에서 pick & place
- **libero_object**: 같은 장소에서 다른 물체를 pick & place
- **libero_goal**: 같은 물체/장소에서 다른 목표 수행
- **libero_10**: 서로 다른 scene의 복합 task (2-step task 포함, 가장 어려움)

### 데이터 캐시 위치
`~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/`

---

## 4. 학습 설정

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.repo_id=local/smolvla_libero_full \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --batch_size=16 \
  --steps=100000 \
  --output_dir=outputs/train/smolvla_libero_full_100k \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=20000 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --wandb.enable=true
```

### 주요 설정 값
| 항목 | 값 |
|------|------|
| Batch size | 16 |
| Steps | 100,000 |
| Learning rate | 0.0001 |
| Optimizer | AdamW (betas=0.9/0.95, weight_decay=1e-10) |
| Scheduler | warmup 1000 steps → decay |
| Chunk size | 50 (action chunk 길이) |
| n_action_steps | 50 (chunk 재사용 수) |
| num_steps (denoising) | 10 |
| rename_map | 사용 안 함 |

### 체크포인트 위치
`outputs/train/smolvla_libero_full_100k/checkpoints/`
- 010000, 020000, ..., 100000, last → 100000

---

## 5. Eval 결과

### Suite별 성공률

| Suite | 성공률 | 에피소드 수 |
|-------|--------|-------------|
| **libero_goal** | **62%** | 100 (10 task × 10 ep) |
| **libero_object** | **51%** | 100 |
| **libero_spatial** | **38%** | 100 |
| **libero_10** | **24%** | 100 |
| **전체 평균** | **43.75%** | 400 |

### Eval 명령어
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_full_100k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \   # spatial / object / goal / libero_10
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.device=cuda
```
※ `--rename_map` 없이 실행 (모델이 image/image2 키로 학습됨)

### Task별 상세

#### libero_goal (62%) — 최고 성적
| Task | 성공률 |
|------|--------|
| 7 (push plate to front of stove) | **100%** |
| 1 (put wine bottle on rack) | 90% |
| 4 (put wine bottle on top of cabinet) | 90% |
| 2 (open top drawer, put bowl inside) | 80% |
| 0 (put bowl on plate) | 60% |
| 3 (turn on stove) | 60% |
| 8 (put bowl on top of cabinet) | 60% |
| 5 (open middle drawer) | 30% |
| 6 (put cream cheese in bowl) | 30% |
| 9 (put bowl on stove) | 20% |

#### libero_object (51%)
| Task | 성공률 |
|------|--------|
| 2 (salad dressing → basket) | 70% |
| 5 (tomato sauce → basket) | 70% |
| 4 (ketchup → basket) | 60% |
| 8 (chocolate pudding → basket) | 60% |
| 6 (butter → basket) | 50% |
| 9 (orange juice → basket) | 50% |
| 0 (alphabet soup → basket) | 40% |
| 1 (cream cheese → basket) | 40% |
| 7 (milk → basket) | 40% |
| 3 (bbq sauce → basket) | 30% |

#### libero_spatial (38%)
| Task | 성공률 |
|------|--------|
| 4 (bowl in top drawer) | 50% |
| 7 (bowl on stove) | 60% |
| 0 (bowl between plate and ramekin) | 40% |
| 6 (bowl next to cookie box) | 40% |
| 8 (bowl next to plate) | 40% |
| 9 (bowl on wooden cabinet) | 40% |
| 1 (bowl next to ramekin) | 30% |
| 3 (bowl on cookie box) | 30% |
| 5 (bowl on ramekin) | 30% |
| 2 (bowl from table center) | 20% |

#### libero_10 (24%) — 최저 성적
| Task | 성공률 |
|------|--------|
| 5 (pick up book → caddy) | 70% |
| 3 (black bowl → bottom drawer, close) | 60% |
| 2 (stove on + moka pot) | 40% |
| 6 (white mug → plate + chocolate pudding) | 20% |
| 9 (yellow mug → microwave, close) | 20% |
| 4 (white mug left + yellow mug right) | 10% |
| 7 (alphabet soup + cream cheese → basket) | 10% |
| 8 (both moka pots → stove) | 10% |
| 0 (alphabet soup + tomato sauce → basket) | 0% |
| 1 (cream cheese + butter → basket) | 0% |

### Eval 결과 파일 위치
- libero_spatial: `outputs/eval/2026-03-11/10-07-05_libero_smolvla/`
- libero_object: `outputs/eval/2026-03-11/10-27-15_libero_smolvla/`
- libero_10: `outputs/eval/2026-03-11/10-52-30_libero_smolvla/`
- libero_goal: `outputs/eval/2026-03-11/11-23-04_libero_smolvla/`

---

## 6. 이전 실험 비교 (smol-libero)

| 항목 | smol-libero (이전) | libero full (이번) |
|------|-------------------|-------------------|
| 데이터셋 | HuggingFaceVLA/smol-libero | HuggingFaceVLA/libero |
| Task 수 | 1 | 40 |
| 에피소드 | 50 | 1,693 |
| 프레임 | 13,021 | 273,465 |
| 학습 steps | 50,000 | 100,000 |
| 크림치즈+버터 task | 40% (libero_10 task 1) | 0% (libero_10 task 1) |
| 전체 평균 | N/A (1 task만) | 43.75% |

### 크림치즈 task가 0%로 떨어진 이유
- smol-libero: 1 task만 학습 → 그 task에 overfitting
- libero full: 40 task 학습 → 2-step 복합 task(크림치즈+버터 둘 다 옮기기)는 난이도가 높아 100k steps로는 부족
- libero_10의 0% task들은 모두 2-step 복합 task (물체 2개를 순서대로 옮겨야 함)

---

## 7. 중요 Q&A 정리

### Q: rename_map이 뭐야?
- 데이터셋의 feature 키 이름과 모델이 기대하는 키 이름이 다를 때 매핑해주는 설정
- SmolVLA base: `camera1/camera2/camera3` 기대
- LIBERO 데이터셋: `image/image2` 사용
- 학습 시 rename_map으로 매핑하면, 저장된 모델의 input_features가 바뀜
- **eval 시에는 모델의 input_features와 env 출력이 맞아야 함** → rename_map 방향 주의!

### Q: loss는 잘 떨어지는데 eval 성공률이 낮은 이유?
- Loss (training loss)와 eval (시뮬레이터 rollout 성공률)은 다른 metric
- Training loss: 데이터셋의 (obs, action) 쌍에 대한 예측 오차
- Eval: 실제 시뮬레이터에서 로봇을 놓고 task를 수행시켜 성공 여부 판단
- Loss가 낮아도 action 분포의 미세한 오차가 누적되면 실패할 수 있음

### Q: 학습 중 eval은 validation loss인가?
- **아님**. 실제 시뮬레이터에 로봇을 놓고 rollout하는 것
- 매우 느림 (에피소드당 수초~수십초 물리 시뮬레이션)
- 그래서 전체 40 task eval을 매번 돌리면 학습보다 eval이 더 오래 걸림
- 학습 중에는 대표 suite 1개로 모니터링, 학습 후 전체 eval 권장

### Q: 에피소드 길이가 달라도 되나?
- ACT, SmolVLA, Diffusion Policy 모두 **프레임 단위 샘플링**
- 에피소드 전체를 넣는 게 아니라 (observation, future_actions) 쌍을 뽑아서 학습
- 에피소드가 100프레임이든 300프레임이든 문제없음

### Q: multi-task 학습하면 개별 task 성능이 떨어지지 않나?
- **실험 결과: 학습량이 부족하면 떨어진다**
  - smol-libero (1 task, 50k steps): 크림치즈 task 40%
  - libero full (40 task, 100k steps): 크림치즈 task 0%
  - task가 40배 늘었는데 steps는 2배만 늘림 → task당 학습 기회가 1/20로 감소
- 논문에서 multi-task가 좋다는 건 **충분히 수렴한 후**의 결과
- 수렴 전에는 single-task가 해당 task에서 더 높은 성공률을 보일 수 있음
- 결론: multi-task의 이점을 보려면 **steps를 task 수에 비례하여 충분히 늘려야 함**

### Q: 물체 위치는 eval마다 랜덤인가?
- LIBERO는 init_states(고정 초기 상태)를 사용
- 에피소드별로 미리 정해진 초기 상태 세트에서 샘플링
- 완전 랜덤은 아니지만, 에피소드마다 약간씩 다른 초기 배치

### Q: inference 파라미터 조정은?
- `configuration_smolvla.py`는 기본값 정의 → 직접 수정하지 말 것
- 모델 로드 후 override:
  ```python
  policy.config.num_steps = 20        # denoising steps (높을수록 정교)
  policy.config.n_action_steps = 10   # action chunk 재사용 (낮을수록 자주 추론)
  ```

---

## 8. 논문 대비 성능 차이 원인 (추정)

논문: ~90%+ (suite별) vs 우리: 43.75% (전체 평균)

1. **학습 steps 부족**: 100k → 200k~300k 필요할 수 있음
2. **하이퍼파라미터**: 논문의 정확한 lr schedule, batch size 등 미확인
3. **데이터 전처리**: 논문은 별도 전처리/augmentation 사용 가능
4. **eval 조건**: 에피소드 수, seed 등 차이 가능

### 다음 시도 방향
- 200k~300k steps로 학습 연장
- 논문의 학습 config 확인 (HuggingFace 모델 카드 등)
- 중간 체크포인트(50k, 70k 등)별 eval로 최적 시점 찾기
