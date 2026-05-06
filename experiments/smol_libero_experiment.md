# SmolVLA + smol-libero 실험 기록

## 실험 일자
2026-03-09 ~ 2026-03-10

## 모델
- **SmolVLA** (450M params)
  - VLM backbone: SmolVLM2-500M-Video-Instruct (~350M)
  - Action expert: Flow Matching Transformer (~100M)
  - Base model: `lerobot/smolvla_base`

## 데이터셋: `HuggingFaceVLA/smol-libero`
- **Task 1개**: "put both the cream cheese box and the butter in the basket"
- 에피소드: 50개
- 프레임: 13,021개
- Features:
  - `observation.images.image` (256x256 RGB, agentview)
  - `observation.images.image2` (256x256 RGB, eye-in-hand)
  - `observation.state` (8D: eef_pos 3 + eef_orientation 3 + gripper 2)
  - `observation.state.joint` (7D: joint angles) ← 풀 libero 데이터셋에는 없음
  - `action` (7D: eef 6 + gripper 1)

## 학습 설정

### 1차: 1k steps (테스트용)
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=local/smolvla_libero_v2 \
  --dataset.repo_id=HuggingFaceVLA/smol-libero \
  --batch_size=16 \
  --steps=1000 \
  --output_dir=outputs/train/smolvla_libero_v2_1k \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --save_freq=1000
```

### 2차: 50k steps (본 학습)
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=local/smolvla_libero_v2 \
  --dataset.repo_id=HuggingFaceVLA/smol-libero \
  --batch_size=16 \
  --steps=50000 \
  --output_dir=outputs/train/smolvla_libero_v2_50k \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=true
```
- wandb run id: `246od1v2`
- 학습 시간: ~10시간
- VRAM: ~11-12GB (RTX 5080)
- Loss: 0.01까지 정상 수렴
- 체크포인트: `outputs/train/smolvla_libero_v2_50k/checkpoints/050000/pretrained_model`

## Eval 결과

### 실패: libero_spatial로 eval (잘못된 task)
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_v2_50k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.device=cuda \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```
- **결과**: 모든 task에서 동일한 무의미한 동작 (오른쪽으로 팔 이동)
- **원인 1**: libero_spatial(black bowl task 10개)은 학습 데이터와 완전히 다른 task
- **원인 2**: rename_map 방향 문제 — env 출력 image/image2를 camera1/camera2로 바꿔버려서 모델의 input_features(image/image2)와 불일치

### 실패: libero_10으로 eval + rename_map (키 불일치)
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_v2_50k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.device=cuda \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```
- **에러**: `ValueError: All image features are missing from the batch`
- **원인**: rename_map이 batch 키를 camera1/camera2로 바꾸는데, 모델은 image/image2를 기대

### 성공: libero_10으로 eval (rename_map 제거)
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_v2_50k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.device=cuda
```
- **Task 1 (cream cheese + butter) 성공률: ~40%**
- 나머지 9개 task: ~0% (학습하지 않은 task)

### 단일 task만 eval
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_v2_50k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[1]' \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.device=cuda
```
- libero_10 환경에서 task 1 = cream cheese + butter task

## 교훈 & 주의사항

### rename_map 정리
- `rename_map`은 학습 시 **데이터셋 키 → 모델 기대 키** 매핑용
- SmolVLA base model은 camera1/camera2/camera3을 기대
- smol-libero 데이터셋은 image/image2를 가짐
- 학습 시 rename_map으로 image→camera1 매핑
- **eval 시에는 rename_map 불필요** (저장된 모델의 input_features가 이미 image/image2로 설정됨)

### 데이터셋 인덱스 vs 환경 인덱스
- 데이터셋의 task_index와 LIBERO 환경의 task 순서가 다름
- 데이터셋: "put both the cream cheese box and the butter in the basket" = task_index 7
- libero_10 환경: 같은 task = task_id 1
- 항상 환경에서 task 이름으로 확인할 것

### smol-libero vs libero (풀) 차이
| 항목 | smol-libero | libero (풀) |
|------|------------|-------------|
| Task 수 | 1 | 40 |
| 에피소드 | 50 | 1,693 |
| 프레임 | 13,021 | 273,465 |
| observation.state.joint | 있음 (7D) | 없음 |
| 용량 | 1.79 GB | 34 GB |

### 성능이 낮은 이유
- smol-libero는 데모/테스트용 소규모 데이터셋
- 50 에피소드, 1 task만으로는 모델이 충분히 학습 불가
- 논문 90%+ 재현하려면 풀 데이터셋(HuggingFaceVLA/libero) 사용 필요

## 파일 위치
- 학습 체크포인트: `outputs/train/smolvla_libero_v2_50k/checkpoints/`
- Eval 비디오: `outputs/eval/2026-03-10/`
- Interactive 스크립트: `interactive_libero.py`
- 데이터셋 뷰어: `view_dataset.py`
- 풀 데이터셋 캐시: `~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/`
