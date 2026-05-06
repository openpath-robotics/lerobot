import argparse
import torch
import cv2
import numpy as np

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import ACTION
from libero.libero import benchmark
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/train/smolvla_libero_v2_50k/checkpoints/last/pretrained_model")
    parser.add_argument("--task", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda")

    # 모델 로드
    print("모델 로딩 중...")
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()

    # preprocessor / postprocessor 구성 (lerobot-eval과 동일)
    env_cfg = LiberoEnvConfig(task=args.task)
    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LIBERO 환경 생성
    print(f"LIBERO 환경 생성 중 ({args.task})...")
    task_suite = benchmark.get_benchmark_dict()[args.task]()

    print("\n사용 가능한 task:")
    for i, task in enumerate(task_suite.tasks):
        marker = " <--" if i == args.task_id else ""
        print(f"  [{i}] {task.language}{marker}")

    current_task_id = args.task_id
    env = _make_env(task_suite, current_task_id, args.task)

    print(f"\n=== SmolVLA Interactive LIBERO Demo ===")
    print(f"모델: {args.model}")
    print(f"State dim: {policy.config.input_features['observation.state'].shape[0]}")
    print("Enter = 현재 task 기본 instruction으로 실행")
    print("숫자 = task 변경 (예: 3)")
    print("텍스트 = 커스텀 instruction")
    print("'q' = 종료\n")

    while True:
        default_instruction = task_suite.tasks[current_task_id].language
        user_input = input(f"[task {current_task_id}] Instruction (기본: {default_instruction}): ").strip()

        if user_input.lower() == "q":
            break

        # 숫자면 task 변경
        if user_input.isdigit():
            new_id = int(user_input)
            if 0 <= new_id < len(task_suite.tasks):
                current_task_id = new_id
                env.close()
                env = _make_env(task_suite, current_task_id, args.task)
                print(f"  Task 변경: [{current_task_id}] {task_suite.tasks[current_task_id].language}")
                continue
            else:
                print(f"  잘못된 task id (0~{len(task_suite.tasks)-1})")
                continue

        instruction = user_input if user_input else default_instruction
        print(f"  -> '{instruction}'")
        print(f"  Rollout 시작 (ESC로 중단)...\n")

        obs, info = env.reset()
        policy.reset()

        for step in range(args.max_steps):
            # 렌더링
            frame = env.render()
            if frame is not None:
                display = frame.copy() if isinstance(frame, np.ndarray) else frame[0].copy()
                cv2.putText(display, f"Step: {step}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"[{current_task_id}] {instruction[:80]}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow("SmolVLA LIBERO", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("  중단됨")
                break

            # lerobot-eval과 동일한 전처리 파이프라인
            observation = preprocess_observation(obs)
            observation["task"] = [instruction]
            # single env이므로 robot_state 텐서에 batch dim 추가
            _add_batch_dim(observation)
            observation = env_preprocessor(observation)

            # language tokenization
            max_length = getattr(policy.config, "tokenizer_max_length", 48)
            tokens = tokenizer(
                instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            observation["observation.language.tokens"] = tokens["input_ids"].to(device)
            observation["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)

            observation = preprocessor(observation)

            with torch.no_grad():
                action = policy.select_action(observation)

            action = postprocessor(action)
            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action_np = action_transition[ACTION].cpu().numpy()

            if action_np.ndim == 2:
                action_np = action_np[0]

            obs, reward, terminated, truncated, info = env.step(action_np)

            if step % 50 == 0:
                print(f"  step={step}, reward={reward}, action={action_np[:4]}...")

            if terminated or truncated:
                print(f"  Episode 종료! step={step}, reward={reward}")
                break

        print(f"  Rollout 완료 ({step + 1} steps)\n")

    cv2.destroyAllWindows()
    env.close()
    print("종료!")


def _add_batch_dim(d):
    """nested dict 내 모든 텐서에 batch dim 추가 (이미 있으면 스킵)"""
    for k, v in d.items():
        if isinstance(v, dict):
            _add_batch_dim(v)
        elif isinstance(v, torch.Tensor):
            # 1D 텐서 -> (1, N), 2D (3,3) 같은 행렬 -> (1, 3, 3)
            # 이미지는 preprocess_observation에서 (1,C,H,W)로 처리됨
            if v.ndim <= 2 and v.shape[0] != 1:
                d[k] = v.unsqueeze(0)


def _make_env(task_suite, task_id, task_name):
    return LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=task_name,
        n_envs=1,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=640,
        visualization_height=480,
    )


if __name__ == "__main__":
    main()
