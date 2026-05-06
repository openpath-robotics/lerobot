"""
HDF5 → LeRobot v3.0 변환 스크립트 (멀티프로세스 + JPEG — 빠른 버전)
듀얼암 로봇 시뮬레이션 데이터 (14 DOF + 4 카메라) → SmolVLA 학습용 데이터셋
"""

import io
import json
import os
import re
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# ============================================================
# 설정 - 여기만 수정하세요
# ============================================================
HDF5_DIR = os.path.expanduser("~/dataset_0324_gzip")
REPO_ID = "local/dual_arm_0402_rot90"
OUTPUT_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/dual_arm_0402_rot90")
FPS = 30
ROBOT_TYPE = "dual_arm"
NUM_WORKERS = 8  # 병렬 처리 워커 수

# 사용할 episode 번호 범위 (파일명 기준, 이 범위 밖 파일은 무시)
EPISODE_NUM_RANGE = (0, 299)

# 에피소드 인덱스 범위 → task 매핑
EPISODE_TASK_MAP = [
    (  0,  99, "pick up the rightmost block with the right arm and place it in the box"),
    (100, 199, "pick up the leftmost block with the left arm and place it in the box"),
    (200, 299, "pick up the leftmost and rightmost blocks with both arms and place them in the boxes"),
]
TASK_DESCRIPTIONS = [desc for _, _, desc in EPISODE_TASK_MAP]

def get_task_index(ep_idx: int) -> int:
    for task_idx, (start, end, _) in enumerate(EPISODE_TASK_MAP):
        if start <= ep_idx <= end:
            return task_idx
    raise ValueError(f"ep_idx={ep_idx} is not covered by EPISODE_TASK_MAP")

CAMERA_MAPPING = {
    "images/cam_center": "observation.images.camera1",
    "images/cam_top": "observation.images.camera2",
    "images/cam_wrist_left": "observation.images.camera3",
    "images/cam_wrist_right": "observation.images.camera4",
}
# ============================================================

WRIST_ROTATE_KEYS = {"images/cam_wrist_left", "images/cam_wrist_right"}

def encode_image_jpeg(img_array: np.ndarray, rotate_cw90: bool = False) -> dict:
    """numpy (H,W,C) uint8 → JPEG bytes (PNG 대비 ~10x 빠름)"""
    img = Image.fromarray(img_array)
    if rotate_cw90:
        img = img.rotate(-90, expand=True)  # 시계방향 90도
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return {"bytes": buf.getvalue(), "path": None}


def process_single_episode(args):
    """한 에피소드를 처리해서 (rows, ep_meta, states, actions) 반환"""
    ep_idx, hdf5_path, global_offset = args
    task_idx = get_task_index(ep_idx)
    task_desc = TASK_DESCRIPTIONS[task_idx]
    cam_hdf5_keys = list(CAMERA_MAPPING.keys())
    cam_lerobot_keys = list(CAMERA_MAPPING.values())

    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["action"].shape[0]
        q_pos = f["q_pos"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)

        # 카메라 데이터 한번에 읽기 (HDF5 I/O 최소화)
        cam_data = {}
        for hdf5_key in cam_hdf5_keys:
            cam_data[hdf5_key] = f[hdf5_key][:]  # (T, H, W, C) 전체 로드

    # 프레임별 row 생성
    rows = []
    for frame_idx in range(n_frames):
        row = {
            "timestamp": float(frame_idx / FPS),
            "frame_index": frame_idx,
            "episode_index": ep_idx,
            "index": global_offset + frame_idx,
            "task_index": task_idx,
            "observation.state": q_pos[frame_idx].tolist(),
            "action": action[frame_idx].tolist(),
        }
        for hdf5_key, lerobot_key in zip(cam_hdf5_keys, cam_lerobot_keys):
            rotate = hdf5_key in WRIST_ROTATE_KEYS
            row[lerobot_key] = encode_image_jpeg(cam_data[hdf5_key][frame_idx], rotate_cw90=rotate)
        rows.append(row)

    # 에피소드 통계
    ep_from = global_offset
    ep_to = global_offset + n_frames
    timestamps = np.arange(n_frames, dtype=np.float32) / FPS
    frame_indices = np.arange(n_frames, dtype=np.int64)

    ep_meta = {
        "episode_index": ep_idx,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": ep_from,
        "dataset_to_index": ep_to,
        "tasks": [task_desc],
        "length": n_frames,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }

    for feat_name, feat_data in [("observation.state", q_pos), ("action", action)]:
        ep_meta[f"stats/{feat_name}/min"] = feat_data.min(axis=0).tolist()
        ep_meta[f"stats/{feat_name}/max"] = feat_data.max(axis=0).tolist()
        ep_meta[f"stats/{feat_name}/mean"] = feat_data.mean(axis=0).tolist()
        ep_meta[f"stats/{feat_name}/std"] = feat_data.std(axis=0).tolist()
        ep_meta[f"stats/{feat_name}/count"] = [n_frames]

    ep_meta["stats/timestamp/min"] = [float(timestamps.min())]
    ep_meta["stats/timestamp/max"] = [float(timestamps.max())]
    ep_meta["stats/timestamp/mean"] = [float(timestamps.mean())]
    ep_meta["stats/timestamp/std"] = [float(timestamps.std())]
    ep_meta["stats/timestamp/count"] = [n_frames]

    ep_meta["stats/frame_index/min"] = [0]
    ep_meta["stats/frame_index/max"] = [n_frames - 1]
    ep_meta["stats/frame_index/mean"] = [float(frame_indices.mean())]
    ep_meta["stats/frame_index/std"] = [float(frame_indices.std())]
    ep_meta["stats/frame_index/count"] = [n_frames]

    ep_meta["stats/episode_index/min"] = [ep_idx]
    ep_meta["stats/episode_index/max"] = [ep_idx]
    ep_meta["stats/episode_index/mean"] = [float(ep_idx)]
    ep_meta["stats/episode_index/std"] = [0.0]
    ep_meta["stats/episode_index/count"] = [n_frames]

    ep_meta["stats/index/min"] = [ep_from]
    ep_meta["stats/index/max"] = [ep_to - 1]
    ep_meta["stats/index/mean"] = [float((ep_from + ep_to - 1) / 2)]
    ep_meta["stats/index/std"] = [float(np.arange(ep_from, ep_to).std())]
    ep_meta["stats/index/count"] = [n_frames]

    ep_meta["stats/task_index/min"] = [task_idx]
    ep_meta["stats/task_index/max"] = [task_idx]
    ep_meta["stats/task_index/mean"] = [float(task_idx)]
    ep_meta["stats/task_index/std"] = [0.0]
    ep_meta["stats/task_index/count"] = [n_frames]

    img_stats = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [n_frames],
    }
    for cam_key in cam_lerobot_keys:
        for stat_name in ["min", "max", "mean", "std", "count"]:
            ep_meta[f"stats/{cam_key}/{stat_name}"] = img_stats[stat_name]

    return ep_idx, rows, ep_meta, q_pos, action, n_frames


def main():
    print("=" * 60)
    print("HDF5 → LeRobot v3.0 변환 (멀티프로세스 + JPEG)")
    print("=" * 60)

    hdf5_files = sorted(
        [f for f in glob.glob(os.path.join(HDF5_DIR, "*.hdf5"))
         if EPISODE_NUM_RANGE[0] <= int(re.search(r'\d+', os.path.basename(f)).group()) <= EPISODE_NUM_RANGE[1]],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
    )
    n_eps = len(hdf5_files)
    print(f"Found {n_eps} HDF5 files, using {NUM_WORKERS} workers")

    # 첫 파일에서 구조 파악
    with h5py.File(hdf5_files[0], "r") as f:
        state_dim = f["q_pos"].shape[1]
        action_dim = f["action"].shape[1]
        first_cam = list(CAMERA_MAPPING.keys())[0]
        img_shape = list(f[first_cam].shape[1:])
        n_frames_per_ep = f["action"].shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Image: {img_shape}")
        print(f"Frames per episode: {n_frames_per_ep}")

    os.makedirs(f"{OUTPUT_DIR}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/meta/episodes/chunk-000", exist_ok=True)

    # global offset 계산 (에피소드별 프레임 수를 먼저 읽음)
    ep_lengths = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            ep_lengths.append(f["action"].shape[0])

    global_offsets = [0]
    for length in ep_lengths[:-1]:
        global_offsets.append(global_offsets[-1] + length)
    total_frames = sum(ep_lengths)

    # 멀티프로세스로 에피소드 처리
    tasks = [(ep_idx, hdf5_path, global_offsets[ep_idx])
             for ep_idx, hdf5_path in enumerate(hdf5_files)]

    results = [None] * n_eps
    all_states = []
    all_actions = []
    episode_metas = []

    print(f"\nProcessing {n_eps} episodes ({total_frames} total frames)...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_episode, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=n_eps, desc="Processing"):
            ep_idx, rows, ep_meta, states, actions, n_frames = future.result()
            results[ep_idx] = rows
            episode_metas.append(ep_meta)
            all_states.append(states)
            all_actions.append(actions)

    # 에피소드 순서대로 정렬
    episode_metas.sort(key=lambda x: x["episode_index"])

    # ========== Data parquet 저장 (에피소드 단위로 파일 분할) ==========
    print("\nWriting data parquet files...")
    all_rows = []
    for ep_idx in range(n_eps):
        all_rows.extend(results[ep_idx])
        results[ep_idx] = None  # 메모리 해제

    # smol-libero 처럼 ~100MB 단위로 나눔
    MAX_ROWS_PER_FILE = 800
    file_idx = 0
    for start in range(0, len(all_rows), MAX_ROWS_PER_FILE):
        chunk = all_rows[start:start + MAX_ROWS_PER_FILE]
        df = pd.DataFrame(chunk)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f"{OUTPUT_DIR}/data/chunk-000/file-{file_idx:03d}.parquet")

        for em in episode_metas:
            if start <= em["dataset_from_index"] < start + MAX_ROWS_PER_FILE:
                em["data/file_index"] = file_idx

        file_idx += 1
        print(f"  file-{file_idx-1:03d}.parquet ({len(chunk)} rows)")

    all_rows = None  # 메모리 해제

    # ========== Episodes metadata ==========
    print("Writing episodes metadata...")
    ep_df = pd.DataFrame(episode_metas)
    pq.write_table(pa.Table.from_pandas(ep_df),
                    f"{OUTPUT_DIR}/meta/episodes/chunk-000/file-000.parquet")

    # ========== Tasks ==========
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(TASK_DESCRIPTIONS)))},
        index=pd.Index(TASK_DESCRIPTIONS, name="task"),
    )
    pq.write_table(pa.Table.from_pandas(tasks_df), f"{OUTPUT_DIR}/meta/tasks.parquet")

    # ========== Global stats.json ==========
    print("Computing global stats...")
    all_states_np = np.concatenate(all_states, axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)

    stats = {}
    for name, data in [("observation.state", all_states_np), ("action", all_actions_np)]:
        stats[name] = {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "count": [total_frames],
        }

    for cam_key in CAMERA_MAPPING.values():
        stats[cam_key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [total_frames],
        }

    all_indices = np.arange(total_frames)
    stats["timestamp"] = {
        "min": [0.0], "max": [float((max(ep_lengths) - 1) / FPS)],
        "mean": [float(np.mean(ep_lengths) / 2 / FPS)],
        "std": [float(np.std(np.arange(max(ep_lengths), dtype=np.float32) / FPS))],
        "count": [total_frames],
    }
    stats["frame_index"] = {
        "min": [0], "max": [max(ep_lengths) - 1],
        "mean": [float(np.mean(all_indices))], "std": [float(np.std(all_indices))],
        "count": [total_frames],
    }
    stats["episode_index"] = {
        "min": [0], "max": [n_eps - 1],
        "mean": [float((n_eps - 1) / 2)], "std": [float(np.arange(n_eps).std())],
        "count": [total_frames],
    }
    stats["index"] = {
        "min": [0], "max": [total_frames - 1],
        "mean": [float((total_frames - 1) / 2)], "std": [float(all_indices.std())],
        "count": [total_frames],
    }
    n_tasks = len(TASK_DESCRIPTIONS)
    stats["task_index"] = {
        "min": [0], "max": [n_tasks - 1],
        "mean": [float((n_tasks - 1) / 2)], "std": [float(np.arange(n_tasks).std())],
        "count": [total_frames],
    }

    with open(f"{OUTPUT_DIR}/meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # ========== info.json ==========
    cam_features = {}
    for cam_key in CAMERA_MAPPING.values():
        cam_features[cam_key] = {
            "dtype": "image", "shape": img_shape,
            "names": ["height", "width", "channel"], "fps": FPS,
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(TASK_DESCRIPTIONS),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            **cam_features,
            "observation.state": {
                "dtype": "float32", "shape": [state_dim],
                "names": [f"joint_{i}" for i in range(state_dim)], "fps": FPS,
            },
            "action": {
                "dtype": "float32", "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)], "fps": FPS,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        },
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
    }

    with open(f"{OUTPUT_DIR}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"변환 완료!")
    print(f"총 에피소드: {n_eps}, 총 프레임: {total_frames}")
    print(f"데이터셋 경로: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
