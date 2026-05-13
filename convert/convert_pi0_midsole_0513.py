"""
HDF5 → LeRobot v3.0 변환 스크립트 (midsole pi0/pi0fast, 0513)

Sources:
  - midsole_grasping_0512 (40 eps, 300 frames)  → task: "grasp the midsole"
  - midsole_buffing_0512  (40 eps, 600 frames)  → task: "buff the midsole"

Excluded: cam_center (all zeros), f_ext_L, f_ext_R, ee_TF, reference_ee_TF
Images: 640x480 → 320x240 JPEG (quality 90)
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
# 설정
# ============================================================
SOURCES = [
    {
        "hdf5_dir": "/media/youngwoo/OPR_LINUX/shoes_demo/midsole_grasping_0512",
        "task": "grasp the midsole",
    },
    {
        "hdf5_dir": "/media/youngwoo/OPR_LINUX/shoes_demo/midsole_buffing_0512",
        "task": "buffing the midsole",
    },
]

OUTPUT_DIR = os.path.expanduser("~/midsole_pi0_0513")
FPS = 30
ROBOT_TYPE = "openarm"
NUM_WORKERS = 8

# cam_center는 all-zero이므로 제외
CAMERA_MAPPING = {
    "images/cam_top":         "observation.images.cam_top",
    "images/cam_wrist_left":  "observation.images.cam_wrist_left",
    "images/cam_wrist_right": "observation.images.cam_wrist_right",
}

RESIZE_WH = (320, 240)   # (width, height)
IMAGE_SHAPE = [240, 320, 3]  # HxWxC
JPEG_QUALITY = 90
# ============================================================


def encode_jpeg(raw_uint8: np.ndarray) -> dict:
    img = Image.fromarray(raw_uint8)
    img = img.resize(RESIZE_WH, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return {"bytes": buf.getvalue(), "path": None}


def process_single_episode(args):
    ep_idx, hdf5_path, global_offset, task_idx, task_desc = args

    cam_hdf5_keys = list(CAMERA_MAPPING.keys())
    cam_lerobot_keys = list(CAMERA_MAPPING.values())

    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["action"].shape[0]
        q_pos = f["q_pos"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)
        cam_data = {k: f[k][:] for k in cam_hdf5_keys}

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
            row[lerobot_key] = encode_jpeg(cam_data[hdf5_key][frame_idx])
        rows.append(row)

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

    for scalar_name, arr in [
        ("timestamp", timestamps),
        ("frame_index", frame_indices.astype(np.float32)),
    ]:
        ep_meta[f"stats/{scalar_name}/min"] = [float(arr.min())]
        ep_meta[f"stats/{scalar_name}/max"] = [float(arr.max())]
        ep_meta[f"stats/{scalar_name}/mean"] = [float(arr.mean())]
        ep_meta[f"stats/{scalar_name}/std"] = [float(arr.std())]
        ep_meta[f"stats/{scalar_name}/count"] = [n_frames]

    for scalar_name, val in [("episode_index", ep_idx), ("task_index", task_idx)]:
        ep_meta[f"stats/{scalar_name}/min"] = [val]
        ep_meta[f"stats/{scalar_name}/max"] = [val]
        ep_meta[f"stats/{scalar_name}/mean"] = [float(val)]
        ep_meta[f"stats/{scalar_name}/std"] = [0.0]
        ep_meta[f"stats/{scalar_name}/count"] = [n_frames]

    ep_meta["stats/index/min"] = [ep_from]
    ep_meta["stats/index/max"] = [ep_to - 1]
    ep_meta["stats/index/mean"] = [float((ep_from + ep_to - 1) / 2)]
    ep_meta["stats/index/std"] = [float(np.arange(ep_from, ep_to).std())]
    ep_meta["stats/index/count"] = [n_frames]

    img_stats = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [n_frames],
    }
    for cam_key in cam_lerobot_keys:
        for stat_name, val in img_stats.items():
            ep_meta[f"stats/{cam_key}/{stat_name}"] = val

    return ep_idx, rows, ep_meta, q_pos, action, n_frames


def collect_episodes():
    """두 소스 디렉토리에서 에피소드 목록과 task 정보를 수집."""
    task_descriptions = [s["task"] for s in SOURCES]
    episodes = []  # (global_ep_idx, hdf5_path, task_idx)

    global_ep = 0
    for task_idx, source in enumerate(SOURCES):
        hdf5_files = sorted(
            glob.glob(os.path.join(source["hdf5_dir"], "*.hdf5")),
            key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()),
        )
        for path in hdf5_files:
            episodes.append((global_ep, path, task_idx))
            global_ep += 1

    return episodes, task_descriptions


def main():
    print("=" * 60)
    print("HDF5 → LeRobot v3.0 변환 (midsole pi0, 0513)")
    print("=" * 60)

    episodes, task_descriptions = collect_episodes()
    n_eps = len(episodes)
    print(f"총 에피소드: {n_eps}  (태스크: {task_descriptions})")

    # 에피소드별 프레임 수 및 global offset 계산
    ep_lengths = []
    for _, hdf5_path, _ in episodes:
        with h5py.File(hdf5_path, "r") as f:
            ep_lengths.append(f["action"].shape[0])

    global_offsets = [0]
    for l in ep_lengths[:-1]:
        global_offsets.append(global_offsets[-1] + l)
    total_frames = sum(ep_lengths)
    print(f"총 프레임: {total_frames}  (grasping: {sum(ep_lengths[:40])}, buffing: {sum(ep_lengths[40:])})")

    with h5py.File(episodes[0][1], "r") as f:
        state_dim = f["q_pos"].shape[1]
        action_dim = f["action"].shape[1]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Camera: {list(CAMERA_MAPPING.values())}")
    print(f"Image: 640x480 → {RESIZE_WH[0]}x{RESIZE_WH[1]} JPEG q{JPEG_QUALITY}")

    os.makedirs(f"{OUTPUT_DIR}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/meta/episodes/chunk-000", exist_ok=True)

    tasks_args = [
        (ep_idx, hdf5_path, global_offsets[i], task_idx, task_descriptions[task_idx])
        for i, (ep_idx, hdf5_path, task_idx) in enumerate(episodes)
    ]

    results = [None] * n_eps
    all_states, all_actions, episode_metas = [], [], []

    print(f"\n에피소드 처리 중 (workers={NUM_WORKERS})...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_episode, t): t[0] for t in tasks_args}
        for future in tqdm(as_completed(futures), total=n_eps, desc="Converting"):
            ep_idx, rows, ep_meta, states, actions, _ = future.result()
            results[ep_idx] = rows
            episode_metas.append(ep_meta)
            all_states.append(states)
            all_actions.append(actions)

    episode_metas.sort(key=lambda x: x["episode_index"])

    print("\nParquet 데이터 파일 작성 중...")
    all_rows = []
    for ep_idx in range(n_eps):
        all_rows.extend(results[ep_idx])
        results[ep_idx] = None

    MAX_ROWS_PER_FILE = 800
    file_idx = 0
    file_ep_map = {}  # ep_from → file_idx
    for start in range(0, len(all_rows), MAX_ROWS_PER_FILE):
        chunk = all_rows[start : start + MAX_ROWS_PER_FILE]
        df = pd.DataFrame(chunk)
        pq.write_table(
            pa.Table.from_pandas(df),
            f"{OUTPUT_DIR}/data/chunk-000/file-{file_idx:03d}.parquet",
        )
        for em in episode_metas:
            if start <= em["dataset_from_index"] < start + MAX_ROWS_PER_FILE:
                em["data/file_index"] = file_idx
        print(f"  file-{file_idx:03d}.parquet ({len(chunk)} rows)")
        file_idx += 1

    all_rows = None

    print("메타데이터 작성 중...")
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(episode_metas)),
        f"{OUTPUT_DIR}/meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {"task_index": list(range(len(task_descriptions)))},
                index=pd.Index(task_descriptions, name="task"),
            )
        ),
        f"{OUTPUT_DIR}/meta/tasks.parquet",
    )

    print("글로벌 통계 계산 중...")
    all_states_np = np.concatenate(all_states, axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)
    all_indices = np.arange(total_frames)
    n_tasks = len(task_descriptions)

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
    stats["timestamp"] = {
        "min": [0.0],
        "max": [float((max(ep_lengths) - 1) / FPS)],
        "mean": [float(np.mean(ep_lengths) / 2 / FPS)],
        "std": [float(np.std(np.arange(max(ep_lengths), dtype=np.float32) / FPS))],
        "count": [total_frames],
    }
    stats["frame_index"] = {
        "min": [0], "max": [max(ep_lengths) - 1],
        "mean": [float(all_indices.mean())], "std": [float(all_indices.std())],
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
    stats["task_index"] = {
        "min": [0], "max": [n_tasks - 1],
        "mean": [float((n_tasks - 1) / 2)],
        "std": [float(np.arange(n_tasks).std()) if n_tasks > 1 else 0.0],
        "count": [total_frames],
    }

    with open(f"{OUTPUT_DIR}/meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            **{
                k: {
                    "dtype": "image",
                    "shape": IMAGE_SHAPE,
                    "names": ["height", "width", "channel"],
                    "fps": FPS,
                }
                for k in CAMERA_MAPPING.values()
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"joint_{i}" for i in range(state_dim)],
                "fps": FPS,
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)],
                "fps": FPS,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        },
        "data_files_size_in_mb": 0,
        "video_files_size_in_mb": 0,
    }
    with open(f"{OUTPUT_DIR}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"변환 완료!")
    print(f"  에피소드: {n_eps} (grasping: 40, buffing: 40)")
    print(f"  프레임:   {total_frames}")
    print(f"  출력:     {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
