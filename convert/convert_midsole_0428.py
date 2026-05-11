"""
HDF5 → LeRobot v3.0 변환 스크립트 (midsole_rotating_0428)
OpenArm 16 DOF, 카메라 3대 (cam_top / cam_wrist_left / cam_wrist_right)
이미지: raw uint8 640x480 → 320x240 JPEG 인코딩
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
HDF5_DIR = "/media/youngwoo/OPR_LINUX/shoes_demo/midsole_rotating_0428"
OUTPUT_DIR = os.path.expanduser("~/midsole_rotating_0428_smolvla")
FPS = 30
ROBOT_TYPE = "openarm"
NUM_WORKERS = 8

EPISODE_NUM_RANGE = (0, 49)

EPISODE_TASK_MAP = [
    (0, 49, "rotate midsole"),
]
TASK_DESCRIPTIONS = [desc for _, _, desc in EPISODE_TASK_MAP]

def get_task_index(ep_idx: int) -> int:
    for task_idx, (start, end, _) in enumerate(EPISODE_TASK_MAP):
        if start <= ep_idx <= end:
            return task_idx
    raise ValueError(f"ep_idx={ep_idx} not in EPISODE_TASK_MAP")

# cam_center는 blank(all zeros)이므로 제외
CAMERA_MAPPING = {
    "images/cam_top":         "observation.images.camera1",
    "images/cam_wrist_left":  "observation.images.camera2",
    "images/cam_wrist_right": "observation.images.camera3",
}

RESIZE_WH = (320, 240)
IMAGE_SHAPE = [240, 320, 3]  # HxWxC
# ============================================================
def encode_jpeg(raw_uint8: np.ndarray) -> dict:
    """raw uint8 (H,W,C) → 320x240 JPEG bytes"""
    img = Image.fromarray(raw_uint8)
    img = img.resize(RESIZE_WH, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return {"bytes": buf.getvalue(), "path": None}

def process_single_episode(args):
    ep_idx, hdf5_path, global_offset = args
    task_idx = get_task_index(ep_idx)
    task_desc = TASK_DESCRIPTIONS[task_idx]
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
    print("HDF5 → LeRobot v3.0 변환 (midsole_rotating_0428)")
    print("=" * 60)

    hdf5_files = sorted(
        [f for f in glob.glob(os.path.join(HDF5_DIR, "*.hdf5"))
         if EPISODE_NUM_RANGE[0] <= int(re.search(r'\d+', os.path.basename(f)).group()) <= EPISODE_NUM_RANGE[1]],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
    )
    n_eps = len(hdf5_files)
    print(f"Found {n_eps} HDF5 files")

    with h5py.File(hdf5_files[0], "r") as f:
        state_dim = f["q_pos"].shape[1]
        action_dim = f["action"].shape[1]
        n_frames_first = f["action"].shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Frames/ep: {n_frames_first}")
        print(f"Camera mapping: {CAMERA_MAPPING}")

    os.makedirs(f"{OUTPUT_DIR}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/meta/episodes/chunk-000", exist_ok=True)

    ep_lengths = []
    for p in hdf5_files:
        with h5py.File(p, "r") as f:
            ep_lengths.append(f["action"].shape[0])

    global_offsets = [0]
    for l in ep_lengths[:-1]:
        global_offsets.append(global_offsets[-1] + l)
    total_frames = sum(ep_lengths)

    tasks_args = [(ep_idx, hdf5_path, global_offsets[ep_idx])
                  for ep_idx, hdf5_path in enumerate(hdf5_files)]

    results = [None] * n_eps
    all_states, all_actions, episode_metas = [], [], []

    print(f"\nProcessing {n_eps} episodes ({total_frames} total frames)...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_episode, t): t[0] for t in tasks_args}
        for future in tqdm(as_completed(futures), total=n_eps, desc="Processing"):
            ep_idx, rows, ep_meta, states, actions, _ = future.result()
            results[ep_idx] = rows
            episode_metas.append(ep_meta)
            all_states.append(states)
            all_actions.append(actions)

    episode_metas.sort(key=lambda x: x["episode_index"])

    print("\nWriting data parquet files...")
    all_rows = []
    for ep_idx in range(n_eps):
        all_rows.extend(results[ep_idx])
        results[ep_idx] = None

    MAX_ROWS_PER_FILE = 800
    file_idx = 0
    for start in range(0, len(all_rows), MAX_ROWS_PER_FILE):
        chunk = all_rows[start:start + MAX_ROWS_PER_FILE]
        df = pd.DataFrame(chunk)
        pq.write_table(pa.Table.from_pandas(df),
                        f"{OUTPUT_DIR}/data/chunk-000/file-{file_idx:03d}.parquet")
        for em in episode_metas:
            if start <= em["dataset_from_index"] < start + MAX_ROWS_PER_FILE:
                em["data/file_index"] = file_idx
        file_idx += 1
        print(f"  file-{file_idx-1:03d}.parquet ({len(chunk)} rows)")

    all_rows = None

    print("Writing metadata...")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(episode_metas)),
                    f"{OUTPUT_DIR}/meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(
            {"task_index": list(range(len(TASK_DESCRIPTIONS)))},
            index=pd.Index(TASK_DESCRIPTIONS, name="task"),
        )),
        f"{OUTPUT_DIR}/meta/tasks.parquet",
    )

    print("Computing global stats...")
    all_states_np = np.concatenate(all_states, axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)
    all_indices = np.arange(total_frames)

    stats = {}
    for name, data in [("observation.state", all_states_np), ("action", all_actions_np)]:
        stats[name] = {
            "min": data.min(axis=0).tolist(), "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(), "std": data.std(axis=0).tolist(),
            "count": [total_frames],
        }
    for cam_key in CAMERA_MAPPING.values():
        stats[cam_key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]], "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [total_frames],
        }
    stats["timestamp"] = {
        "min": [0.0], "max": [float((max(ep_lengths) - 1) / FPS)],
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
    n_tasks = len(TASK_DESCRIPTIONS)
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
            **{k: {"dtype": "image", "shape": IMAGE_SHAPE,
                   "names": ["height", "width", "channel"], "fps": FPS}
               for k in CAMERA_MAPPING.values()},
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
        "video_files_size_in_mb": 0,
    }
    with open(f"{OUTPUT_DIR}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"변환 완료! 에피소드: {n_eps}, 프레임: {total_frames}")
    print(f"출력: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
