import io, json, os, glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC_A = "/home/youngwoo/midsole_grasping_0427_smolvla"
SRC_B = "/home/youngwoo/midsole_rotating_0428_smolvla"
OUTPUT = "/home/youngwoo/midsole_grasp_rotate_0428"
TASK_DESCRIPTIONS = ["grasp midsole", "rotate midsole"]
MAX_ROWS = 800

os.makedirs(f"{OUTPUT}/data/chunk-000", exist_ok=True)
os.makedirs(f"{OUTPUT}/meta/episodes/chunk-000", exist_ok=True)

print("Loading dataset A (grasping_0427)...")
df_a = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SRC_A}/data/chunk-000/file-*.parquet"))], ignore_index=True)
print(f"  {len(df_a)} rows")

print("Loading dataset B (rotating_0428)...")
df_b = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SRC_B}/data/chunk-000/file-*.parquet"))], ignore_index=True)
print(f"  {len(df_b)} rows")

n_frames_a = len(df_a)
df_b["episode_index"] = df_b["episode_index"] + 50
df_b["index"]         = df_b["index"] + n_frames_a
df_b["task_index"]    = 1

df_all = pd.concat([df_a, df_b], ignore_index=True)
total_frames = len(df_all)
print(f"Total: {total_frames} rows")

print("\nWriting data parquet files...")
file_idx = 0
for start in range(0, total_frames, MAX_ROWS):
    chunk = df_all.iloc[start:start + MAX_ROWS]
    pq.write_table(pa.Table.from_pandas(chunk),
                   f"{OUTPUT}/data/chunk-000/file-{file_idx:03d}.parquet")
    print(f"  file-{file_idx:03d}.parquet ({len(chunk)} rows)")
    file_idx += 1

print("Writing episodes metadata...")
ep_a = pd.read_parquet(f"{SRC_A}/meta/episodes/chunk-000/file-000.parquet")
ep_b = pd.read_parquet(f"{SRC_B}/meta/episodes/chunk-000/file-000.parquet")

ep_b["episode_index"]            = ep_b["episode_index"] + 50
ep_b["dataset_from_index"]       = ep_b["dataset_from_index"] + n_frames_a
ep_b["dataset_to_index"]         = ep_b["dataset_to_index"]   + n_frames_a
ep_b["stats/index/min"]          = ep_b["stats/index/min"].apply(lambda x: [x[0] + n_frames_a])
ep_b["stats/index/max"]          = ep_b["stats/index/max"].apply(lambda x: [x[0] + n_frames_a])
ep_b["stats/index/mean"]         = ep_b["stats/index/mean"].apply(lambda x: [x[0] + n_frames_a])
ep_b["stats/episode_index/min"]  = ep_b["stats/episode_index/min"].apply(lambda x: [x[0] + 50])
ep_b["stats/episode_index/max"]  = ep_b["stats/episode_index/max"].apply(lambda x: [x[0] + 50])
ep_b["stats/episode_index/mean"] = ep_b["stats/episode_index/mean"].apply(lambda x: [x[0] + 50])
ep_b["stats/task_index/min"]     = [[1]] * len(ep_b)
ep_b["stats/task_index/max"]     = [[1]] * len(ep_b)
ep_b["stats/task_index/mean"]    = [[1.0]] * len(ep_b)

ep_all = pd.concat([ep_a, ep_b], ignore_index=True).sort_values("episode_index").reset_index(drop=True)
for i, row in ep_all.iterrows():
    ep_all.at[i, "data/file_index"] = int(row["dataset_from_index"]) // MAX_ROWS

pq.write_table(pa.Table.from_pandas(ep_all),
               f"{OUTPUT}/meta/episodes/chunk-000/file-000.parquet")

pq.write_table(
    pa.Table.from_pandas(pd.DataFrame(
        {"task_index": [0, 1]},
        index=pd.Index(TASK_DESCRIPTIONS, name="task"),
    )),
    f"{OUTPUT}/meta/tasks.parquet",
)

print("Computing global stats...")
states_np  = np.concatenate([np.array(df_a["observation.state"].tolist(), dtype=np.float32),
                              np.array(df_b["observation.state"].tolist(), dtype=np.float32)])
actions_np = np.concatenate([np.array(df_a["action"].tolist(), dtype=np.float32),
                              np.array(df_b["action"].tolist(), dtype=np.float32)])
all_indices = np.arange(total_frames)

stats = {}
for name, data in [("observation.state", states_np), ("action", actions_np)]:
    stats[name] = {"min": data.min(axis=0).tolist(), "max": data.max(axis=0).tolist(),
                   "mean": data.mean(axis=0).tolist(), "std": data.std(axis=0).tolist(),
                   "count": [total_frames]}
for cam in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
    stats[cam] = {"min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]],
                  "mean": [[[0.5]], [[0.5]], [[0.5]]], "std": [[[0.25]], [[0.25]], [[0.25]]],
                  "count": [total_frames]}
stats["timestamp"]     = {"min": [0.0], "max": [float((600-1)/30)], "mean": [float(600/2/30)],
                          "std": [float(np.std(np.arange(600)/30))], "count": [total_frames]}
stats["frame_index"]   = {"min": [0], "max": [599], "mean": [float(all_indices.mean())],
                          "std": [float(all_indices.std())], "count": [total_frames]}
stats["episode_index"] = {"min": [0], "max": [99], "mean": [49.5],
                          "std": [float(np.arange(100).std())], "count": [total_frames]}
stats["index"]         = {"min": [0], "max": [total_frames-1], "mean": [float((total_frames-1)/2)],
                          "std": [float(all_indices.std())], "count": [total_frames]}
stats["task_index"]    = {"min": [0], "max": [1], "mean": [0.5], "std": [0.5], "count": [total_frames]}

with open(f"{OUTPUT}/meta/stats.json", "w") as f:
    json.dump(stats, f, indent=4)

state_dim  = states_np.shape[1]
action_dim = actions_np.shape[1]
info = {
    "codebase_version": "v3.0",
    "robot_type": "openarm",
    "total_episodes": 100,
    "total_frames": total_frames,
    "total_tasks": 2,
    "chunks_size": 1000,
    "fps": 30,
    "splits": {"train": "0:100"},
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": None,
    "features": {
        **{f"observation.images.camera{i+1}": {"dtype": "image", "shape": [240, 320, 3],
           "names": ["height", "width", "channel"], "fps": 30} for i in range(3)},
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": [f"joint_{i}" for i in range(state_dim)], "fps": 30},
        "action": {"dtype": "float32", "shape": [action_dim],
                   "names": [f"action_{i}" for i in range(action_dim)], "fps": 30},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None, "fps": 30},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None, "fps": 30},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None, "fps": 30},
        "index":         {"dtype": "int64",   "shape": [1], "names": None, "fps": 30},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None, "fps": 30},
    },
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 0,
}
with open(f"{OUTPUT}/meta/info.json", "w") as f:
    json.dump(info, f, indent=4)

print(f"\n완료! 총 에피소드: 100, 총 프레임: {total_frames}")
print(f"  - grasp midsole (ep 0~49):  {n_frames_a} frames")
print(f"  - rotate midsole (ep 50~99): {total_frames - n_frames_a} frames")
print(f"출력: {OUTPUT}")
