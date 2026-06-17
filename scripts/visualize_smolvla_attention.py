#!/usr/bin/env python
"""Visualize how much a trained SmolVLA policy attends to the current observation.

Runs inference (`predict_action_chunk`) on dataset frames while capturing the softmax
attention probabilities inside the model, then aggregates the attention that the
action-expert tokens pay to each part of the prefix (camera images, language, state,
wrench/force).

NOTE: SmolVLA only ever sees the CURRENT observation (n_obs_steps=1, no history), so
attention is always per-frame. Use --sweep / --video to scan a whole episode and see
how the attention distribution (e.g. on the force/wrench token) evolves over time.

Requires the [ATTN-VIZ] capture hook in
`src/lerobot/policies/smolvla/smolvlm_with_expert.py` (`attn_capture` attribute).

How captures are interpreted (see SmolVLMWithExpertModel.forward):
  - Prefix pass (fill_kv_cache=True): all `num_vlm_layers` layers run prefix
    self-attention -> first N captures have q_len == k_len == prefix_len.
  - Each denoise step (num_steps of them) runs all layers again with the chunk_size
    action tokens as queries:
      * even layers (self-attn): keys = [cached prefix + action tokens]
        -> capture shape (B, H, chunk, prefix_len + chunk)
      * odd layers (cross-attn): keys = cached prefix only
        -> capture shape (B, H, chunk, prefix_len)
  - Prefix token layout (embed_prefix, add_image_special_tokens=False):
      [cam1 img tokens][cam2]...[camN][language tokens][state][wrench?]

Modes:
  single frame (--episode E --frame F):
    attention_overlay.png / attention_groups.png / attention_dynamics.png
    attention_raw.npz / summary.txt
  sweep (--episode E --sweep [--stride K]):
    attention_timeline.png / attention_sweep.npz / summary.txt
  video (--episode E --video [--stride K] [--fps N]): everything from sweep, plus
    attention_video.mp4 — per-timestep camera attention overlays + current group
    shares (images/language/state/wrench) + timeline cursor, playable/scrubbable.
  ablation (--ablate wrench|state|cameraN, combinable with any mode above):
    runs a second inference per frame with that input replaced by 'no information'
    (same flow-matching noise) and reports |Δaction| — i.e. how much the input
    CAUSALLY contributes to the output, complementing the attention (correlation)
    view. Adds ablation_timeline.png / ablation_joints.png / ablation.npz.

Examples:
  uv run python scripts/visualize_smolvla_attention.py \
      --checkpoint outputs/train/.../checkpoints/last/pretrained_model \
      --episode 0 --frame 30

  uv run python scripts/visualize_smolvla_attention.py \
      --checkpoint outputs/train/.../checkpoints/last/pretrained_model \
      --episode 0 --video --stride 5
"""

import argparse
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE, OBS_WRENCH


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a pretrained_model dir (contains config.json, model.safetensors, policy_preprocessor.json)",
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root (default: from train_config.json)")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo_id (default: from train_config.json)")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Analyze a raw HDF5 episode file directly (instead of a LeRobot dataset). "
        "Expects datasets: action, q_pos (state), f_ext_L (wrench), images/<cam> (JPEG-encoded). "
        "When set, --episode/--repo-id/--dataset-root are ignored.",
    )
    parser.add_argument(
        "--hdf5-camera", type=str, default="cam_wrist_left",
        help="Which HDF5 images/<name> maps to observation.images.camera1 (default: cam_wrist_left)",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--frame", type=int, default=0, help="Frame index within the episode (single-frame mode)")
    parser.add_argument("--sweep", action="store_true", help="Analyze the whole episode instead of one frame")
    parser.add_argument("--video", action="store_true", help="Like --sweep, but also render attention_video.mp4")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride in sweep/video mode")
    parser.add_argument("--fps", type=float, default=None, help="Video fps (default: dataset fps / stride = real-time)")
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        help="Causal check: run a second inference with this input replaced by 'no information' "
        "(zeros in normalized space = dataset mean) and report how much the actions change. "
        "Targets: 'wrench', 'state', or a camera name like 'camera1' (blacked out). "
        "Both inferences share the same flow-matching start noise (--seed) so the action "
        "difference is caused only by the ablated input.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fixed flow-matching noise (--ablate)")
    parser.add_argument("--task", type=str, default=None, help="Override task instruction (default: from dataset)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: outputs/attention_viz_<MMDD>/ep<E>_<mode>)",
    )
    return parser.parse_args()


def checkpoint_tag(checkpoint: Path) -> str:
    """Short id for the default output dir so runs with different checkpoints don't overwrite
    each other: .../outputs/train/<run>/checkpoints/<step>/pretrained_model -> '<run>_<step>'."""
    try:
        if checkpoint.name == "pretrained_model" and checkpoint.parents[1].name == "checkpoints":
            return f"{checkpoint.parents[2].name}_{checkpoint.parents[0].name}"
    except IndexError:
        pass
    return checkpoint.name


def load_dataset_info_from_train_config(checkpoint: Path) -> tuple[str | None, str | None]:
    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.exists():
        return None, None
    with open(train_config_path) as f:
        cfg = json.load(f)
    ds = cfg.get("dataset", {})
    return ds.get("repo_id"), ds.get("root")


class HDF5Episode:
    """Minimal LeRobotDataset-like view over one raw HDF5 episode file, so build_batch and
    the rest of the pipeline work unchanged. Maps:
      images/<hdf5_camera> -> observation.images.camera1  (JPEG decoded -> CHW float [0,1])
      q_pos                -> observation.state
      f_ext_L              -> observation.wrench
      action               -> action (ground truth, for reference)
    """

    def __init__(self, path: Path, camera: str, task: str, fps: int = 30):
        import h5py

        self._f = h5py.File(str(path), "r")
        self._cam_key = f"images/{camera}"
        assert self._cam_key in self._f, (
            f"camera '{camera}' not in HDF5 (available: {list(self._f['images'].keys())})"
        )
        self.num_frames = self._f["action"].shape[0]
        self.fps = fps
        self.task = task

    def _decode_img(self, raw) -> torch.Tensor:
        import io

        from PIL import Image

        b = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
        arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW, [0,1]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i: int) -> dict:
        f = self._f
        return {
            "observation.images.camera1": self._decode_img(f[self._cam_key][i]),
            "observation.state": torch.from_numpy(f["q_pos"][i].astype(np.float32)),
            "observation.wrench": torch.from_numpy(f["f_ext_L"][i].astype(np.float32)),
            "action": torch.from_numpy(f["action"][i].astype(np.float32)),
            "task": self.task,
        }


def build_batch(
    dataset, frame_idx: int, task_override: str | None, input_keys: set | None = None
) -> tuple[dict, str]:
    """Turn one dataset item into a batch of size 1 with only observation keys + task.

    `input_keys` (a model's config.input_features) filters the observation keys so a model is
    only fed what it was trained on — e.g. the no-force checkpoint never receives wrench."""
    item = dataset[frame_idx]
    batch = {}
    for key, value in item.items():
        if key.startswith("observation."):
            if input_keys is not None and key not in input_keys:
                continue
            batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
    task = task_override if task_override is not None else item.get("task", "")
    batch["task"] = [task]
    return batch, task


def ablate_batch_key(batch: dict, target: str) -> dict:
    """Return a copy of the (already preprocessed) batch with `target` replaced by zeros.
    Zero in normalized space = dataset mean for state/wrench ('typical, uninformative' value);
    for cameras it is a black frame."""
    if target == "wrench":
        key = OBS_WRENCH
    elif target == "state":
        key = OBS_STATE
    else:
        key = f"observation.images.{target}"
    assert key in batch, f"--ablate target '{target}' not in batch (keys: {list(batch.keys())})"
    ablated = dict(batch)
    ablated[key] = torch.zeros_like(batch[key])
    return ablated


def analyze_frame(
    policy,
    preprocessor,
    dataset,
    frame_idx: int,
    task_override: str | None,
    ablate: str | None = None,
    noise: torch.Tensor | None = None,
    postprocessor=None,
) -> dict:
    """Run one inference on `frame_idx` with attention capture and aggregate the result.

    With `ablate`, runs a second inference on the ablated batch using the SAME flow-matching
    noise, so the action difference is caused only by the ablated input.

    Returns a dict with:
      groups            : list of (name, start, end) prefix token spans
      cross_attn        : (num_steps, n_cross_layers, prefix_len) head/query-averaged attention
      self_prefix_mass  : (num_steps, n_self_layers) prefix share of self-attn layers
      mean_cross        : (prefix_len,) mean over steps and cross layers
      shares            : {group: share} from mean_cross (sums to 1)
      raw_images, task, layout info ...
    """
    batch, task = build_batch(dataset, frame_idx, task_override, input_keys=set(policy.config.input_features))
    camera_keys = [k for k in policy.config.image_features if k in batch]
    raw_images = {k: batch[k].clone() for k in camera_keys}  # keep [0,1] images for overlays
    wrench_raw = batch[OBS_WRENCH][0].clone() if OBS_WRENCH in batch else None

    batch = preprocessor(batch)

    vwe = policy.model.vlm_with_expert
    num_layers = vwe.num_vlm_layers
    num_steps = policy.config.num_steps
    chunk = policy.config.chunk_size

    vwe.attn_capture = []
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch, noise=noise)
    captures = vwe.attn_capture
    vwe.attn_capture = None

    # Ablation: second inference (no capture) with the target input zeroed, same start noise
    ablation = None
    if ablate is not None:
        assert noise is not None, "ablation needs a fixed noise tensor"
        with torch.no_grad():
            actions_ablated = policy.predict_action_chunk(ablate_batch_key(batch, ablate), noise=noise)
        a, b = actions, actions_ablated
        if postprocessor is not None:  # compare in unnormalized (robot) units
            a, b = postprocessor(a), postprocessor(b)
        diff = (a - b).abs()  # (1, chunk, action_dim)
        ablation = {
            "target": ablate,
            "diff_mean": float(diff.mean().item()),
            "diff_max": float(diff.max().item()),
            "diff_per_joint": diff.mean(dim=(0, 1)).cpu().numpy(),
        }

    expected = num_layers * (1 + num_steps)
    assert len(captures) == expected, f"expected {expected} attention captures, got {len(captures)}"
    prefix_len = captures[0].shape[-1]

    # Prefix token layout
    n_lang = batch[OBS_LANGUAGE_TOKENS].shape[1]
    n_lang_real = int(batch[OBS_LANGUAGE_ATTENTION_MASK][0].sum().item())
    has_wrench = OBS_WRENCH in batch
    n_extra = 1 + int(has_wrench)  # state + optional wrench
    n_cams = len(camera_keys)
    n_img = (prefix_len - n_lang - n_extra) // n_cams
    assert n_img * n_cams + n_lang + n_extra == prefix_len, (
        f"prefix layout mismatch: prefix_len={prefix_len}, n_lang={n_lang}, n_cams={n_cams}, n_extra={n_extra}"
    )
    grid = int(round(n_img**0.5))
    assert grid * grid == n_img, f"image tokens per camera ({n_img}) is not a square grid"

    groups = []
    pos = 0
    for key in camera_keys:
        groups.append((key.removeprefix("observation.images."), pos, pos + n_img))
        pos += n_img
    groups.append(("language", pos, pos + n_lang))
    pos += n_lang
    groups.append(("state", pos, pos + 1))
    pos += 1
    if has_wrench:
        groups.append(("wrench", pos, pos + 1))

    cross_layers = [
        i for i in range(num_layers)
        if not (vwe.self_attn_every_n_layers > 0 and i % vwe.self_attn_every_n_layers == 0)
    ]
    self_layers = [i for i in range(num_layers) if i not in cross_layers]

    cross_attn = np.zeros((num_steps, len(cross_layers), prefix_len), dtype=np.float64)
    self_prefix_mass = np.zeros((num_steps, len(self_layers)), dtype=np.float64)

    for s in range(num_steps):
        step_caps = captures[num_layers * (1 + s) : num_layers * (2 + s)]
        for layer_idx, cap in enumerate(step_caps):
            assert cap.shape[2] == chunk, f"step {s} layer {layer_idx}: q_len {cap.shape[2]} != chunk {chunk}"
            attn = cap[0].mean(dim=0).mean(dim=0).numpy()  # mean over heads, then over action-token queries
            if layer_idx in cross_layers:  # keys = prefix only, sums to 1 over prefix
                assert cap.shape[-1] == prefix_len
                cross_attn[s, cross_layers.index(layer_idx)] = attn
            else:  # keys = prefix + action tokens
                assert cap.shape[-1] == prefix_len + chunk
                self_prefix_mass[s, self_layers.index(layer_idx)] = attn[:prefix_len].sum()

    mean_cross = cross_attn.mean(axis=(0, 1))

    def group_shares(vec: np.ndarray) -> dict[str, float]:
        return {name: float(vec[a:b].sum()) for name, a, b in groups}

    return {
        "task": task,
        "camera_keys": camera_keys,
        "raw_images": raw_images,
        "wrench_raw": wrench_raw,
        "prefix_len": prefix_len,
        "n_img": n_img,
        "grid": grid,
        "n_lang": n_lang,
        "n_lang_real": n_lang_real,
        "has_wrench": has_wrench,
        "groups": groups,
        "cross_layers": cross_layers,
        "self_layers": self_layers,
        "cross_attn": cross_attn,
        "self_prefix_mass": self_prefix_mass,
        "mean_cross": mean_cross,
        "shares": group_shares(mean_cross),
        "group_shares_fn": group_shares,
        "ablation": ablation,
    }


def slim_record(res: dict) -> dict:
    """Keep only the small per-frame arrays needed for sweep/video aggregation
    (drops raw images and full per-step/per-layer attention to bound memory)."""
    return {
        "shares": res["shares"],
        "mean_cross": res["mean_cross"],
        "self_prefix_mass_mean": float(res["self_prefix_mass"].mean()),
        "wrench_norm": (
            float(torch.linalg.vector_norm(res["wrench_raw"]).item()) if res["wrench_raw"] is not None else None
        ),
        "ablation": res["ablation"],
    }


def save_single_frame_outputs(res: dict, policy, output_dir: Path, header: str):
    groups = res["groups"]
    names = [n for n, _, _ in groups]
    n_cams = len(res["camera_keys"])
    shares = res["shares"]
    mean_cross = res["mean_cross"]
    cross_attn = res["cross_attn"]
    num_steps = cross_attn.shape[0]
    grid = res["grid"]

    # Figure 1: per-camera heatmap overlays
    fig, axes = plt.subplots(2, n_cams, figsize=(4 * n_cams, 8.5))
    if n_cams == 1:
        axes = axes[:, None]
    vmax = max(mean_cross[a:b].max() for _, a, b in groups[:n_cams])
    for ci, key in enumerate(res["camera_keys"]):
        # Same resize+pad as prepare_images so token grid and pixels align (pads left/top)
        img = resize_with_pad(res["raw_images"][key].float(), *policy.config.resize_imgs_with_padding, pad_value=0)
        img_np = img[0].permute(1, 2, 0).clamp(0, 1).numpy()
        name, a, b = groups[ci]
        att_map = torch.from_numpy(mean_cross[a:b].reshape(1, 1, grid, grid))
        att_up = torch.nn.functional.interpolate(att_map, size=img_np.shape[:2], mode="bilinear", align_corners=False)
        att_up = att_up[0, 0].numpy()

        axes[0, ci].imshow(img_np)
        axes[0, ci].set_title(f"{name}\n(share={shares[name]:.1%})")
        axes[0, ci].axis("off")
        axes[1, ci].imshow(img_np)
        im = axes[1, ci].imshow(att_up, cmap="jet", alpha=0.55, vmin=0.0, vmax=vmax)
        axes[1, ci].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label="attention (head/query/layer/step mean)")
    fig.suptitle(f"SmolVLA expert→obs cross-attention | {header} | task: {res['task']}")
    fig.savefig(output_dir / "attention_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: group shares
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [shares[n] for n in names]
    bars = ax.bar(names, vals, color=["tab:blue"] * n_cams + ["tab:orange", "tab:green", "tab:red"][: len(names) - n_cams])
    for bar, v in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("attention share (cross-attn layers)")
    ax.set_title("Where does the action expert attend? (sums to 1 over prefix)")
    fig.savefig(output_dir / "attention_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: dynamics across denoise steps and layers
    shares_per_step = [res["group_shares_fn"](cross_attn[s].mean(axis=0)) for s in range(num_steps)]
    shares_per_layer = [res["group_shares_fn"](cross_attn[:, li].mean(axis=0)) for li in range(len(res["cross_layers"]))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in names:
        ax1.plot(range(num_steps), [s[name] for s in shares_per_step], marker="o", label=name)
        ax2.plot(res["cross_layers"], [s[name] for s in shares_per_layer], marker="s", label=name)
    ax1.set_xlabel("denoise step (t: 1 → 0)")
    ax1.set_ylabel("attention share")
    ax1.set_title("Across flow-matching steps")
    ax2.set_xlabel("expert cross-attn layer index")
    ax2.set_title("Across layers")
    ax1.legend(fontsize=8)
    fig.savefig(output_dir / "attention_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        output_dir / "attention_raw.npz",
        cross_attn=cross_attn,
        self_prefix_mass=res["self_prefix_mass"],
        mean_cross=mean_cross,
        groups=np.array([(n, a, b) for n, a, b in groups], dtype=object),
        cross_layers=np.array(res["cross_layers"]),
        self_layers=np.array(res["self_layers"]),
    )

    lines = [
        header,
        f"task: '{res['task']}'",
        f"prefix_len={res['prefix_len']} | {n_cams} cams x {res['n_img']} img tokens ({grid}x{grid}) "
        f"+ {res['n_lang']} lang ({res['n_lang_real']} real) + state{' + wrench' if res['has_wrench'] else ''}",
        "",
        "Expert -> prefix attention shares (cross-attn layers, mean over heads/queries/layers/steps):",
    ]
    for name in names:
        lines.append(f"  {name:>10s}: {shares[name]:6.2%}")
    img_total = sum(shares[n] for n in names[:n_cams])
    lines += [
        f"  -> all images combined: {img_total:.2%}",
        "",
        f"Self-attn layers: action tokens put {res['self_prefix_mass'].mean():.2%} of their attention on the prefix "
        f"(rest {1 - res['self_prefix_mass'].mean():.2%} on other action tokens).",
    ]
    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)


def save_sweep_outputs(frames, records, layout, args, output_dir: Path, header: str):
    names = [n for n, _, _ in layout["groups"]]
    n_cams = len(layout["camera_keys"])
    shares_arr = np.array([[r["shares"][n] for n in names] for r in records])  # (n_frames, n_groups)
    has_wrench = layout["has_wrench"]
    # Measured force magnitude from the dataset (L2 norm of the full wrench vector)
    wrench_norms = np.array([r["wrench_norm"] for r in records]) if has_wrench else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for gi, name in enumerate(names):
        axes[0].plot(frames, shares_arr[:, gi], label=name, alpha=0.85)
    axes[0].set_ylabel("attention share")
    axes[0].set_title(f"Expert→prefix attention over episode | task: {layout['task']}")
    axes[0].legend(fontsize=8, ncol=4)

    corr = float("nan")
    if has_wrench:
        wi = names.index("wrench")
        axes[1].plot(frames, shares_arr[:, wi], color="tab:red", marker=".", label="wrench attention share")
        axes[1].set_ylabel("wrench attention share", color="tab:red")
        axes[1].tick_params(axis="y", labelcolor="tab:red")
        ax_force = axes[1].twinx()
        ax_force.plot(frames, wrench_norms, color="tab:gray", alpha=0.6, label="measured ||wrench||")
        ax_force.set_ylabel("measured ||wrench||", color="tab:gray")
        if len(frames) > 2:
            corr = float(np.corrcoef(shares_arr[:, wi], wrench_norms)[0, 1])
        axes[1].set_title(f"Force token usage vs measured force (corr={corr:.3f})")
    else:
        axes[1].text(0.5, 0.5, "no wrench input in this model", ha="center", va="center")
    axes[1].set_xlabel("frame index")
    fig.savefig(output_dir / "attention_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        output_dir / "attention_sweep.npz",
        frames=np.array(frames),
        shares=shares_arr,
        group_names=np.array(names),
        mean_cross=np.stack([r["mean_cross"] for r in records]),
        wrench_norms=wrench_norms if wrench_norms is not None else np.array([]),
        self_prefix_mass=np.array([r["self_prefix_mass_mean"] for r in records]),
    )

    lines = [
        header,
        f"task: '{layout['task']}' | {len(frames)} frames (stride={args.stride})",
        "",
        "Mean attention shares over the episode (cross-attn layers):",
    ]
    for gi, name in enumerate(names):
        lines.append(
            f"  {name:>10s}: mean {shares_arr[:, gi].mean():6.2%} "
            f"(min {shares_arr[:, gi].min():.2%}, max {shares_arr[:, gi].max():.2%})"
        )
    lines.append(f"  -> all images combined: {shares_arr[:, :n_cams].sum(axis=1).mean():.2%}")
    if has_wrench:
        wi = names.index("wrench")
        top = np.argsort(shares_arr[:, wi])[::-1][:5]
        lines += [
            "",
            f"Wrench (force token) usage: mean {shares_arr[:, wi].mean():.2%}, "
            f"peak {shares_arr[:, wi].max():.2%} @ frame {frames[int(np.argmax(shares_arr[:, wi]))]}",
            f"Correlation(wrench attention, measured ||wrench||) = {corr:.3f}",
            "Top-5 frames by wrench attention: "
            + ", ".join(f"f{frames[i]} ({shares_arr[i, wi]:.2%})" for i in top),
            "(run single-frame mode with --frame <N> to see the camera heatmaps there)",
        ]
    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)


def save_ablation_outputs(frames, records, layout, output_dir: Path):
    """Plots for --ablate over an episode: action change vs attention vs measured force,
    plus per-joint impact heatmap. Returns summary lines."""
    names = [n for n, _, _ in layout["groups"]]
    target = records[0]["ablation"]["target"]
    diffs = np.array([r["ablation"]["diff_mean"] for r in records])
    joints = np.stack([r["ablation"]["diff_per_joint"] for r in records])  # (n_frames, action_dim)
    att = np.array([r["shares"][target] for r in records]) if target in names else None
    wrench_norms = (
        np.array([r["wrench_norm"] for r in records]) if records[0]["wrench_norm"] is not None else None
    )

    n_rows = 1 + (att is not None) + (wrench_norms is not None)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].plot(frames, diffs, color="tab:purple")
    axes[0].set_ylabel("|Δaction| (mean)")
    axes[0].set_title(f"How much do actions change when '{target}' is removed? (same noise, causal)")
    row = 1
    corr_att = corr_force = float("nan")
    if att is not None:
        axes[row].plot(frames, att, color="tab:red")
        axes[row].set_ylabel(f"{target} attention share")
        if len(frames) > 2:
            corr_att = float(np.corrcoef(diffs, att)[0, 1])
        axes[row].set_title(f"corr(|Δaction|, {target} attention) = {corr_att:.3f}")
        row += 1
    if wrench_norms is not None:
        axes[row].plot(frames, wrench_norms, color="tab:gray")
        axes[row].set_ylabel("measured ||wrench||")
        if len(frames) > 2:
            corr_force = float(np.corrcoef(diffs, wrench_norms)[0, 1])
        axes[row].set_title(f"corr(|Δaction|, measured force) = {corr_force:.3f}")
    axes[-1].set_xlabel("frame index")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Which joints does the ablated input influence?
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), width_ratios=[2.2, 1])
    im = ax1.imshow(joints.T, aspect="auto", cmap="magma", interpolation="nearest",
                    extent=[frames[0], frames[-1], joints.shape[1] - 0.5, -0.5])
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("action dim (joint)")
    ax1.set_title(f"|Δaction| per joint when '{target}' removed")
    fig.colorbar(im, ax=ax1, fraction=0.03)
    ax2.barh(range(joints.shape[1]), joints.mean(axis=0), color="tab:purple")
    ax2.set_xlabel("mean |Δaction|")
    ax2.invert_yaxis()
    ax2.set_title("episode mean per joint")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_joints.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    top = np.argsort(diffs)[::-1][:5]
    lines = [
        "",
        f"Ablation '{target}' (action change when input removed, same flow noise):",
        f"  |Δaction| mean {diffs.mean():.4f}, max {diffs.max():.4f} @ frame {frames[int(np.argmax(diffs))]}",
        f"  corr(|Δaction|, {target} attention) = {corr_att:.3f}",
        f"  corr(|Δaction|, measured ||wrench||) = {corr_force:.3f}",
        f"  most affected joints (mean): {np.argsort(joints.mean(axis=0))[::-1][:4].tolist()}",
        "  top-5 frames by |Δaction|: " + ", ".join(f"f{frames[i]} ({diffs[i]:.4f})" for i in top),
    ]
    return lines, diffs, joints


def render_video(dataset, frames, records, layout, policy, fps: float, output_path: Path):
    """Second pass: compose per-timestep frames (camera overlays + current shares +
    timeline cursor) and encode to mp4 via imageio (bundled ffmpeg)."""
    import imageio.v2 as imageio

    groups = layout["groups"]
    names = [n for n, _, _ in groups]
    camera_keys = layout["camera_keys"]
    n_cams = len(camera_keys)
    n_img, grid = layout["n_img"], layout["grid"]
    has_wrench = layout["has_wrench"]

    shares_arr = np.array([[r["shares"][n] for n in names] for r in records])
    mean_cross_arr = np.stack([r["mean_cross"] for r in records])  # (n_frames, prefix_len)
    # Fixed color scale across the whole video (99.5th percentile of image-token attention)
    img_att_all = mean_cross_arr[:, : n_cams * n_img]
    vmax = float(np.percentile(img_att_all, 99.5))

    img_size = policy.config.resize_imgs_with_padding

    fig = plt.figure(figsize=(4 * n_cams, 9), dpi=100)
    gs = fig.add_gridspec(2, n_cams, height_ratios=[1.8, 1.0], hspace=0.15)
    cam_axes = [fig.add_subplot(gs[0, i]) for i in range(n_cams)]
    bar_ax = fig.add_subplot(gs[1, 0])
    tl_ax = fig.add_subplot(gs[1, 1:]) if n_cams > 1 else fig.add_subplot(gs[1, 0])

    # Camera panels: background image + attention overlay (data updated per frame)
    blank = np.zeros((*img_size, 3), dtype=np.float32)
    im_bg, im_att = [], []
    for ci in range(n_cams):
        im_bg.append(cam_axes[ci].imshow(blank))
        im_att.append(cam_axes[ci].imshow(np.zeros(img_size), cmap="jet", alpha=0.55, vmin=0.0, vmax=vmax))
        cam_axes[ci].set_title(names[ci], fontsize=11)
        cam_axes[ci].axis("off")
    fig.colorbar(im_att[-1], ax=cam_axes, fraction=0.015, pad=0.01, label="attention")

    # Current-frame group shares (bar heights updated per frame)
    bar_colors = ["tab:blue"] * n_cams + ["tab:orange", "tab:green", "tab:red"][: len(names) - n_cams]
    bars = bar_ax.bar(names, shares_arr[0], color=bar_colors)
    bar_ax.set_ylim(0, shares_arr.max() * 1.15)
    bar_ax.set_ylabel("attention share")
    bar_ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    # Timeline with a moving cursor; full series drawn once
    tl_series = {"images (all)": shares_arr[:, :n_cams].sum(axis=1), "language": shares_arr[:, n_cams]}
    tl_series["state"] = shares_arr[:, names.index("state")]
    if has_wrench:
        tl_series["wrench"] = shares_arr[:, names.index("wrench")]
    for label, series in tl_series.items():
        tl_ax.plot(frames, series, label=label, alpha=0.9)
    if has_wrench:
        wrench_norms = np.array([r["wrench_norm"] for r in records])
        tl_force = tl_ax.twinx()
        tl_force.plot(frames, wrench_norms, color="tab:gray", alpha=0.45)
        tl_force.set_ylabel("measured ||wrench||", color="tab:gray", fontsize=9)
    cursor = tl_ax.axvline(frames[0], color="k", linestyle="--", linewidth=1.5)
    tl_ax.set_xlabel("frame index")
    tl_ax.legend(fontsize=8, ncol=4, loc="upper left")
    title = fig.suptitle("", fontsize=12)

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        for k, fidx in enumerate(frames):
            item = dataset[fidx]
            mean_cross = mean_cross_arr[k]
            for ci, key in enumerate(camera_keys):
                img = resize_with_pad(item[key].unsqueeze(0).float(), *img_size, pad_value=0)
                im_bg[ci].set_data(img[0].permute(1, 2, 0).clamp(0, 1).numpy())
                a, b = groups[ci][1], groups[ci][2]
                att_map = torch.from_numpy(mean_cross[a:b].reshape(1, 1, grid, grid))
                att_up = torch.nn.functional.interpolate(att_map, size=img_size, mode="bilinear", align_corners=False)
                im_att[ci].set_data(att_up[0, 0].numpy())
                cam_axes[ci].set_title(f"{names[ci]} ({shares_arr[k, ci]:.1%})", fontsize=11)
            for bar, v in zip(bars, shares_arr[k], strict=True):
                bar.set_height(v)
            cursor.set_xdata([fidx, fidx])
            info = f"frame {fidx} | images {shares_arr[k, :n_cams].sum():.1%} | state {shares_arr[k, names.index('state')]:.1%}"
            if has_wrench:
                info += f" | wrench {shares_arr[k, names.index('wrench')]:.1%}"
            title.set_text(f"SmolVLA expert→obs attention | task: {layout['task']}\n{info}")

            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame_rgba[:, :, :3])
            if (k + 1) % 25 == 0 or k == len(frames) - 1:
                print(f"  rendered {k + 1}/{len(frames)} video frames")
    finally:
        writer.close()
        plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    assert (checkpoint / "config.json").exists(), f"config.json not found in {checkpoint}"

    if args.hdf5:
        repo_id, root = str(args.hdf5), None
        ep_tag = args.hdf5.stem
    else:
        repo_id, root = load_dataset_info_from_train_config(checkpoint)
        repo_id = args.repo_id or repo_id
        root = args.dataset_root or root
        assert repo_id is not None, "Could not infer dataset repo_id; pass --repo-id/--dataset-root"
        ep_tag = f"ep{args.episode}"

    date_suffix = datetime.datetime.now().strftime("%m%d")
    if args.video:
        mode_tag = f"{ep_tag}_video"
    elif args.sweep:
        mode_tag = f"{ep_tag}_sweep"
    else:
        mode_tag = f"{ep_tag}_f{args.frame}"
    if args.ablate:
        mode_tag += f"_ablate-{args.ablate}"
    # Separate dir per checkpoint so comparison runs don't overwrite each other
    output_dir = args.output_dir or Path(f"outputs/attention_viz_{date_suffix}/{checkpoint_tag(checkpoint)}/{mode_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    # Fixed flow-matching start noise so original/ablated inferences differ only by the input
    noise = None
    if args.ablate:
        gen = torch.Generator().manual_seed(args.seed)
        noise = torch.randn(
            1, policy.config.chunk_size, policy.config.max_action_dim, generator=gen
        ).to(args.device)
        print(f"Ablation target: '{args.ablate}' (seed={args.seed})")

    if args.hdf5:
        task = args.task if args.task is not None else "open gripper when human hand applies sufficient force"
        print(f"Loading HDF5 episode {args.hdf5} (camera '{args.hdf5_camera}' -> camera1)")
        dataset = HDF5Episode(args.hdf5, args.hdf5_camera, task=task)
        print(f"  {dataset.num_frames} frames | task: '{task}'")
    else:
        print(f"Loading dataset {repo_id} (root={root}), episode {args.episode}")
        dataset = LeRobotDataset(repo_id, root=root, episodes=[args.episode])

    if args.sweep or args.video:
        frames = list(range(0, dataset.num_frames, args.stride))
        header = f"checkpoint: {checkpoint}\ndataset: {repo_id} | {ep_tag} ({dataset.num_frames} frames)"
        records, layout = [], None
        for i, f in enumerate(frames):
            res = analyze_frame(
                policy, preprocessor, dataset, f, args.task,
                ablate=args.ablate, noise=noise, postprocessor=postprocessor,
            )
            if layout is None:  # token layout is identical for every frame
                layout = {k: res[k] for k in ("groups", "camera_keys", "n_img", "grid", "has_wrench", "task")}
            records.append(slim_record(res))
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                print(f"  analyzed {i + 1}/{len(frames)} frames")
        save_sweep_outputs(frames, records, layout, args, output_dir, header)
        if args.ablate:
            lines, diffs, joints = save_ablation_outputs(frames, records, layout, output_dir)
            np.savez(output_dir / "ablation.npz", frames=np.array(frames), diff_mean=diffs, diff_per_joint=joints)
            summary_path = output_dir / "summary.txt"
            summary_path.write_text(summary_path.read_text() + "\n".join(lines) + "\n")
            print("\n".join(lines))
        if args.video:
            dataset_fps = getattr(dataset, "fps", 30)
            fps = args.fps or max(1.0, dataset_fps / args.stride)  # real-time playback by default
            print(f"Rendering video at {fps:.1f} fps ...")
            render_video(dataset, frames, records, layout, policy, fps, output_dir / "attention_video.mp4")
    else:
        assert args.frame < dataset.num_frames, f"frame {args.frame} out of range (episode has {dataset.num_frames})"
        header = f"checkpoint: {checkpoint}\ndataset: {repo_id} | {ep_tag}, frame {args.frame}"
        res = analyze_frame(
            policy, preprocessor, dataset, args.frame, args.task,
            ablate=args.ablate, noise=noise, postprocessor=postprocessor,
        )
        save_single_frame_outputs(res, policy, output_dir, f"{ep_tag} f{args.frame}")
        if res["ablation"] is not None:
            ab = res["ablation"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(ab["diff_per_joint"])), ab["diff_per_joint"], color="tab:purple")
            ax.set_xlabel("action dim (joint)")
            ax.set_ylabel("|Δaction| (mean over chunk)")
            ax.set_title(f"Action change when '{ab['target']}' removed | mean {ab['diff_mean']:.4f}, max {ab['diff_max']:.4f}")
            fig.savefig(output_dir / "ablation_joints.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            ab_lines = (
                f"\nAblation '{ab['target']}': |Δaction| mean {ab['diff_mean']:.4f}, max {ab['diff_max']:.4f}\n"
                f"per-joint: {np.array2string(ab['diff_per_joint'], precision=4)}\n"
            )
            with open(output_dir / "summary.txt", "a") as f:
                f.write(ab_lines)
            print(ab_lines)
        (output_dir / "summary.txt").write_text(header + "\n" + (output_dir / "summary.txt").read_text())

    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
