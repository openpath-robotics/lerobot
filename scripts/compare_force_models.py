#!/usr/bin/env python
"""Compare a force-trained vs a no-force SmolVLA checkpoint on the same episode.

Runs both policies frame-by-frame on the SAME observations (with the SAME fixed
flow-matching noise) and compares their immediate predicted action — especially the
gripper — against the ground-truth action and the measured force. This answers, for
THIS episode (offline / open-loop), whether the force input actually helps:

  - If the no-force model tracks the ground-truth gripper as well as the force model,
    vision+state already carry the cue -> force is a SIDEGRADE here.
  - If only the force model opens the gripper at the right moment, force does real
    work -> force is an UPGRADE here.

Caveat: this is open-loop on a logged episode. It cannot capture closed-loop recovery
or compounding error; the definitive test is from-scratch training + closed-loop rollout
across seeds (see project notes). This is the cheap, strong first screen.

Reuses HDF5Episode / build_batch from visualize_smolvla_attention.py.

Example:
  .venv/bin/python scripts/compare_force_models.py \
      --force-checkpoint   .../smolvla_force_test_0615_all/checkpoints/032000/pretrained_model \
      --noforce-checkpoint .../smolvla_force_test_0615_all_noforce/checkpoints/032000/pretrained_model \
      --hdf5 .../episode_67.hdf5 --stride 1
"""

import argparse
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Reuse the HDF5 reader and batch builder from the visualization script
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "viz_attn", str(Path(__file__).resolve().parent / "visualize_smolvla_attention.py")
)
_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viz)
HDF5Episode = _viz.HDF5Episode
build_batch = _viz.build_batch

GRIPPER_DIM = 7  # last action dim = gripper for this 8-DoF single-arm setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force-checkpoint", type=Path, required=True)
    p.add_argument("--noforce-checkpoint", type=Path, required=True)
    p.add_argument("--hdf5", type=Path, required=True, help="Raw HDF5 episode to analyze")
    p.add_argument("--hdf5-camera", type=str, default="cam_wrist_left")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Shared flow-matching noise seed for both models")
    p.add_argument("--task", type=str, default="open gripper when human hand applies sufficient force")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_model(checkpoint: Path, device: str):
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def predict_immediate(policy, pre, post, dataset, frame_idx, task, noise, device):
    """Return the model's immediate predicted action (first action of the chunk), unnormalized."""
    batch, _ = build_batch(dataset, frame_idx, task, input_keys=set(policy.config.input_features))
    batch = pre(batch)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch, noise=noise)  # (1, chunk, action_dim)
    action = post(chunk[:, 0])  # immediate action, unnormalized -> (1, action_dim)
    return action[0].cpu().numpy()


def main():
    args = parse_args()
    date_suffix = datetime.datetime.now().strftime("%m%d")
    out = args.output_dir or Path(f"outputs/force_compare_{date_suffix}/{args.hdf5.stem}")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading force model ...")
    f_policy, f_pre, f_post = load_model(args.force_checkpoint.resolve(), args.device)
    print("Loading no-force model ...")
    n_policy, n_pre, n_post = load_model(args.noforce_checkpoint.resolve(), args.device)

    dataset = HDF5Episode(args.hdf5, args.hdf5_camera, task=args.task)
    frames = list(range(0, dataset.num_frames, args.stride))
    action_dim = dataset[0]["action"].shape[0]
    print(f"{dataset.num_frames} frames, analyzing {len(frames)} (stride {args.stride})")

    # Shared fixed noise so the two models differ only by weights+inputs, not the random start
    gen = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(
        1, f_policy.config.chunk_size, f_policy.config.max_action_dim, generator=gen
    ).to(args.device)

    f_pred = np.zeros((len(frames), action_dim))
    n_pred = np.zeros((len(frames), action_dim))
    gt = np.zeros((len(frames), action_dim))
    force_mag = np.zeros(len(frames))

    for i, fr in enumerate(frames):
        item = dataset[fr]
        gt[i] = item["action"].numpy()
        force_mag[i] = float(torch.linalg.vector_norm(item["observation.wrench"]).item())
        f_pred[i] = predict_immediate(f_policy, f_pre, f_post, dataset, fr, args.task, noise, args.device)
        n_pred[i] = predict_immediate(n_policy, n_pre, n_post, dataset, fr, args.task, noise, args.device)
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            print(f"  {i + 1}/{len(frames)}")

    g_gt, g_f, g_n = gt[:, GRIPPER_DIM], f_pred[:, GRIPPER_DIM], n_pred[:, GRIPPER_DIM]

    # ---- Plot 1: gripper trajectories vs force ----
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frames, g_gt, "k-", lw=2.5, label="ground truth gripper", alpha=0.7)
    ax.plot(frames, g_f, "tab:green", lw=2, label="FORCE model")
    ax.plot(frames, g_n, "tab:red", lw=2, label="NO-FORCE model")
    ax.set_xlabel("frame")
    ax.set_ylabel("gripper action (dim 7)")
    ax.legend(loc="upper left")
    axf = ax.twinx()
    axf.plot(frames, force_mag, color="tab:gray", alpha=0.4, label="measured ||force||")
    axf.set_ylabel("measured ||force||", color="tab:gray")
    ax.set_title(f"Gripper prediction: force vs no-force | {args.hdf5.stem}")
    fig.tight_layout()
    fig.savefig(out / "gripper_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: per-model tracking error + model divergence ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    a1.plot(frames, np.abs(g_f - g_gt), "tab:green", label="FORCE |pred-GT|")
    a1.plot(frames, np.abs(g_n - g_gt), "tab:red", label="NO-FORCE |pred-GT|")
    a1.set_ylabel("gripper abs error")
    a1.legend()
    a1.set_title("Gripper tracking error vs ground truth")
    a2.plot(frames, np.abs(g_f - g_n), "tab:purple", label="|force - noforce| gripper")
    a2.plot(frames, force_mag / max(force_mag.max(), 1e-9) * np.abs(g_f - g_n).max(), color="tab:gray", alpha=0.4, label="force (scaled)")
    a2.set_ylabel("model divergence")
    a2.set_xlabel("frame")
    a2.legend()
    a2.set_title("How differently do the two models command the gripper?")
    fig.tight_layout()
    fig.savefig(out / "gripper_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(out / "compare.npz", frames=np.array(frames), gt=gt, f_pred=f_pred, n_pred=n_pred, force_mag=force_mag)

    # ---- Summary ----
    def mae(a, b):
        return float(np.abs(a - b).mean())

    # Focus on the "contact" window where force is applied (most informative)
    contact = force_mag > (force_mag.min() + 0.3 * (force_mag.max() - force_mag.min()))
    lines = [
        f"episode: {args.hdf5}",
        f"force checkpoint:    {args.force_checkpoint}",
        f"no-force checkpoint: {args.noforce_checkpoint}",
        f"frames analyzed: {len(frames)} (stride {args.stride}) | gripper dim = {GRIPPER_DIM}",
        f"measured force range: [{force_mag.min():.1f}, {force_mag.max():.1f}]",
        "",
        "GRIPPER tracking vs ground truth (lower = better):",
        f"  FORCE    model gripper MAE = {mae(g_f, g_gt):.4f}   (contact window {mae(g_f[contact], g_gt[contact]):.4f})",
        f"  NO-FORCE model gripper MAE = {mae(g_n, g_gt):.4f}   (contact window {mae(g_n[contact], g_gt[contact]):.4f})",
        "",
        f"Full-action MAE vs GT:  FORCE {mae(f_pred, gt):.4f} | NO-FORCE {mae(n_pred, gt):.4f}",
        f"Model divergence on gripper |force-noforce|: mean {mae(g_f, g_n):.4f}, max {np.abs(g_f - g_n).max():.4f}",
        f"corr(FORCE gripper, measured force)    = {np.corrcoef(g_f, force_mag)[0, 1]:.3f}",
        f"corr(NO-FORCE gripper, measured force) = {np.corrcoef(g_n, force_mag)[0, 1]:.3f}",
        f"corr(GT gripper, measured force)       = {np.corrcoef(g_gt, force_mag)[0, 1]:.3f}",
    ]
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
