"""HDF5 데이터셋 뷰어
이미지(raw uint8 또는 JPEG bytes)와 state/action 값을 실시간으로 확인

사용법:
  python view_hdf5.py episode_0.hdf5
  python view_hdf5.py episode_0.hdf5 --speed 50
"""

import argparse
import io
import sys

import cv2
import h5py
import numpy as np
from PIL import Image

def decode_image(raw) -> np.ndarray:
    """raw uint8 배열 또는 JPEG bytes → BGR numpy"""
    if raw.dtype == np.uint8 and raw.ndim == 3:
        # (H, W, C) raw uint8
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    else:
        # 1D uint8 = JPEG bytes
        img = Image.open(io.BytesIO(bytes(raw)))
        return np.array(img)[:, :, ::-1]


def make_info_panel(h: int, ep_idx: int, frame_idx: int, n_frames: int,
                    q_pos, action) -> np.ndarray:
    panel = np.zeros((h, 440, 3), dtype=np.uint8)
    texts = [
        f"Episode: {ep_idx}",
        f"Frame: {frame_idx}/{n_frames - 1}",
        "",
    ]

    if q_pos is not None:
        dim = len(q_pos)
        half = dim // 2
        texts.append(f"State ({dim}D):")
        texts.append(f" R: {[round(float(v), 3) for v in q_pos[:half]]}")
        texts.append(f" L: {[round(float(v), 3) for v in q_pos[half:]]}")
        texts.append("")

    if action is not None:
        dim = len(action)
        half = dim // 2
        texts.append(f"Action ({dim}D):")
        texts.append(f" R: {[round(float(v), 3) for v in action[:half]]}")
        texts.append(f" L: {[round(float(v), 3) for v in action[half:]]}")

    y = 25
    for t in texts:
        if t:
            cv2.putText(panel, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)
        y += 20
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="HDF5 파일 경로")
    parser.add_argument("--speed", type=int, default=66, help="재생 ms (낮을수록 빠름)")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        # 구조 출력
        print("=== HDF5 구조 ===")
        def show(name, obj):
            if hasattr(obj, "shape"):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {name}/")
        f.visititems(show)
        print()

        n_frames = f["action"].shape[0]
        q_pos_all = f["q_pos"][:] if "q_pos" in f else None
        action_all = f["action"][:] if "action" in f else None

        # 카메라 키 탐색
        cam_keys = []
        if "images" in f:
            cam_keys = list(f["images"].keys())
        print(f"카메라: {cam_keys}")
        print(f"프레임 수: {n_frames}")
        print()
        print("Space=일시정지  A/D=프레임이동  Q=종료")

        # 이미지 전체 로드 (메모리에)
        cam_data = {}
        for k in cam_keys:
            cam_data[k] = f[f"images/{k}"][:]

    ep_num = int("".join(filter(str.isdigit, args.file.split("/")[-1])) or "0")

    paused = False
    frame_idx = 0

    while True:
        imgs = []
        for k in cam_keys:
            raw = cam_data[k][frame_idx]
            img = decode_image(raw)
            # 표시 크기 통일 (가로 320)
            h, w = img.shape[:2]
            dw, dh = 320, int(h * 320 / w)
            img = cv2.resize(img, (dw, dh))
            cv2.putText(img, k, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1)
            imgs.append(img)

        panel_h = imgs[0].shape[0] if imgs else 300
        q = q_pos_all[frame_idx] if q_pos_all is not None else None
        a = action_all[frame_idx] if action_all is not None else None
        panel = make_info_panel(panel_h, ep_num, frame_idx, n_frames, q, a)

        # 카메라가 여럿이면 2열 그리드
        if len(imgs) > 2:
            row1 = np.hstack(imgs[:2])
            pad = np.zeros_like(imgs[0])
            row2_list = imgs[2:] + [pad] * (2 - len(imgs[2:]))
            row2 = np.hstack(row2_list[:2])
            grid = np.vstack([row1, row2])
            # panel 높이 맞춤
            if panel.shape[0] < grid.shape[0]:
                panel = np.vstack([panel,
                    np.zeros((grid.shape[0] - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)])
            display = np.hstack([grid, panel])
        else:
            display = np.hstack(imgs + [panel])

        cv2.imshow("HDF5 Viewer", display)
        key = cv2.waitKey(0 if paused else args.speed) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (81, ord("a")):
            frame_idx = max(0, frame_idx - 1)
        elif key in (83, ord("d")):
            frame_idx = min(n_frames - 1, frame_idx + 1)
        else:
            if not paused:
                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
