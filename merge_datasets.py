#!/usr/bin/env python3
"""
기존 gzip 데이터(320x180) + 새 데이터(640x360) → 440 에피소드 병합
새 데이터는 640x360 → 320x180 리사이즈 후 gzip 저장
"""
import os
import shutil
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

OLD_DIR = os.path.expanduser("~/Merged_VLA_gzip_440")
NEW_DIR = os.path.expanduser("~/Merged_VLA_new")
OUT_DIR = os.path.expanduser("~/Merged_VLA_gzip_440_merged")
TARGET_H, TARGET_W = 180, 320
NUM_WORKERS = 8

# (src_dir, src_ep_start, src_ep_end, dest_ep_start, needs_resize)
# 새 데이터 실제 순서: 120~169=task0, 170~219=task1, 0~29=task2, 30~59=task3, 60~89=task4, 90~119=task5
MERGE_PLAN = [
    (OLD_DIR,   0,  49,   0, False),  # task 0: 기존 0~49    → dest 0~49
    (NEW_DIR, 120, 169,  50, True),   # task 0: 새   120~169 → dest 50~99
    (OLD_DIR,  50,  99, 100, False),  # task 1: 기존 50~99   → dest 100~149
    (NEW_DIR, 170, 219, 150, True),   # task 1: 새   170~219 → dest 150~199
    (OLD_DIR, 100, 129, 200, False),  # task 2: 기존 100~129 → dest 200~229
    (NEW_DIR,   0,  29, 230, True),   # task 2: 새   0~29    → dest 230~259
    (OLD_DIR, 130, 159, 260, False),  # task 3: 기존 130~159 → dest 260~289
    (NEW_DIR,  30,  59, 290, True),   # task 3: 새   30~59   → dest 290~319
    (OLD_DIR, 160, 189, 320, False),  # task 4: 기존 160~189 → dest 320~349
    (NEW_DIR,  60,  89, 350, True),   # task 4: 새   60~89   → dest 350~379
    (OLD_DIR, 190, 219, 380, False),  # task 5: 기존 190~219 → dest 380~409
    (NEW_DIR,  90, 119, 410, True),   # task 5: 새   90~119  → dest 410~439
]


def process_episode(args):
    src_dir, src_ep, dest_ep, needs_resize = args
    src_path = os.path.join(src_dir, f"episode_{src_ep}.hdf5")
    dest_path = os.path.join(OUT_DIR, f"episode_{dest_ep}.hdf5")

    if not needs_resize:
        shutil.copy2(src_path, dest_path)
        return dest_ep

    # 640x360 → 320x180 리사이즈 후 gzip 저장
    with h5py.File(src_path, 'r') as f_in, h5py.File(dest_path, 'w') as f_out:
        # 이미지 외 데이터는 그대로 복사
        for key in f_in.keys():
            if key != 'images':
                f_in.copy(key, f_out)

        # 이미지 리사이즈
        f_out.create_group('images')
        for cam in f_in['images'].keys():
            imgs = f_in['images'][cam][:]  # (T, H, W, 3)
            resized = np.stack([
                cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                for img in imgs
            ])
            f_out['images'].create_dataset(
                cam, data=resized, compression='gzip', compression_opts=4
            )

    return dest_ep


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = []
    for src_dir, src_start, src_end, dest_start, needs_resize in MERGE_PLAN:
        for i, src_ep in enumerate(range(src_start, src_end + 1)):
            tasks.append((src_dir, src_ep, dest_start + i, needs_resize))

    print(f"총 {len(tasks)}개 에피소드 처리 시작 (OUT: {OUT_DIR})")

    with Pool(NUM_WORKERS) as p:
        for _ in tqdm(p.imap_unordered(process_episode, tasks), total=len(tasks)):
            pass

    print(f"\n완료! 결과: {OUT_DIR}")
    print(f"파일 수: {len(os.listdir(OUT_DIR))}")


if __name__ == '__main__':
    main()
