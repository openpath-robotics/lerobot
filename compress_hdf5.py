"""
HDF5 이미지 gzip 압축 스크립트
입력: /home/youngwoo/dataset_0324  (raw 이미지)
출력: /home/youngwoo/dataset_0324_gzip (320x180, gzip 압축)
"""
import os
import h5py
import cv2
import numpy as np
import glob
import re
from tqdm import tqdm
from multiprocessing import Pool

# ============================================================
# 설정
# ============================================================
INPUT_DIR  = "/media/youngwoo/OPR_LINUX/0324_rand_redblock"
OUTPUT_DIR = os.path.expanduser("~/openarm_0324_gzip")
TARGET_W, TARGET_H = 320, 180
GZIP_OPTS  = 4
NUM_WORKERS = 8
# ============================================================


def compress_episode(file_path):
    file_name = os.path.basename(file_path)
    out_path  = os.path.join(OUTPUT_DIR, file_name)

    with h5py.File(file_path, 'r') as f_in, h5py.File(out_path, 'w') as f_out:
        # 이미지 외 데이터 그대로 복사
        for key in f_in.keys():
            if key != 'images':
                f_out.create_dataset(key, data=f_in[key][:])

        # 이미지 리사이즈 + gzip 압축
        if 'images' in f_in:
            grp_out = f_out.create_group('images')
            for cam_name in f_in['images'].keys():
                cam_data   = f_in['images'][cam_name][:]
                n_frames   = cam_data.shape[0]
                resized    = np.zeros((n_frames, TARGET_H, TARGET_W, 3), dtype=np.uint8)
                for i in range(n_frames):
                    resized[i] = cv2.resize(cam_data[i], (TARGET_W, TARGET_H),
                                            interpolation=cv2.INTER_AREA)
                grp_out.create_dataset(cam_name, data=resized,
                                       compression='gzip', compression_opts=GZIP_OPTS)
    return file_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    h5_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, '*.hdf5')),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )
    print(f"총 {len(h5_files)}개 파일 압축 시작")
    print(f"  입력: {INPUT_DIR}")
    print(f"  출력: {OUTPUT_DIR}  ({TARGET_W}x{TARGET_H}, gzip={GZIP_OPTS})")

    with Pool(NUM_WORKERS) as p:
        for _ in tqdm(p.imap_unordered(compress_episode, h5_files), total=len(h5_files)):
            pass

    print(f"\n완료! 파일 수: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
