"""LeRobot 데이터셋 뷰어 - parquet 파일 하나만 지정해서 보기
데이터 셋 맞게 가져왔는지 확인하는 용도 """
import argparse
import pandas as pd
from PIL import Image
import io
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 뷰어")
    parser.add_argument("file", type=str, help="parquet 파일 경로")
    parser.add_argument("--tasks", type=str, default=None, help="tasks.parquet 경로 (없으면 자동탐색)")
    parser.add_argument("--speed", type=int, default=100, help="재생 속도 (ms, 낮을수록 빠름)")
    args = parser.parse_args()

    # parquet 파일 하나만 로드
    print(f"파일 로딩: {args.file}")
    df = pd.read_parquet(args.file)
    print(f"프레임: {len(df)}")

    # tasks.parquet 자동 탐색
    tasks = None
    if args.tasks:
        tasks = pd.read_parquet(args.tasks)
    else:
        # file 경로에서 meta/tasks.parquet 찾기 (data/ 기준 또는 부모 디렉토리 순회)
        found = False
        parts = args.file.split("/data/")
        if len(parts) == 2:
            tasks_path = parts[0] + "/meta/tasks.parquet"
            if os.path.exists(tasks_path):
                tasks = pd.read_parquet(tasks_path)
                found = True
        if not found:
            # 상위 디렉토리를 순회하며 meta/tasks.parquet 탐색
            search_dir = os.path.dirname(os.path.abspath(args.file))
            for _ in range(6):
                tasks_path = os.path.join(search_dir, "meta", "tasks.parquet")
                if os.path.exists(tasks_path):
                    tasks = pd.read_parquet(tasks_path)
                    break
                search_dir = os.path.dirname(search_dir)
    # 기본 정보 출력
    print(f"컬럼: {list(df.columns)}")
    episodes = sorted(df['episode_index'].unique())
    print(f"에피소드: {episodes}")
    # 이미지 컬럼 자동 탐색
    img_cols = [c for c in df.columns if 'image' in c.lower()]
    state_col = 'observation.state' if 'observation.state' in df.columns else None
    action_col = 'action' if 'action' in df.columns else None
    print(f"이미지 컬럼: {img_cols}")
    print()

    if tasks is not None:
        print("=== Task 목록 ===")
        for idx, task_name in enumerate(tasks.index):
            print(f"  [{idx}] {task_name}")
        print()

    ep_idx = 0
    current_ep = episodes[ep_idx]
    while True:
        ep_df = df[df['episode_index'] == current_ep].sort_values('frame_index')
        if len(ep_df) == 0:
            print(f"에피소드 {current_ep} 없음")
            break
        task_idx = int(ep_df['task_index'].iloc[0])
        task_name = tasks.index[task_idx] if tasks is not None else f"task_{task_idx}"
        print(f"=== Episode {current_ep} | Task[{task_idx}]: {task_name} | {len(ep_df)} frames ===")
        print("Space=일시정지  A/D=프레임이동  N/P=에피소드이동  Q=종료")
        paused = False
        frame_idx = 0
        while frame_idx < len(ep_df):
            row = ep_df.iloc[frame_idx]
            # 이미지 디코딩
            imgs = []
            for col in img_cols:
                raw = row[col]
                if isinstance(raw, dict) and 'bytes' in raw:
                    img = Image.open(io.BytesIO(raw['bytes']))
                    img = np.array(img)[:, :, ::-1]
                    h, w = img.shape[:2]
                    new_w = 384
                    new_h = int(h * new_w / w)
                    img = cv2.resize(img, (new_w, new_h))
                    label = col.split('.')[-1]
                    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    imgs.append(img)

            # 텍스트 먼저 구성
            texts = [
                f"Ep:{current_ep} Frame:{int(row['frame_index'])}/{len(ep_df)-1}",
                f"Task[{task_idx}]:",
            ]
            for i in range(0, len(task_name), 42):
                texts.append(f"  {task_name[i:i+42]}")
            texts.append("")
            VALS_PER_ROW = 4
            if state_col and state_col in df.columns:
                state = row[state_col]
                state_dim = len(state)
                texts.append(f"State ({state_dim}D):")
                for i in range(0, state_dim, VALS_PER_ROW):
                    chunk = [round(float(state[j]), 3) for j in range(i, min(i + VALS_PER_ROW, state_dim))]
                    texts.append(f" [{i}~{min(i+VALS_PER_ROW-1,state_dim-1)}]: {chunk}")
                texts.append("")
            if action_col and action_col in df.columns:
                action = row[action_col]
                action_dim = len(action)
                texts.append(f"Action ({action_dim}D):")
                for i in range(0, action_dim, VALS_PER_ROW):
                    chunk = [round(float(action[j]), 3) for j in range(i, min(i + VALS_PER_ROW, action_dim))]
                    texts.append(f" [{i}~{min(i+VALS_PER_ROW-1,action_dim-1)}]: {chunk}")

            # 패널 높이: 텍스트 줄 수와 이미지 높이 중 큰 값
            LINE_H = 20
            PANEL_W = 480
            img_h = imgs[0].shape[0] if imgs else 384
            panel_h = max(img_h, len(texts) * LINE_H + 30)
            info_panel = np.zeros((panel_h, PANEL_W, 3), dtype=np.uint8)

            y = 25
            for t in texts:
                if t:
                    cv2.putText(info_panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
                y += LINE_H

            # 이미지 높이가 패널보다 작으면 아래쪽 패딩
            if imgs and imgs[0].shape[0] < panel_h:
                imgs = [np.pad(img, ((0, panel_h - img.shape[0]), (0, 0), (0, 0))) for img in imgs]

            # 합치기
            if imgs:
                display = np.hstack(imgs + [info_panel])
            else:
                display = info_panel
            cv2.imshow("Dataset Viewer", display)

            key = cv2.waitKey(0 if paused else args.speed) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                ep_idx = min(ep_idx + 1, len(episodes) - 1)
                current_ep = episodes[ep_idx]
                break
            elif key == ord('p'):
                ep_idx = max(ep_idx - 1, 0)
                current_ep = episodes[ep_idx]
                break
            elif key == 81 or key == ord('a'):
                frame_idx = max(0, frame_idx - 1)
            elif key == 83 or key == ord('d'):
                frame_idx = min(len(ep_df) - 1, frame_idx + 1)
            else:
                if not paused:
                    frame_idx += 1
        else:
            print(f"  에피소드 {current_ep} 재생 완료")
            ep_idx = min(ep_idx + 1, len(episodes) - 1)
            current_ep = episodes[ep_idx]

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
