# env_smolVLA 환경 설치 가이드

OpenPath Robotics 내부용 lerobot 환경 설치 절차 및 버전 정보.

---

## 시스템 환경

| 항목 | 버전 |
|------|------|
| GPU | NVIDIA GeForce RTX 5080 (16GB) |
| Driver | 595.58.03 |
| CUDA | 13.0 |
| OS | Ubuntu (x86_64) |

---

## 설치 순서

### 1. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

버전 확인:
```bash
uv --version
# uv 0.11.13 (x86_64-unknown-linux-gnu)
```

### 2. lerobot 의존성 설치

conda 환경 불필요. Python 3.12를 uv가 직접 관리:

```bash
cd ~/lerobot
uv venv --prompt smolVLA --python 3.12
uv sync --locked --extra all
```

> uv는 프로젝트 폴더 안 `.venv`에 설치됨 (`~/lerobot/.venv`).  
> `--prompt` 생략 시 프로젝트명(lerobot)으로 표시됨.

### 4. opencv GUI 버전으로 교체

lerobot 기본 설치 시 `opencv-python-headless`가 설치되어 `cv2.imshow` 사용 불가.  
viewer 스크립트 사용을 위해 GUI 버전으로 교체:

```bash
uv pip uninstall opencv-python-headless
uv pip install opencv-python
```

### 5. 추가 패키지 설치

```bash
# OpenARM CAN 통신 라이브러리
uv pip install openarm_can

# OSMC 모터 드라이버 (로컬 wheel)
uv pip install osmc-1.1.0-cp312-cp312-linux_x86_64.whl
```

---

## 설치된 주요 패키지 버전

| 패키지 | 버전 |
|--------|------|
| Python | 3.12.13 |
| torch | 2.11.0+cu130 |
| CUDA (torch) | 13.0 |
| opencv-python | 4.13.0 |
| openarm_can | - |
| osmc | 1.1.0 |

---

## 실행 방법

### uv run 사용 (권장)

```bash
cd ~/lerobot
uv run python script.py
uv run lerobot-train ...
```

### VSCode에서 실행

인터프리터 경로를 `.venv`로 지정:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. `Enter interpreter path` 클릭
3. `/home/youngwoo/lerobot/.venv/bin/python` 입력

### 추가 패키지 설치 시

```bash
uv pip install 패키지명          # PyPI 패키지
uv pip install /path/to/pkg.whl  # 로컬 wheel
```

> `pip install` 사용 시 conda base에 설치될 수 있으니 반드시 `uv pip` 사용.

---

## Remote 구조 (Git)

```
origin    → https://github.com/openpath-robotics/lerobot.git  (회사 repo)
upstream  → https://github.com/huggingface/lerobot.git        (원본 lerobot)
```

upstream 업데이트 가져오기:
```bash
git fetch upstream
git merge upstream/main
git push origin main
```
