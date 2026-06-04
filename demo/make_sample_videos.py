"""
make_sample_videos.py — 샘플 이미지로 테스트 영상 3종 생성.
data/ 디렉토리 없이도 동작. Optical Flow 테스트를 위해 zoom+pan 효과 적용.

생성 파일:
  demo/test_assets/test_fire_smoke.mp4   — 검은연기·흰연기 순환 (화재 시나리오)
  demo/test_assets/test_fire_flame.mp4   — 화염 이미지 연속 (화재 시나리오)
  demo/test_assets/test_normal.mp4       — 구름·안개 순환 (정상 시나리오)

실행: python demo/make_sample_videos.py
"""
import os
import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "demo", "test_assets", "samples")
OUT_DIR = os.path.join(ROOT, "demo", "test_assets")

FPS = 10
N_FRAMES = 60      # 6초 영상
OUT_SIZE = (640, 480)


def _load(name):
    img = Image.open(os.path.join(SAMPLES, name)).convert("RGB")
    img = img.resize(OUT_SIZE, Image.LANCZOS)
    return np.array(img)[:, :, ::-1].copy()   # RGB→BGR


def _zoom_pan_frames(img, n, zoom_start=1.0, zoom_end=1.08, dx=6, dy=3):
    """n 프레임에 걸쳐 천천히 zoom+pan → Optical Flow 움직임 확보."""
    h, w = img.shape[:2]
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        scale = zoom_start + (zoom_end - zoom_start) * t
        cx = w / 2 + dx * t
        cy = h / 2 + dy * t
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        frames.append(cv2.warpAffine(img, M, (w, h)))
    return frames


def _make_video(path, frame_lists):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, OUT_SIZE)
    for fl in frame_lists:
        for f in fl:
            vw.write(f)
    vw.release()
    rel = os.path.relpath(path, ROOT)
    total = sum(len(fl) for fl in frame_lists)
    print(f"  생성: {rel}  ({total}프레임, {total/FPS:.1f}초)")


def _make_normal_frames(n):
    """파란 하늘 + 녹색 들판으로 연기 없는 명확한 정상 장면 합성.
    흰 구름 없음 — 흰연기와의 혼동 방지."""
    w, h = OUT_SIZE
    horizon = int(h * 0.40)   # 하늘 60% / 들판 40%
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 하늘: 파란 gradient (위쪽 짙고 아래쪽 밝아짐) + 미세 노이즈
        for y in range(horizon):
            sky_t = y / horizon
            # 위→아래: (30,80,160) → (100,170,230)
            b = int(160 + 70 * sky_t)
            g = int(80  + 90 * sky_t)
            r = int(30  + 70 * sky_t)
            # 시간에 따라 하늘 밝기 0.95~1.05 변화 (바람·햇빛 느낌)
            factor = 0.95 + 0.10 * np.sin(2 * np.pi * t + y * 0.05)
            frame[y, :] = np.clip([b * factor, g * factor, r * factor], 0, 255)
        noise_sky = rng.integers(-6, 7, (horizon, w, 3))
        frame[:horizon] = np.clip(frame[:horizon].astype(int) + noise_sky, 0, 255)

        # 들판: 녹색 gradient (지평선 밝고 아래쪽 짙어짐) + 노이즈
        for y in range(horizon, h):
            field_t = (y - horizon) / (h - horizon)
            b = int(30  - 10 * field_t)
            g = int(140 - 40 * field_t)
            r = int(20  -  5 * field_t)
            frame[y, :] = np.clip([b, g, r], 0, 255)
        noise_field = rng.integers(-10, 11, (h - horizon, w, 3))
        frame[horizon:] = np.clip(frame[horizon:].astype(int) + noise_field, 0, 255)

        frames.append(frame)
    return frames


print("테스트 영상 생성 중...\n")

# 1. 연기 영상 (검은연기 30f + 흰연기 30f)
black = _load("fire_black_smoke.jpg")
white = _load("fire_white_smoke.jpg")
_make_video(
    os.path.join(OUT_DIR, "test_fire_smoke.mp4"),
    [_zoom_pan_frames(black, N_FRAMES // 2, dx=8, dy=2),
     _zoom_pan_frames(white, N_FRAMES // 2, dx=-5, dy=4)],
)

# 2. 화염 영상 (화염 60f)
flame = _load("fire_flame.jpg")
_make_video(
    os.path.join(OUT_DIR, "test_fire_flame.mp4"),
    [_zoom_pan_frames(flame, N_FRAMES, zoom_end=1.12, dx=10, dy=5)],
)

# 3. 정상 영상 — 파란 하늘+녹색 들판 합성 (흰연기 혼동 없음)
_make_video(
    os.path.join(OUT_DIR, "test_normal.mp4"),
    [_make_normal_frames(N_FRAMES)],
)

print("\nGradio 영상 입력 탭에서 아래 파일로 테스트하세요:")
for name in ("test_fire_smoke.mp4", "test_fire_flame.mp4", "test_normal.mp4"):
    print(f"  demo/test_assets/{name}")
print("\n주의: 가중치 없으면 모델이 무작위 예측 → 화재/정상 결과가 임의로 나옵니다.")
print("      파이프라인 흐름(GradCAM, 교차검증, VLM 보고) 자체는 정상 동작합니다.")
