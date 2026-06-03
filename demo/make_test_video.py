"""
make_test_video.py — YouTube 연속 프레임으로 테스트 영상(.mp4) 생성 +
Optical Flow 연기방향/풍향 검증.

- demo/test_assets/smoke_test.mp4 생성 → Gradio 영상 입력 테스트용
- 두 연속 프레임으로 pipeline 실행 → 실제 연기방향 추정 확인
실행: python demo/make_test_video.py
"""
import os, sys, glob, re
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.fire_pipeline import FireDetectionPipeline, PipelineConfig
from demo.model_loader import load_model, rank_top_models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YT = os.path.join(ROOT, "data", "fireimage", "abnormal", "youtube")
OUT_DIR = os.path.join(ROOT, "demo", "test_assets")
os.makedirs(OUT_DIR, exist_ok=True)


def _frame_num(p):
    m = re.search(r"_f(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


# 1. 연속 프레임 정렬 — 화재 전개 후(연기 가시) 구간을 위해 후반부 사용
all_frames = sorted(glob.glob(os.path.join(YT, "*.jpg")), key=_frame_num)
if len(all_frames) < 2:
    print("연속 프레임 부족. 종료.")
    sys.exit(0)
# 영상 후반 60~85% 구간에서 24장 (연기가 또렷한 시점)
s = int(len(all_frames) * 0.60)
frames = all_frames[s:s + 24]
print(f"연속 프레임 {len(frames)}장 사용(후반부): "
      f"{os.path.basename(frames[0])} ~ {os.path.basename(frames[-1])}")

# 2. 테스트 영상 생성 (5fps)
sample = cv2.imread(frames[0])
h, w = sample.shape[:2]
out_path = os.path.join(OUT_DIR, "smoke_test.mp4")
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
for fp in frames:
    vw.write(cv2.imread(fp))
vw.release()
print(f"테스트 영상 생성: {os.path.relpath(out_path, ROOT)}  ({w}x{h}, {len(frames)}프레임)")

# 3. 두 연속 프레임으로 Optical Flow 방향 검증
#    같은 burst 내에서 충분히 떨어진 두 프레임(움직임 확보)
prev = np.array(Image.open(frames[0]).convert("RGB"))
curr = np.array(Image.open(frames[4]).convert("RGB"))

# 검증된 가중치 모델 사용 (DenseNet121)
name = "DenseNet121"
model, loaded = load_model(name, fold=0)
# 160px 축소 시 이동량이 작아지므로 최소 이동량 임계값 완화
pipe = FireDetectionPipeline(model=model, model_name=name,
                             config=PipelineConfig(image_size=160,
                                                   flow_min_magnitude=0.2))

print(f"\n[{name}] 영상(연속프레임) 추론  가중치={loaded}")
r = pipe.run(curr, wind_direction=270.0, prev_frame=prev)
print(f"  화재감지={r.fire_detected}  신뢰도={r.fire_confidence:.3f}  경로={r.path_type}")
print(f"  연기방향={r.smoke_direction_deg}  (추정원={r.direction_source})")
print(f"  이동량={r.flow_magnitude:.2f}px/f")
print(f"  풍향일치={r.wind_aligned}  (cosine={r.wind_align_score:.2f})")

if r.direction_source == "optical_flow":
    print("\n  ✓ Optical Flow로 실제 연기방향 추정 성공 (영상 입력의 효과)")
else:
    print(f"\n  △ 방향추정원={r.direction_source} (이동량 부족 시 PCA 폴백)")

print(f"\n→ Gradio에서 이 영상으로 테스트: {os.path.relpath(out_path, ROOT)}")
