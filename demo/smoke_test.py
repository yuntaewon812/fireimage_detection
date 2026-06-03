"""
smoke_test.py — Gradio 없이 데모 배관 검증.
백업된 유효 모델(DenseNet121, Resnet50 fold0) + 실제 화재 이미지 1장으로
파이프라인 + VLM 응답까지 한 번에 돌려본다.

실행: python demo/smoke_test.py
"""
import os, sys, glob
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.fire_pipeline import FireDetectionPipeline, PipelineConfig
from demo.model_loader import load_model, rank_top_models, weight_status
from demo.vlm_responder import WildfireResponder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  데모 스모크 테스트")
print("=" * 55)

# 1. 가중치 상태
print("\n[가중치 상태]")
for n, folds in weight_status().items():
    print(f"  {n}: " + " ".join(f"fold{f}{'O' if ok else 'X'}" for f, ok in folds.items()))

top = rank_top_models(2)
print(f"\n[상위 2개 모델] {top}")

# 2. 테스트 이미지 한 장 (abnormal)
cands = glob.glob(os.path.join(ROOT, "data", "fireimage", "abnormal", "**", "*.jpg"),
                  recursive=True)
if not cands:
    print("테스트 이미지 없음 (data/fireimage/abnormal). 종료.")
    sys.exit(0)
img_path = cands[0]
img = np.array(Image.open(img_path).convert("RGB"))
print(f"[테스트 이미지] {os.path.relpath(img_path, ROOT)}  shape={img.shape}")

# 3. 두 모델 교차검증
responder = WildfireResponder(backend="auto")
results = []
for name in top:
    model, loaded = load_model(name, fold=0)
    pipe = FireDetectionPipeline(model=model, model_name=name,
                                 config=PipelineConfig(image_size=160))
    r = pipe.run(img, wind_direction=270.0)
    results.append((name, r))
    print(f"\n[{name}]  가중치로드={loaded}")
    print(f"  화재감지={r.fire_detected}  신뢰도={r.fire_confidence:.3f}  "
          f"경로={r.path_type}  지연={r.latency_ms:.0f}ms")

# 4. 교차검증 + VLM
nameA, rA = results[0]
nameB, rB = results[1]
agree = rA.fire_detected == rB.fire_detected
print(f"\n[교차검증] 일치={agree}  평균신뢰도={(rA.fire_confidence+rB.fire_confidence)/2:.3f}")

primary = rA if rA.fire_confidence >= rB.fire_confidence else rB
vlm = responder.respond(img, primary, wind_direction=270.0)
print("\n[VLM 대응 보고]")
print(vlm["text"])
print("\n" + "=" * 55)
print("  스모크 테스트 완료 — 배관 정상")
print("=" * 55)
