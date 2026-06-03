"""
app.py — 산불 감지·대응 데모 (Gradio)

핵심 설계:  성능평가 상위 2개 모델로 교차검증(앙상블)
  - 두 모델이 모두 화재 판정 → 신뢰도 높음 (일치)
  - 한쪽만 판정 → 주의 플래그 (불일치, 추가확인 권고)
  - 최종 신뢰도 = 두 모델 평균, GradCAM은 모델별로 나란히 표시

입력:  이미지 또는 영상 + 풍향 + (선택)위경도
처리:  모델 A,B 각각 FireDetectionPipeline → 비교 → VLM 대응 보고
실행:  python demo/app.py
"""
import os
import sys
import json
import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.fire_pipeline import FireDetectionPipeline, PipelineConfig, VLMAnalyzer
from demo.model_loader import load_model, rank_top_models, weight_status
from demo.vlm_responder import WildfireResponder
from demo import map_utils

RESPONDER = WildfireResponder(backend="auto")


def _make_pipeline(model_name: str) -> FireDetectionPipeline:
    model, loaded = load_model(model_name, fold=0)
    pipe = FireDetectionPipeline(model=model, model_name=model_name,
                                 config=PipelineConfig(image_size=160))
    pipe._weights_loaded = loaded
    return pipe


def _read_video_frames(path: str, max_gap: int = 5):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None, None
    curr = frames[-1]
    prev = frames[-1 - max_gap] if len(frames) > max_gap else frames[0]
    return curr, prev


def _overlay_cam(img_rgb, cam):
    if cam is None:
        return img_rgb
    cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (img_rgb * 0.55 + heat * 0.45).astype(np.uint8)


def analyze(image, video, wind_dir, use_wind, lat, lon):
    if image is None and video is None:
        return None, None, "이미지 또는 영상을 입력하세요.", "", "{}", ""

    top = rank_top_models(2)
    wind = float(wind_dir) if use_wind else None

    # 입력 준비 (영상이면 Optical Flow용 prev_frame)
    prev_frame = None
    if video is not None:
        img_rgb, prev_frame = _read_video_frames(video)
        if img_rgb is None:
            return None, None, "영상을 읽을 수 없습니다.", "", "{}", ""
    else:
        img_rgb = image if isinstance(image, np.ndarray) else np.array(image)

    # ── 두 모델 각각 추론 ──
    runs = []
    for name in top:
        pipe = _make_pipeline(name)
        res = pipe.run(img_rgb, wind_direction=wind, prev_frame=prev_frame)
        runs.append((name, pipe, res))

    (nameA, _, rA), (nameB, _, rB) = runs[0], runs[1]

    # ── 교차검증 ──
    agree = rA.fire_detected == rB.fire_detected
    combined_conf = (rA.fire_confidence + rB.fire_confidence) / 2
    both_fire = rA.fire_detected and rB.fire_detected
    reliability = "높음 (두 모델 일치)" if agree else "낮음 (모델 불일치 — 추가확인 권고)"

    # 대응 보고는 더 신뢰도 높은(또는 화재 판정) 모델 결과 기준
    primary = rA if rA.fire_confidence >= rB.fire_confidence else rB

    # ── VLM 대응 보고 ──
    loc_str = f"{float(lat):.5f}, {float(lon):.5f}" if (lat and lon) else None
    vlm = RESPONDER.respond(img_rgb, primary, wind_direction=wind, location=loc_str)

    # 교차검증 헤더를 VLM 텍스트 앞에 부착
    header = (
        f"### 🧪 모델 교차검증 (상위 2개)\n"
        f"| 모델 | 화재판정 | 신뢰도 | 경로 |\n"
        f"|---|---|---|---|\n"
        f"| {nameA} | {'🔴화재' if rA.fire_detected else '🟢정상'} | {rA.fire_confidence:.2f} | {rA.path_type} |\n"
        f"| {nameB} | {'🔴화재' if rB.fire_detected else '🟢정상'} | {rB.fire_confidence:.2f} | {rB.path_type} |\n\n"
        f"**판정 신뢰성: {reliability}**  ·  평균 신뢰도 {combined_conf:.2f}\n\n---\n"
    )
    report = header + vlm["text"]

    # ── 시각화 (두 모델 GradCAM 나란히) ──
    camA = _overlay_cam(img_rgb, rA.cam_map)
    camB = _overlay_cam(img_rgb, rB.cam_map)
    if primary.fire_mask is not None:
        mask_img = VLMAnalyzer.overlay_mask(img_rgb, primary.fire_mask, primary.path_type)
    else:
        mask_img = img_rgb

    # ── 지도 ──
    map_html = ""
    if lat and lon:
        map_html = map_utils.build_map_html(
            float(lat), float(lon),
            spread_deg=primary.smoke_direction_deg,
            grade=vlm["intensity"]["grade"])

    raw = {
        "top_models": top,
        "agreement": agree,
        "combined_confidence": round(combined_conf, 3),
        nameA: {"fire": rA.fire_detected, "conf": round(rA.fire_confidence, 3),
                "path": rA.path_type, "latency_ms": round(rA.latency_ms, 1)},
        nameB: {"fire": rB.fire_detected, "conf": round(rB.fire_confidence, 3),
                "path": rB.path_type, "latency_ms": round(rB.latency_ms, 1)},
        "smoke_direction_deg": primary.smoke_direction_deg,
        "direction_source": primary.direction_source,
        "wind_aligned": primary.wind_aligned,
    }
    return (gr.update(value=camA, label=f"GradCAM — {nameA}"),
            gr.update(value=camB, label=f"GradCAM — {nameB}"),
            report, mask_img, json.dumps(raw, ensure_ascii=False, indent=2), map_html)


def build_ui():
    top = rank_top_models(2)
    with gr.Blocks(title="산불 감지·대응 데모", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 🔥 산불 감지 · 대응 의사결정 데모\n"
                    f"성능평가 상위 2개 모델(**{top[0]}**, **{top[1]}**)로 교차검증 → "
                    f"GradCAM·연기방향·풍향 분석 → VLM 대응 보고.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("이미지"):
                    in_image = gr.Image(label="화재 이미지", type="numpy")
                with gr.Tab("영상"):
                    in_video = gr.Video(label="화재 영상 (연기방향 정확)")
                use_wind = gr.Checkbox(True, label="풍향 게이트 사용")
                wind = gr.Slider(0, 360, 270, step=5, label="풍향 (0=북풍 90=동풍)")
                with gr.Row():
                    lat = gr.Number(label="위도 (선택)")
                    lon = gr.Number(label="경도 (선택)")
                btn = gr.Button("분석 실행", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    out_camA = gr.Image(label=f"GradCAM — {top[0]}")
                    out_camB = gr.Image(label=f"GradCAM — {top[1]}")
                out_mask = gr.Image(label="화재 마스크 / 경로")
                out_report = gr.Markdown()
                with gr.Accordion("원시 분석값", open=False):
                    out_json = gr.Code(language="json")
                out_map = gr.HTML()

        btn.click(analyze,
                  inputs=[in_image, in_video, wind, use_wind, lat, lon],
                  outputs=[out_camA, out_camB, out_report, out_mask, out_json, out_map])
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
