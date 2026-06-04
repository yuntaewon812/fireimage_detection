"""
app.py — 산불 감지·대응 데모 (Gradio)

핵심 설계:  성능평가 상위 2개 모델로 교차검증(앙상블)
  - 두 모델이 모두 화재 판정 → 신뢰도 높음 (일치)
  - 한쪽만 판정 → 주의 플래그 (불일치, 추가확인 권고)
  - 최종 신뢰도 = 두 모델 평균, GradCAM은 모델별로 나란히 표시

입력:  이미지 또는 영상 + 풍향(자동) + 위경도(자동)
처리:  모델 A,B 각각 FireDetectionPipeline → 비교 → VLM 대응 보고
실행:  python demo/app.py
풍향자동: set OPENWEATHER_API_KEY=<키> 후 실행 (없으면 수동 슬라이더)
"""
import os
import sys
import json
import urllib.request
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


def _read_video_frames(path, max_gap: int = 5):
    if isinstance(path, dict):
        path = (path.get("video") or {}).get("path") or path.get("path") or ""
    cap = cv2.VideoCapture(str(path))
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
    h, w = img_rgb.shape[:2]
    heat = cv2.resize(heat, (w, h))
    return (img_rgb * 0.55 + heat * 0.45).astype(np.uint8)


def fetch_wind_direction(lat: float, lon: float):
    """OpenWeatherMap API로 해당 좌표의 현재 풍향(도) 조회. 키 없으면 None."""
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        return None
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={api_key}")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        deg = float(data["wind"]["deg"])
        print(f"[풍향 API] {lat:.4f},{lon:.4f} → {deg:.0f}°")
        return deg
    except Exception as e:
        print(f"[풍향 API] 조회 실패: {e}")
        return None



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
        # type="filepath" → cv2로 읽기
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
    reliability = "높음 (두 모델 일치)" if agree else "낮음 (모델 불일치 — 추가확인 권고)"

    # 대응 보고는 더 신뢰도 높은 모델 결과 기준
    primary = rA if rA.fire_confidence >= rB.fire_confidence else rB

    # ── VLM 대응 보고 ──
    loc_str = f"{float(lat):.5f}, {float(lon):.5f}" if (lat and lon) else None
    vlm = RESPONDER.respond(img_rgb, primary, wind_direction=wind, location=loc_str)

    header = (
        f"### 🧪 모델 교차검증 (상위 2개)\n"
        f"| 모델 | 화재판정 | 신뢰도 | 경로 |\n"
        f"|---|---|---|---|\n"
        f"| {nameA} | {'🔴화재' if rA.fire_detected else '🟢정상'} | {rA.fire_confidence:.2f} | {rA.path_type} |\n"
        f"| {nameB} | {'🔴화재' if rB.fire_detected else '🟢정상'} | {rB.fire_confidence:.2f} | {rB.path_type} |\n\n"
        f"**판정 신뢰성: {reliability}**  ·  평균 신뢰도 {combined_conf:.2f}\n\n---\n"
    )
    report = header + vlm["text"]

    # ── 시각화 ──
    camA = _overlay_cam(img_rgb, rA.cam_map)
    camB = _overlay_cam(img_rgb, rB.cam_map)
    if primary.fire_mask is not None:
        h, w = img_rgb.shape[:2]
        mask_resized = cv2.resize(primary.fire_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_img = VLMAnalyzer.overlay_mask(img_rgb, mask_resized, primary.path_type)
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


def analyze_webcam(frame, wind_dir, use_wind, lat, lon):
    """웹캠 스트리밍용 — 2초마다 자동 호출."""
    if frame is None:
        return gr.update(), gr.update(), "웹캠 대기 중...", None, "{}", "", "⏳ 대기"
    result = analyze(frame, None, wind_dir, use_wind, lat, lon)
    camA, camB, report, mask, raw_json, map_html = result
    # 판정 결과를 상태 텍스트로 요약
    try:
        data = json.loads(raw_json)
        conf = data.get("combined_confidence", 0)
        agree = data.get("agreement", False)
        top = data.get("top_models", ["A", "B"])
        fires = [data[m]["fire"] for m in top if m in data]
        if all(fires):
            status = f"🔴 화재 감지 (신뢰도 {conf:.2f})"
        elif any(fires):
            status = f"⚠️ 불일치 — 추가 확인 필요 (신뢰도 {conf:.2f})"
        else:
            status = f"🟢 정상 (신뢰도 {conf:.2f})"
    except Exception:
        status = "분석 완료"
    return camA, camB, report, mask, raw_json, map_html, status


def build_ui():
    top = rank_top_models(2)
    has_weather_key = bool(os.environ.get("OPENWEATHER_API_KEY", ""))
    wind_label = "풍향 (자동조회됨, 수동 조정 가능)" if has_weather_key else "풍향 (수동 입력)"

    with gr.Blocks(title="산불 감지·대응 데모") as demo:
        gr.Markdown(f"# 🔥 산불 감지 · 대응 의사결정 데모\n"
                    f"성능평가 상위 2개 모델(**{top[0]}**, **{top[1]}**)로 교차검증 → "
                    f"GradCAM·연기방향·풍향 분석 → VLM 대응 보고.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("영상"):
                    in_video = gr.Video(label="화재 영상 (연기방향 정확)")
                with gr.Tab("웹캠 (실시간)"):
                    in_webcam = gr.Image(sources=["webcam"], type="numpy",
                                         label="웹캠 — 촬영 후 자동 분석")
                    webcam_auto = gr.Checkbox(False, label="자동 분석 (2초마다)")
                    webcam_status = gr.Markdown("웹캠으로 촬영하거나 자동 분석을 켜세요.")
                use_wind = gr.Checkbox(True, label="풍향 게이트 사용")
                wind = gr.Slider(0, 360, 270, step=1, label=wind_label)
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
                  inputs=[gr.State(None), in_video, wind, use_wind, lat, lon],
                  outputs=[out_camA, out_camB, out_report, out_mask, out_json, out_map])

        # 웹캠: 촬영 즉시 분석
        in_webcam.change(analyze_webcam,
                         inputs=[in_webcam, wind, use_wind, lat, lon],
                         outputs=[out_camA, out_camB, out_report, out_mask,
                                  out_json, out_map, webcam_status])

        # 웹캠: 자동 분석 체크 시 2초마다 반복
        timer = gr.Timer(value=2, active=False)
        webcam_auto.change(lambda on: gr.Timer(active=on), webcam_auto, timer)
        timer.tick(analyze_webcam,
                   inputs=[in_webcam, wind, use_wind, lat, lon],
                   outputs=[out_camA, out_camB, out_report, out_mask,
                             out_json, out_map, webcam_status])
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True, theme=gr.themes.Soft())
