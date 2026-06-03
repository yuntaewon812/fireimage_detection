"""
vlm_responder.py — 파이프라인 분석 결과를 VLM에 넘겨 산불 대응 텍스트를 생성.

흐름:
  FireDetectionResult + 이미지
     │
     ├─ 연기색 분석   (검은/회백/흰 → 연소물질 추정)
     ├─ 강도 추정     (화염/연기 경로 + 면적 + 신뢰도 → 소형/중형/대형)
     ├─ 자원 권고     (강도별 헬기·진화차·인력 가이드라인)
     │
     └─ 위 신호들을 '근거'로 VLM이 종합 → 산불 상태/위치/대응/주의 텍스트 생성

VLM 백엔드:
  - AnthropicVLM : Claude 비전 (ANTHROPIC_API_KEY 필요, 고품질 한국어)  ← 기본
  - 백엔드 없으면 근거 데이터 기반 텍스트로 폴백 (동적 값 반영)
"""
from __future__ import annotations
import os
import base64
import io
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────
# 1. 보조 분석 (VLM 입력 근거 생성)
# ──────────────────────────────────────────────────────────────

def analyze_smoke_color(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """마스크 영역의 평균 밝기/채도로 연기색 분류 + 연소물질 추정."""
    roi = mask > 0.5
    if roi.sum() < 10:
        return {"color": "불명", "fuel_hint": "판단불가"}
    region = image_rgb[roi].astype(np.float32)
    brightness = float(region.mean())          # 0~255
    if brightness < 90:
        return {"color": "검은 연기", "fuel_hint": "유류·합성수지·시설물 연소 가능"}
    if brightness < 170:
        return {"color": "회백색 연기", "fuel_hint": "혼합 연소(초목+일부 인공물)"}
    return {"color": "흰색 연기", "fuel_hint": "초목 수분 증발(임야 화재 전형)"}


def estimate_intensity(result) -> dict:
    """경로 + 면적 + 신뢰도로 산불 강도와 권장 대응단계 추정."""
    cov = result.coverage_ratio
    conf = result.fire_confidence
    is_flame = result.path_type == "flame"

    # 화염 경로이거나 면적이 크면 강도 상향
    score = cov + (0.2 if is_flame else 0.0) + (conf - 0.5) * 0.3
    if score < 0.12:
        grade, stage = "소형", "1단계 (시·군·구)"
    elif score < 0.30:
        grade, stage = "중형", "2단계 (시·도)"
    else:
        grade, stage = "대형", "3단계 (산림청·중앙)"
    return {"grade": grade, "stage": stage,
            "coverage_pct": round(cov * 100, 1), "score": round(score, 3)}


def recommend_resources(grade: str, night: bool = False) -> dict:
    """강도별 진압 자원 가이드라인 (산림청 운용 기준 참고)."""
    table = {
        "소형": dict(heli="소형헬기 1대", truck="진화차 1대",
                    crew="지상진화 10명", action="국지 진화, 잔불 감시"),
        "중형": dict(heli="중형헬기 2대", truck="진화차 3대",
                    crew="특수진화대 1개반 + 지상 30명",
                    action="방화선 구축, 확산방향 차단"),
        "대형": dict(heli="대형/초대형헬기 3대+", truck="진화차 5대+",
                    crew="특수진화대 다수 + 지상 100명+",
                    action="광역 방화선, 주민대피, 유관기관 공조"),
    }
    rec = dict(table.get(grade, table["중형"]))
    if night:
        rec["action"] += " / 야간: 헬기 운용제한 → 지상진화 중심·일출 후 공중투입"
    return rec


# ──────────────────────────────────────────────────────────────
# 2. VLM 백엔드
# ──────────────────────────────────────────────────────────────

WILDFIRE_SYSTEM = (
    "당신은 산불 대응 의사결정 지원 AI입니다. 제공된 산불 현장 이미지와 분석 근거를 종합해 "
    "현장 지휘관이 즉시 활용할 수 있는 대응 보고를 한국어로 작성합니다. "
    "반드시 아래 4개 섹션으로 간결하게 작성하세요:\n"
    "🔴 산불 상태 (감지/강도/연소물질 추정)\n"
    "📍 발화 위치·확산 (이미지 내 위치, 확산 방향, 풍향 일치 여부)\n"
    "🚁 권장 대응 (대응단계, 장비, 인력, 우선조치)\n"
    "⚠️ 주의·위험요소\n"
    "근거 데이터를 그대로 나열하지 말고 지휘관 언어로 해석해 제시하세요."
)


def _build_grounding(result, smoke, intensity, resources,
                     wind_direction, location) -> str:
    lines = [
        f"- 화재감지: {result.fire_detected} (신뢰도 {result.fire_confidence:.2f}, 불확실성 {result.uncertainty:.2f})",
        f"- 경로분류: {result.path_type}",
        f"- 추정강도: {intensity['grade']} (영향면적 {intensity['coverage_pct']}%) → 권장 {intensity['stage']}",
        f"- 연기색: {smoke['color']} ({smoke['fuel_hint']})",
    ]
    if result.smoke_direction_deg is not None:
        lines.append(f"- 연기 이동방향: {result.smoke_direction_deg:.0f}° "
                     f"(추정원: {result.direction_source})")
    if result.wind_aligned is not None:
        lines.append(f"- 풍향 일치: {'예' if result.wind_aligned else '아니오'} "
                     f"(cosine {result.wind_align_score:.2f})")
    if wind_direction is not None:
        lines.append(f"- 입력 풍향: {wind_direction:.0f}°")
    if location:
        lines.append(f"- 위치(위경도): {location}")
    lines.append(f"- 자원 가이드: 헬기 {resources['heli']}, {resources['truck']}, "
                 f"인력 {resources['crew']}, 조치 {resources['action']}")
    return "\n".join(lines)


class AnthropicVLM:
    """Claude 비전 기반 (ANTHROPIC_API_KEY 필요)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic   # 지연 import
        self.client = Anthropic()
        self.model = model

    def generate(self, image_rgb, grounding: str) -> str:
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            system=WILDFIRE_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
                {"type": "text", "text": f"[분석 근거]\n{grounding}\n\n위 현장과 근거로 대응 보고를 작성하세요."},
            ]}],
        )
        return msg.content[0].text


def _fallback_text(result, smoke, intensity, resources,
                   wind_direction, location) -> str:
    """VLM 백엔드 미설정 시 근거 기반 텍스트(동적 값 반영)."""
    dir_str = (f"{result.smoke_direction_deg:.0f}° ({result.direction_source})"
               if result.smoke_direction_deg is not None else "추정불가")
    wind_str = "—"
    if result.wind_aligned is not None:
        wind_str = ("풍향 일치 ✓ (실제 산불 가능성↑)" if result.wind_aligned
                    else "풍향 불일치 ✗ (증기·안개 등 오탐 가능성)")
    loc_str = location or "GPS 정보 없음 (영상 내 GradCAM 위치 참조)"
    return (
        f"🔴 **산불 상태**\n"
        f"- 감지: {'화재' if result.fire_detected else '미감지/오탐의심'} "
        f"(신뢰도 {result.fire_confidence*100:.0f}%)\n"
        f"- 강도: **{intensity['grade']}** (영향면적 {intensity['coverage_pct']}%)\n"
        f"- 연소물질 추정: {smoke['color']} → {smoke['fuel_hint']}\n\n"
        f"📍 **발화 위치·확산**\n"
        f"- 위치: {loc_str}\n"
        f"- 확산 방향: {dir_str}\n"
        f"- {wind_str}\n\n"
        f"🚁 **권장 대응 — {intensity['stage']}**\n"
        f"- 장비: {resources['heli']}, {resources['truck']}\n"
        f"- 인력: {resources['crew']}\n"
        f"- 우선조치: {resources['action']}\n\n"
        f"⚠️ **주의**\n"
        f"- 풍속 상승 시 수관화·비화 전환 위험\n"
        f"- 경사 상향부 확산 가속 주의\n"
        f"\n_(VLM 백엔드 미설정 — 근거 기반 자동생성. ANTHROPIC_API_KEY 설정 시 VLM 생성 텍스트로 대체)_"
    )


# ──────────────────────────────────────────────────────────────
# 3. 통합 진입점
# ──────────────────────────────────────────────────────────────

class WildfireResponder:
    def __init__(self, backend: str = "auto"):
        self.vlm = None
        if backend in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                self.vlm = AnthropicVLM()
            except Exception as e:
                print(f"[VLM] Anthropic 초기화 실패 → 폴백: {e}")

    def respond(self, image_rgb, result, wind_direction=None,
                location: str | None = None, night: bool = False) -> dict:
        smoke = analyze_smoke_color(image_rgb,
                                    result.fire_mask if result.fire_mask is not None
                                    else np.ones(image_rgb.shape[:2], np.float32))
        intensity = estimate_intensity(result)
        resources = recommend_resources(intensity["grade"], night=night)
        grounding = _build_grounding(result, smoke, intensity, resources,
                                     wind_direction, location)

        if self.vlm is not None:
            try:
                text = self.vlm.generate(image_rgb, grounding)
            except Exception as e:
                text = _fallback_text(result, smoke, intensity, resources,
                                      wind_direction, location)
                text += f"\n\n_(VLM 호출 실패: {e})_"
        else:
            text = _fallback_text(result, smoke, intensity, resources,
                                  wind_direction, location)

        return {"text": text, "smoke": smoke,
                "intensity": intensity, "resources": resources}
