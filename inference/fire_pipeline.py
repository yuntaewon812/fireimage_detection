"""
화재 감지 분기 추론 파이프라인
=====================================
경로 구성 및 목표 지연:

  Flame Path  (낙뢰·방화): GradCAM + HSV mask + VLM              ~80ms
              속도 최우선 — 골든타임 내 즉각 알람

  Smoke Low   (저불확실): GradCAM + 풍향 게이트 + VLM            ~120ms
              풍향 일치 확인으로 오탐(증기·안개·먼지) 필터링

  Smoke High  (고불확실): GradCAM + 풍향 게이트 + SAM + VLM      ~400ms
              판단 어려운 케이스에만 SAM 정밀 마스크 적용

설계 원칙:
  - 대부분의 화재는 연기에서 시작 → Smoke path가 주경로
  - 낙뢰·방화처럼 즉각 발화하는 케이스만 Flame path
  - 풍향 불일치 연기 → 오탐 플래그 (산불이 아닐 가능성 높음)
  - SAM은 불확실할 때만 → 처리 속도 희생 최소화
"""

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import torchvision.transforms as transforms


# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # 라우팅 임계값
    fire_threshold: float = 0.50        # 화재 감지 최소 신뢰도 (softmax)
    flame_score_threshold: float = 0.25 # GradCAM 영역 내 화염 색상 비율 임계값
    uncertainty_threshold: float = 0.35 # Shannon 엔트로피 기반 불확실성 임계값

    # 풍향 게이트
    wind_align_threshold: float = 0.30  # cosine 유사도 임계값 (≈72° 이내 = 일치)

    # GradCAM 이진화
    cam_threshold: float = 0.40

    # Optical Flow
    flow_cam_threshold: float = 0.35   # flow 추출에 쓸 CAM 임계값 (GradCAM보다 느슨하게)
    flow_min_magnitude: float = 0.5    # 이 픽셀/프레임 미만이면 정지로 간주 → PCA 폴백

    # Ablation 플래그 — False 로 설정하면 해당 모듈을 비활성화
    use_hsv: bool = True          # False → 항상 Smoke Path (HSV 라우팅 제거)
    use_optical_flow: bool = True # False → 항상 PCA 방향 추정 (OF 제거)
    use_wind_gate: bool = True    # False → 풍향 게이트 스킵 (모든 연기 통과)
    use_sam: bool = True          # False → Smoke High 경로에서 GradCAM 마스크 사용
    use_uncertainty: bool = True  # False → 항상 Smoke Low 경로 (불확실성 라우팅 제거)

    # 이미지 크기
    image_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# 결과 데이터클래스
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FireDetectionResult:
    # 경로 및 감지
    path_type: str = "normal"           # 'normal'|'flame'|'smoke_low'|'smoke_high'
    fire_detected: bool = False
    fire_confidence: float = 0.0
    uncertainty: float = 0.0

    # 마스크·면적
    cam_map: Optional[np.ndarray] = None    # GradCAM heatmap  [H, W] float32
    fire_mask: Optional[np.ndarray] = None  # 최종 이진 마스크 [H, W] float32
    coverage_ratio: float = 0.0             # 화재 영역 비율 [0, 1]

    # 연기 경로 전용
    wind_aligned: Optional[bool] = None
    wind_align_score: float = 0.0
    smoke_direction_deg: Optional[float] = None  # 연기 이동 방향 (기상학적 각도)

    # Optical Flow 결과
    flow_vector: Optional[Tuple[float, float]] = None  # (vx, vy) 연기 영역 평균 flow
    flow_magnitude: float = 0.0                        # 평균 이동 크기 (픽셀/프레임)
    flow_map: Optional[np.ndarray] = None              # (H, W, 2) 전체 flow 맵
    direction_source: str = "none"                     # 'optical_flow' | 'pca' | 'none'

    # VLM 분석 결과
    vlm_result: Dict[str, Any] = field(default_factory=dict)

    # 처리 시간 (ms)
    latency_ms: float = 0.0
    latency_breakdown: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# GradCAM 추출기
# ──────────────────────────────────────────────────────────────────────────────

class GradCAMExtractor:
    """
    커스텀 GradCAM 구현.
    pytorch_grad_cam과 달리 channels-last 모델(InternImage)도 지원.
    forward/backward hook으로 중간 특징맵과 기울기를 캡처.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 channels_last: bool = False):
        self.model = model
        self.channels_last = channels_last
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        feat = out
        if self.channels_last and feat.ndim == 4:
            feat = feat.permute(0, 3, 1, 2).contiguous()   # (B,H,W,C) → (B,C,H,W)
        self._act = feat.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        grad = grad_out[0]
        if grad is None:
            return
        if self.channels_last and grad.ndim == 4:
            grad = grad.permute(0, 3, 1, 2).contiguous()
        self._grad = grad.detach()

    def extract(self, image_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Args:
            image_tensor: (1, 3, H, W) 정규화 텐서
            class_idx: GradCAM 대상 클래스 (1=화재)
        Returns:
            cam: [H, W] float32, [0, 1] 정규화
        """
        dev = next(self.model.parameters()).device
        image_tensor = image_tensor.to(dev)

        self.model.eval()
        self.model.zero_grad()

        output = self.model(image_tensor)   # labels=None → logits만 반환
        output[:, class_idx].backward()

        # 채널 방향 가중 평균 (Grad-CAM 공식)
        weights = self._grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 원본 이미지 크기로 업샘플
        cam = F.interpolate(cam, size=image_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)

        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        return cam_np


def get_target_layer(
    model: torch.nn.Module, model_name: str
) -> Tuple[torch.nn.Module, bool]:
    """
    모델 이름으로 GradCAM 대상 레이어를 자동 선택.
    Returns: (layer, channels_last)
    """
    n = model_name.lower()

    if "efficientnetv2" in n:
        return model.backbone.blocks[-1], False
    if "nextvit" in n:
        return model.backbone.features[-1], False
    if "maxvit" in n:
        # MaxViTBlock 출력은 (B, C, H, W)
        return model.backbone.stages[-1][-1], False
    if "internimage" in n:
        # InternImageBlock 출력은 (B, H, W, C) → channels_last=True
        return model.backbone.stages[-1][-1], True
    if "swintransformerv2" in n or "swinv2" in n:
        return model.backbone.layers[-1].blocks[-1], False
    if "convnextv2" in n:
        return model.backbone.stages[-1][-1], False
    if "densenet" in n:
        # DeepModel wraps backbone as self.feature_extractor (Sequential)
        # last child = norm5 (BatchNorm2d), followed by CBAM
        return model.feature_extractor[-1], False
    if "resnet" in n:
        # DeepModel wraps backbone as self.feature_extractor (Sequential)
        # last child = layer4 (Sequential of Bottleneck blocks)
        return model.feature_extractor[-1], False

    raise ValueError(f"GradCAM 대상 레이어 미지원 모델: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# HSV 화염 마스크 (Flame Path 전용)
# ──────────────────────────────────────────────────────────────────────────────

class HSVFlameMask:
    """
    HSV 색공간에서 화염 특유의 주황·빨강·노랑 영역을 50ms 이내로 추출.

    SAM 대비 1/7 처리 시간으로 낙뢰·방화 즉각 발화 케이스에 충분한 마스크를 제공.
    GradCAM 활성화 영역 내 화염 색상 비율(flame_score)로 화염/연기 경로를 분기.
    """

    # cv2 HSV 기준: H=[0,179], S=[0,255], V=[0,255]
    _RANGES = [
        ((0,   140,  80), (10,  255, 255)),   # 빨강 (저)
        ((160, 140,  80), (179, 255, 255)),   # 빨강 (고)
        ((10,  140,  80), (25,  255, 255)),   # 주황
        ((25,  140, 130), (35,  255, 255)),   # 노랑
    ]
    _KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """화염 색상 이진 마스크 반환. [0, 1] float32"""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self._RANGES:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._KERNEL)
        return (mask / 255.0).astype(np.float32)

    def flame_score(self, image_rgb: np.ndarray, cam: np.ndarray,
                    cam_threshold: float = 0.40) -> float:
        """
        GradCAM 고활성 영역 내 화염 색상 픽셀 비율.
        flame_score >= 0.25 → Flame Path로 라우팅.
        """
        roi = (cam > cam_threshold).astype(np.uint8)
        active = int(roi.sum())
        if active == 0:
            return 0.0
        flame_px = float((self.extract(image_rgb) * roi).sum())
        return flame_px / active


# ──────────────────────────────────────────────────────────────────────────────
# 연기 방향 추정기
# ──────────────────────────────────────────────────────────────────────────────

class SmokeAnalyzer:
    """
    GradCAM 히트맵에서 연기 drift 방향을 이미지 모멘트(PCA)로 추정.
    실제 풍향과 비교하여 오탐 여부를 판단하는 Gate1의 입력으로 사용.
    """

    def estimate_direction(self, cam: np.ndarray,
                           threshold: float = 0.40) -> Optional[float]:
        """
        GradCAM 활성 픽셀의 주축 방향을 기상학 각도로 반환.

        Returns:
            direction_deg: 연기가 향하는 방향 (0=북, 90=동, 180=남, 270=서)
                           None: 활성 픽셀이 너무 적어 추정 불가
        """
        ys, xs = np.where(cam > threshold)
        if len(ys) < 20:
            return None

        pts = np.column_stack([ys, xs]).astype(np.float64)
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        _, evecs = np.linalg.eigh(cov)

        # 최대 분산 방향 (주축)
        principal = evecs[:, -1]        # [row_dir, col_dir]
        dy, dx = principal[0], principal[1]

        # 기상학 각도 변환 (row↓=남, col→=동)
        angle_deg = float(np.degrees(np.arctan2(dx, -dy)) % 360)
        return angle_deg

    def smoke_mask(self, image_rgb: np.ndarray, cam: np.ndarray,
                   threshold: float = 0.40) -> np.ndarray:
        """
        GradCAM 이진화 + 낮은 채도 영역(연기 특성) 강화 마스크.
        저채도(흰색·회색) 픽셀을 연기 영역으로 보정.
        """
        cam_bin = (cam > threshold).astype(np.float32)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        low_sat = (hsv[:, :, 1] < 80).astype(np.float32)   # 채도 < 80 → 연기
        combined = cam_bin * (0.6 + 0.4 * low_sat)
        return np.clip(combined, 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Optical Flow 기반 연기 방향 추정기
# ──────────────────────────────────────────────────────────────────────────────

class OpticalFlowAnalyzer:
    """
    연속 프레임 간 Farneback Optical Flow로 연기 이동 방향 추정.

    SmokeAnalyzer(PCA) 대비 장점:
      - 정적 GradCAM 형태가 아닌 실제 픽셀 이동 벡터 사용
      - 연기와 구름의 물리적 움직임 차이를 직접 포착
      - 풍향 게이트 입력 정확도 향상

    사용:
      prev_frame이 있으면 OpticalFlow 방향 사용.
      없으면 SmokeAnalyzer PCA로 자동 폴백 (파이프라인이 관리).
    """

    def estimate_direction(
        self,
        frame_curr: np.ndarray,     # (H, W, 3) uint8 RGB 현재 프레임
        frame_prev: np.ndarray,     # (H, W, 3) uint8 RGB 이전 프레임
        cam: np.ndarray,            # (H, W) float32 GradCAM 히트맵
        cam_threshold: float = 0.35,
        min_magnitude: float = 0.5,
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """
        GradCAM 활성 영역 내 평균 flow 벡터로 연기 이동 방향 반환.

        Returns:
            direction_deg: 기상학 각도 (0=북, 90=동, 180=남, 270=서).
                           None: 활성 픽셀 부족 or 이동량 너무 작음 → PCA 폴백 권장.
            magnitude:     평균 이동 크기 (픽셀/프레임)
            flow:          (H, W, 2) 전체 flow 맵 (시각화용)
        """
        # 두 프레임을 동일 크기로 맞춤 (pipeline 전처리 후에도 크기 차이 방지)
        if frame_prev.shape != frame_curr.shape:
            frame_prev = cv2.resize(frame_prev, (frame_curr.shape[1], frame_curr.shape[0]))

        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # shape: (H, W, 2) — [..., 0]=fx(동서), [..., 1]=fy(남북↓)

        # GradCAM 활성 영역(연기 의심 영역)에서만 flow 추출
        smoke_roi = cam > cam_threshold
        n_active = int(smoke_roi.sum())
        if n_active < 20:
            return None, 0.0, flow

        fx = float(flow[smoke_roi, 0].mean())
        fy = float(flow[smoke_roi, 1].mean())
        magnitude = float(np.sqrt(fx ** 2 + fy ** 2))

        # 이동량이 너무 작으면 정지로 간주 → PCA 폴백이 더 신뢰성 있음
        if magnitude < min_magnitude:
            return None, magnitude, flow

        # 기상학 각도 변환
        # OpenCV 좌표: x→동(+), y→남(+)  /  기상학: 0=북, 시계방향
        direction_deg = float(np.degrees(np.arctan2(fx, -fy)) % 360)
        return direction_deg, magnitude, flow

    def flow_overlay(self, image_rgb: np.ndarray, flow: np.ndarray,
                     cam: np.ndarray, cam_threshold: float = 0.35,
                     step: int = 16) -> np.ndarray:
        """
        연기 활성 영역에 flow 화살표를 그린 시각화 이미지 반환.
        step: 화살표 격자 간격 (픽셀)
        """
        vis = image_rgb.copy()
        h, w = vis.shape[:2]
        smoke_roi = cam > cam_threshold

        for y in range(0, h, step):
            for x in range(0, w, step):
                if not smoke_roi[y, x]:
                    continue
                fx = float(flow[y, x, 0])
                fy = float(flow[y, x, 1])
                mag = np.sqrt(fx ** 2 + fy ** 2)
                if mag < 0.3:
                    continue
                scale = min(step * 0.8, mag * 3)
                ex = int(x + fx / mag * scale)
                ey = int(y + fy / mag * scale)
                cv2.arrowedLine(vis, (x, y), (ex, ey),
                                (0, 255, 100), 1, tipLength=0.3)
        return vis


# ──────────────────────────────────────────────────────────────────────────────
# Gate1 — 풍향 정렬 게이트
# ──────────────────────────────────────────────────────────────────────────────

class WindAlignmentGate:
    """
    연기 이동 방향과 실제 풍향의 물리적 일치 여부를 확인.

    원리:
      바람이 북→남(풍향=180°)으로 불면 연기는 남쪽으로 drift.
      연기 drift 방향과 (풍향+180°) 사이 각도 차가 ≤72° → '일치'로 판정.

    일치하면: 실제 산불 연기일 가능성 ↑
    불일치하면: 증기·안개·먼지 등 오탐 가능성 ↑ → fire_detected=False 처리
    """

    def __init__(self, align_threshold: float = 0.30):
        # cosine 임계값: 0.30 ≈ 72.5° 이내
        self.align_threshold = align_threshold

    def check(self, smoke_dir_deg: Optional[float],
              wind_dir_deg: float) -> Tuple[bool, float]:
        """
        Args:
            smoke_dir_deg: 연기 이동 방향 (SmokeAnalyzer.estimate_direction)
            wind_dir_deg:  풍향 (바람이 불어오는 방향, 기상학 관례)
        Returns:
            aligned: True면 물리적으로 일치 → 실제 연기로 처리
            cosine_score: [-1, 1]
        """
        if smoke_dir_deg is None:
            return True, 0.0  # 방향 추정 불가 → 보수적으로 통과

        # 연기 이동 방향의 기대값 = 풍향의 반대
        expected = (wind_dir_deg + 180.0) % 360.0

        diff = abs(smoke_dir_deg - expected) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff

        score = float(np.cos(np.radians(diff)))
        return score >= self.align_threshold, score


# ──────────────────────────────────────────────────────────────────────────────
# 불확실성 측정
# ──────────────────────────────────────────────────────────────────────────────

def compute_uncertainty(probs: np.ndarray) -> float:
    """Shannon 엔트로피 기반 불확실성 [0, 1] 정규화."""
    p = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_ent = np.log(max(len(p), 2))
    return float(entropy / max_ent)


# ──────────────────────────────────────────────────────────────────────────────
# SAM 세그멘터 (Smoke High Path 전용)
# ──────────────────────────────────────────────────────────────────────────────

class SAMSegmenter:
    """
    불확실 연기 케이스에만 적용하는 정밀 마스크 추출기.

    segment_anything 설치 시: SAM point-prompt 추론
    미설치 시: GrabCut 기반 폴백 (처리 시간 ~50ms, SAM 대비 정밀도 낮음)

    SAM 체크포인트 다운로드:
      wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 model_type: str = "vit_b"):
        self._predictor = None
        if checkpoint:
            self._load_sam(checkpoint, model_type)

    def _load_sam(self, checkpoint: str, model_type: str):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self._predictor = SamPredictor(sam.to(dev))
        except ImportError:
            pass  # GrabCut 폴백 사용

    def segment(self, image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
        """
        Args:
            image_rgb: (H, W, 3) uint8 RGB
            cam: (H, W) float32 GradCAM heatmap
        Returns:
            mask: (H, W) float32 [0, 1]
        """
        if self._predictor is not None:
            return self._sam_segment(image_rgb, cam)
        return self._grabcut_segment(image_rgb, cam)

    def _sam_segment(self, image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
        peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
        self._predictor.set_image(image_rgb)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[peak_x, peak_y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        return best.astype(np.float32)

    def _grabcut_segment(self, image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
        """SAM 미설치 폴백: GrabCut 기반 세그멘테이션."""
        h, w = cam.shape
        ys, xs = np.where(cam > 0.4)
        if len(ys) == 0:
            return (cam > 0.4).astype(np.float32)

        m = 10  # 마진
        x1, y1 = max(0, xs.min() - m), max(0, ys.min() - m)
        x2, y2 = min(w - 1, xs.max() + m), min(h - 1, ys.max() + m)
        rect = (x1, y1, x2 - x1, y2 - y1)

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        mask_gc = np.zeros((h, w), np.uint8)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        return np.where((mask_gc == 2) | (mask_gc == 0), 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# VLM 분석기
# ──────────────────────────────────────────────────────────────────────────────

class VLMAnalyzer:
    """화재 마스크 영역을 VLM으로 분석하는 추상 인터페이스."""

    _FLAME_PROMPT = (
        "A wildfire FLAME region is highlighted. "
        "Analyze: 1) risk_level 1-5 (5=extreme), "
        "2) spread_direction (N/NE/E/SE/S/SW/W/NW), "
        "3) coverage_pct (estimated %). "
        "Reply ONLY in JSON: {risk_level, spread_direction, coverage_pct, description}"
    )
    _SMOKE_PROMPT = (
        "A wildfire SMOKE region is highlighted. "
        "Analyze: 1) risk_level 1-5, "
        "2) drift_direction (N/NE/E/SE/S/SW/W/NW), "
        "3) density (light/medium/dense), "
        "4) coverage_pct (estimated %). "
        "Reply ONLY in JSON: {risk_level, drift_direction, density, coverage_pct, description}"
    )

    def analyze(self, image_rgb: np.ndarray, mask: np.ndarray,
                path_type: str) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                     path_type: str) -> np.ndarray:
        """마스크를 원본 이미지에 오버레이하여 VLM 입력 이미지 생성."""
        overlay = image_rgb.astype(np.float32)
        roi = mask > 0.5
        color = np.array([255, 80, 0]) if path_type == "flame" else np.array([180, 180, 255])
        overlay[roi] = overlay[roi] * 0.45 + color * 0.55
        return np.clip(overlay, 0, 255).astype(np.uint8)


class MockVLMAnalyzer(VLMAnalyzer):
    """테스트·개발용 Mock. 실제 VLM 없이 파이프라인 전체 흐름을 검증할 때 사용."""

    def analyze(self, image_rgb: np.ndarray, mask: np.ndarray,
                path_type: str) -> Dict[str, Any]:
        cov = float(mask.mean())
        if path_type == "flame":
            return dict(risk_level=min(5, max(1, int(cov * 12) + 2)),
                        spread_direction="NE", coverage_pct=round(cov * 100, 1),
                        description="[MOCK] Flame region detected.", vlm_model="mock")
        return dict(risk_level=min(5, max(1, int(cov * 8) + 1)),
                    drift_direction="E", density="medium",
                    coverage_pct=round(cov * 100, 1),
                    description="[MOCK] Smoke region detected.", vlm_model="mock")


class BLIPVLMAnalyzer(VLMAnalyzer):
    """
    HuggingFace BLIP-2 기반 로컬 VLM.
    requirements.txt의 transformers==5.5.4 환경에서 동작.

    Usage:
        vlm = BLIPVLMAnalyzer()          # 첫 실행 시 모델 다운로드
        pipeline = FireDetectionPipeline(model, "maxvit", vlm=vlm)
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = Blip2Processor.from_pretrained(model_name)
        self._vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(dev)
        self._vlm.eval()
        self._dev = dev

    def analyze(self, image_rgb: np.ndarray, mask: np.ndarray,
                path_type: str) -> Dict[str, Any]:
        vis = self.overlay_mask(image_rgb, mask, path_type)
        prompt = self._FLAME_PROMPT if path_type == "flame" else self._SMOKE_PROMPT
        inputs = self._proc(Image.fromarray(vis), text=prompt,
                            return_tensors="pt").to(self._dev, torch.float16)
        with torch.no_grad():
            out = self._vlm.generate(**inputs, max_new_tokens=150)
        text = self._proc.decode(out[0], skip_special_tokens=True)
        return {"raw_response": text, "vlm_model": "blip2"}


# ──────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────────────────────

class FireDetectionPipeline:
    """
    화재 감지 분기 추론 파이프라인.

    Usage (기본):
        pipeline = FireDetectionPipeline(
            model=trained_model,
            model_name="efficientnetv2",        # or "maxvit", "nextvit", ...
        )
        result = pipeline.run("wildfire.jpg", wind_direction=270.0, verbose=True)

    Usage (SAM + BLIP 풀 구성):
        pipeline = FireDetectionPipeline(
            model=trained_model,
            model_name="maxvit",
            vlm=BLIPVLMAnalyzer(),
            sam_checkpoint="sam_vit_b_01ec64.pth",
        )

    wind_direction 규칙 (기상학):
        0 = 북풍(북→남), 90 = 동풍(동→서), 180 = 남풍(남→북), 270 = 서풍(서→동)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        vlm: Optional[VLMAnalyzer] = None,
        sam_checkpoint: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig()
        self.model = model
        self.model_name = model_name
        self.device = torch.device(self.config.device)
        self.model.to(self.device).eval()

        # 컴포넌트 초기화
        self.hsv_mask = HSVFlameMask()
        self.smoke_analyzer = SmokeAnalyzer()
        self.flow_analyzer = OpticalFlowAnalyzer()
        self.wind_gate = WindAlignmentGate(self.config.wind_align_threshold)
        self.sam = SAMSegmenter(checkpoint=sam_checkpoint)
        self.vlm = vlm or MockVLMAnalyzer()

        # GradCAM: 모델별 타겟 레이어 자동 선택
        target_layer, ch_last = get_target_layer(model, model_name)
        self.grad_cam = GradCAMExtractor(model, target_layer, ch_last)

        self._tf = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
        ])

    # ── 내부 헬퍼 ─────────────────────────────────────────────

    def _preprocess(self, src) -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(src, str):
            pil = Image.open(src).convert("RGB")
        elif isinstance(src, np.ndarray):
            pil = Image.fromarray(src)
        else:
            pil = src
        pil = pil.resize((self.config.image_size, self.config.image_size))
        img_np = np.array(pil)                     # (H, W, 3) uint8
        tensor = self._tf(pil).unsqueeze(0)        # (1, 3, H, W)
        return tensor, img_np

    def _classify(self, tensor: torch.Tensor) -> Tuple[float, float, np.ndarray]:
        with torch.no_grad():
            logits = self.model(tensor.to(self.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return float(probs[1]), compute_uncertainty(probs), probs

    # ── 메인 추론 ─────────────────────────────────────────────

    def run(
        self,
        image_input,
        wind_direction: Optional[float] = None,
        prev_frame: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> FireDetectionResult:
        """
        Args:
            image_input:    파일 경로(str) | numpy uint8 RGB | PIL.Image
            wind_direction: 풍향 (0=북풍, 90=동풍 ...). None이면 풍향 게이트 스킵.
            prev_frame:     이전 프레임 numpy uint8 RGB (Optical Flow용).
                            None이면 PCA 기반 연기 방향 추정으로 폴백.
            verbose:        경로·처리시간 콘솔 출력
        Returns:
            FireDetectionResult
        """
        result = FireDetectionResult()
        t0 = time.perf_counter()
        timing: Dict[str, float] = {}

        # ── Step 1. 전처리 ─────────────────────────────────────
        _t = time.perf_counter()
        tensor, img_np = self._preprocess(image_input)
        timing["preprocess"] = (time.perf_counter() - _t) * 1000

        # ── Step 2. 분류 ───────────────────────────────────────
        _t = time.perf_counter()
        fire_prob, uncertainty, probs = self._classify(tensor)
        timing["classify"] = (time.perf_counter() - _t) * 1000

        result.fire_confidence = fire_prob
        result.uncertainty = uncertainty

        # 화재 미감지 → 종료
        if fire_prob < self.config.fire_threshold:
            result.path_type = "normal"
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.latency_breakdown = timing
            if verbose:
                self._log(result)
            return result

        result.fire_detected = True

        # ── Step 3. GradCAM ────────────────────────────────────
        _t = time.perf_counter()
        cam = self.grad_cam.extract(tensor, class_idx=1)
        result.cam_map = cam
        timing["gradcam"] = (time.perf_counter() - _t) * 1000

        # ── Step 4. 화염 vs 연기 판단 ─────────────────────────
        # HSV 분석: GradCAM 활성 영역 내 화염 색상 비율
        _t = time.perf_counter()
        if self.config.use_hsv:
            flame_sc = self.hsv_mask.flame_score(img_np, cam, self.config.cam_threshold)
            is_flame = flame_sc >= self.config.flame_score_threshold
        else:
            flame_sc = 0.0
            is_flame = False   # HSV 비활성 → 항상 Smoke Path
        timing["hsv"] = (time.perf_counter() - _t) * 1000

        # ══════════════════════════════════════════════════════
        # FLAME PATH  ── 낙뢰·방화 등 즉각 발화
        # 정확성보다 속도 최우선. 골든타임 내 알람 송출.
        # ══════════════════════════════════════════════════════
        if is_flame:
            _t = time.perf_counter()
            raw_mask = self.hsv_mask.extract(img_np)
            # GradCAM 활성 영역으로 마스크 제한
            fire_mask = raw_mask * (cam > self.config.cam_threshold).astype(np.float32)
            result.fire_mask = fire_mask
            result.coverage_ratio = float(fire_mask.mean())
            timing["flame_mask"] = (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            result.vlm_result = self.vlm.analyze(img_np, fire_mask, "flame")
            timing["vlm"] = (time.perf_counter() - _t) * 1000

            result.path_type = "flame"

        # ══════════════════════════════════════════════════════
        # SMOKE PATH  ── 대부분 화재의 주경로 (연기에서 시작)
        # ══════════════════════════════════════════════════════
        else:
            # ── Gate1: 연기 방향 추정 (Optical Flow > PCA 폴백) ──
            _t = time.perf_counter()

            if prev_frame is not None and self.config.use_optical_flow:
                # Optical Flow: 실제 픽셀 이동 벡터로 방향 추정
                smoke_dir, flow_mag, flow_map = self.flow_analyzer.estimate_direction(
                    frame_curr=img_np,
                    frame_prev=prev_frame,
                    cam=cam,
                    cam_threshold=self.config.flow_cam_threshold,
                    min_magnitude=self.config.flow_min_magnitude,
                )
                result.flow_magnitude = flow_mag
                result.flow_map = flow_map
                if flow_mag > 0:
                    result.flow_vector = (
                        float(flow_map[cam > self.config.flow_cam_threshold, 0].mean()),
                        float(flow_map[cam > self.config.flow_cam_threshold, 1].mean()),
                    )

                if smoke_dir is not None:
                    result.direction_source = "optical_flow"
                else:
                    # flow 이동량 부족 → PCA 폴백
                    smoke_dir = self.smoke_analyzer.estimate_direction(
                        cam, self.config.cam_threshold)
                    result.direction_source = "pca" if smoke_dir is not None else "none"
            else:
                # prev_frame 없음 → PCA 폴백
                smoke_dir = self.smoke_analyzer.estimate_direction(
                    cam, self.config.cam_threshold)
                result.direction_source = "pca" if smoke_dir is not None else "none"

            result.smoke_direction_deg = smoke_dir

            if wind_direction is not None and self.config.use_wind_gate:
                aligned, align_sc = self.wind_gate.check(smoke_dir, wind_direction)
                result.wind_aligned = aligned
                result.wind_align_score = align_sc

                # 풍향 불일치 → 오탐 가능성 높음 (증기·안개·먼지)
                if not aligned:
                    timing["wind_gate"] = (time.perf_counter() - _t) * 1000
                    smoke_mask = self.smoke_analyzer.smoke_mask(img_np, cam)
                    result.fire_mask = smoke_mask
                    result.coverage_ratio = float(smoke_mask.mean())
                    result.fire_detected = False          # 오탐 처리
                    result.path_type = "smoke_low"
                    result.vlm_result = {
                        "risk_level": 1,
                        "false_positive_flag": True,
                        "description": (
                            f"풍향({wind_direction:.0f}°)과 연기 방향({smoke_dir:.1f}° 추정)이 "
                            f"불일치 (cosine={align_sc:.2f}). "
                            "증기·안개·먼지 등 오탐 가능성. 추가 확인 권고."
                        ),
                    }
                    result.latency_ms = (time.perf_counter() - t0) * 1000
                    result.latency_breakdown = timing
                    if verbose:
                        self._log(result)
                    return result
            timing["wind_gate"] = (time.perf_counter() - _t) * 1000

            # ── 연기 저불확실 경로 ────────────────────────────
            if not self.config.use_uncertainty or uncertainty < self.config.uncertainty_threshold:
                _t = time.perf_counter()
                smoke_mask = self.smoke_analyzer.smoke_mask(img_np, cam)
                result.fire_mask = smoke_mask
                result.coverage_ratio = float(smoke_mask.mean())
                timing["smoke_mask"] = (time.perf_counter() - _t) * 1000

                _t = time.perf_counter()
                result.vlm_result = self.vlm.analyze(img_np, smoke_mask, "smoke")
                timing["vlm"] = (time.perf_counter() - _t) * 1000

                result.path_type = "smoke_low"

            # ── 연기 고불확실 경로 → SAM 정밀 마스크 ─────────
            else:
                _t = time.perf_counter()
                refined = self.sam.segment(img_np, cam) if self.config.use_sam \
                    else (cam > self.config.cam_threshold).astype(np.float32)
                result.fire_mask = refined
                result.coverage_ratio = float(refined.mean())
                timing["sam"] = (time.perf_counter() - _t) * 1000

                _t = time.perf_counter()
                result.vlm_result = self.vlm.analyze(img_np, refined, "smoke")
                timing["vlm"] = (time.perf_counter() - _t) * 1000

                result.path_type = "smoke_high"

        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.latency_breakdown = timing
        if verbose:
            self._log(result)
        return result

    # ── 시각화 ────────────────────────────────────────────────

    def visualize(self, result: FireDetectionResult, image_input,
                  save_path: Optional[str] = None):
        """
        원본 | GradCAM | 화재마스크 | Optical Flow(있을 때) 4분할 시각화.
        flow_map이 없으면 기존 3분할로 동작.
        """
        import matplotlib.pyplot as plt

        _, img_np = self._preprocess(image_input)
        path_ko = {"normal": "정상", "flame": "화염 경로 (낙뢰·방화)",
                   "smoke_low": "연기 (저불확실)", "smoke_high": "연기 (고불확실·SAM)"}

        has_flow = result.flow_map is not None
        ncols = 4 if has_flow else 3
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))

        src_label = {"optical_flow": "Optical Flow", "pca": "PCA", "none": ""}
        title = (f"{path_ko.get(result.path_type, result.path_type)}  |  "
                 f"신뢰도 {result.fire_confidence:.2f}  |  "
                 f"불확실성 {result.uncertainty:.2f}  |  "
                 f"방향추정: {src_label.get(result.direction_source, '')}  |  "
                 f"{result.latency_ms:.0f}ms")
        fig.suptitle(title, fontsize=11)

        # ① 원본
        axes[0].imshow(img_np)
        axes[0].set_title("원본 이미지")
        axes[0].axis("off")

        # ② GradCAM
        axes[1].imshow(img_np)
        if result.cam_map is not None:
            im = axes[1].imshow(result.cam_map, alpha=0.55, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("GradCAM")
        axes[1].axis("off")

        # ③ 화재 마스크
        if result.fire_mask is not None:
            overlay = VLMAnalyzer.overlay_mask(img_np, result.fire_mask, result.path_type)
            axes[2].imshow(overlay)
            label = f"화재 마스크  면적 {result.coverage_ratio*100:.1f}%"
            if result.wind_aligned is not None:
                align_str = "풍향일치 ✓" if result.wind_aligned else "풍향불일치 ✗"
                label += f"\n{align_str}  cosine={result.wind_align_score:.2f}"
        else:
            axes[2].imshow(img_np)
            label = "마스크 없음"
        axes[2].set_title(label)
        axes[2].axis("off")

        # ④ Optical Flow 화살표 (있을 때만)
        if has_flow and result.cam_map is not None:
            flow_vis = self.flow_analyzer.flow_overlay(
                img_np, result.flow_map, result.cam_map,
                cam_threshold=self.config.flow_cam_threshold,
            )
            axes[3].imshow(flow_vis)
            dir_str = (f"{result.smoke_direction_deg:.1f}°"
                       if result.smoke_direction_deg is not None else "N/A")
            axes[3].set_title(
                f"Optical Flow  방향:{dir_str}\n"
                f"이동량:{result.flow_magnitude:.2f}px/f"
            )
            axes[3].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.show()
        plt.close()

    # ── 콘솔 로그 ─────────────────────────────────────────────

    def _log(self, r: FireDetectionResult):
        path_ko = {"normal": "정상", "flame": "화염 경로 (낙뢰·방화)",
                   "smoke_low": "연기 (저불확실)", "smoke_high": "연기 (고불확실·SAM)"}
        print(f"\n{'─'*55}")
        print(f"[Pipeline] {path_ko.get(r.path_type, r.path_type)}")
        print(f"  감지:{r.fire_detected}  신뢰도:{r.fire_confidence:.3f}  "
              f"불확실성:{r.uncertainty:.3f}")
        if r.smoke_direction_deg is not None:
            src = {"optical_flow": "[OF]", "pca": "[PCA]", "none": ""}.get(
                r.direction_source, "")
            print(f"  연기 방향:{r.smoke_direction_deg:.1f}° {src}", end="")
            if r.flow_magnitude > 0:
                print(f"  이동량:{r.flow_magnitude:.2f}px/f", end="")
            if r.wind_aligned is not None:
                flag = "✓일치" if r.wind_aligned else "✗불일치"
                print(f"  풍향 {flag} (cosine={r.wind_align_score:.2f})", end="")
            print()
        if r.fire_mask is not None:
            print(f"  화재 면적:{r.coverage_ratio*100:.1f}%")
        breakdown = " | ".join(f"{k}={v:.1f}ms" for k, v in r.latency_breakdown.items())
        print(f"  총:{r.latency_ms:.1f}ms  [{breakdown}]")
        if r.vlm_result:
            print(f"  VLM:{r.vlm_result}")
        print(f"{'─'*55}")


# ──────────────────────────────────────────────────────────────────────────────
# Ablation 평가 도구 (층화 테스트셋)
# ──────────────────────────────────────────────────────────────────────────────

def ablation_evaluate(
    pipeline: FireDetectionPipeline,
    test_cases: list,
    wind_direction: Optional[float] = None,
) -> Dict[str, Any]:
    """
    층화 테스트셋 ablation 평가.

    test_cases 형식:
        [{"image": "path_or_array", "label": 1, "case_type": "flame"}, ...]
        case_type: 'flame' | 'smoke' | 'mixed' | 'normal'

    Returns:
        metrics dict (case_type별 정확도·지연시간 분해)
    """
    from collections import defaultdict

    strata: Dict[str, list] = defaultdict(list)
    for case in test_cases:
        img = case["image"]
        true_label = case["label"]             # 0=normal, 1=fire
        case_type = case.get("case_type", "unknown")

        result = pipeline.run(img, wind_direction=wind_direction)
        pred = 1 if result.fire_detected else 0
        correct = int(pred == true_label)

        strata[case_type].append({
            "correct": correct,
            "path_type": result.path_type,
            "latency_ms": result.latency_ms,
            "confidence": result.fire_confidence,
            "uncertainty": result.uncertainty,
            "wind_aligned": result.wind_aligned,
        })

    report = {}
    for case_type, records in strata.items():
        n = len(records)
        acc = sum(r["correct"] for r in records) / n
        avg_lat = np.mean([r["latency_ms"] for r in records])
        path_dist: Dict[str, int] = defaultdict(int)
        for r in records:
            path_dist[r["path_type"]] += 1
        report[case_type] = {
            "n": n,
            "accuracy": round(acc, 4),
            "avg_latency_ms": round(float(avg_lat), 1),
            "path_distribution": dict(path_dist),
        }

    # 전체 요약
    all_records = [r for recs in strata.values() for r in recs]
    report["_total"] = {
        "n": len(all_records),
        "accuracy": round(np.mean([r["correct"] for r in all_records]), 4),
        "avg_latency_ms": round(float(np.mean([r["latency_ms"] for r in all_records])), 1),
    }

    print("\n[Ablation Report]")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report
