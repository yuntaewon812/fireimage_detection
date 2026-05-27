"""
오탐(False Positive) 케이스 증강기
=====================================
화재로 오인하기 쉬운 11가지 시각적 패턴을 생성.

카테고리별 오탐 케이스:
  [색상·형태 유사]
  1.  autumn_foliage       가을 단풍   HSV 편향으로 나뭇잎을 주황·빨강으로
  2.  sunset_backlight     일출·일몰   하늘 전체를 주황빛으로 물들임
  3.  greenhouse_reflect   비닐하우스  흰 반사광이 연기처럼 보임
  4.  yellow_dust          황사·미세먼지 전체 시야가 누런 뿌연 색조
  5.  steam_extinguish     진화 후 수증기  흰 가우시안 블롭 합성

  [기상 조건]
  6.  valley_fog           물안개      하단 계곡에 집중된 흰 안개
  7.  ridge_fog            역무·능선   능선을 따라 퍼지는 안개
  8.  blizzard             눈보라      흰색 분산 패턴 + 밝기 과다

  [광원 조건]
  9.  nighttime_light      야간 불빛   어두운 배경에 점광원
  10. lightning_flash      번개 섬광   순간 전체 밝기 급증
  11. helicopter_spot      헬기 조명   원형 스포트라이트 빔

입출력 규약:
  - 입력/출력: BGR uint8 numpy array (cv2 기본 형식)
  - 파일 저장 시 BGR → 그대로 imwrite
"""

import cv2
import numpy as np
import random
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 증강 함수 모음
# ─────────────────────────────────────────────────────────────────────────────

def autumn_foliage(image: np.ndarray) -> np.ndarray:
    """
    가을 단풍 — HSV에서 녹색 계열 Hue를 주황·빨강으로 이동.
    산림 지대에서 붉게 변한 나뭇잎이 화염과 혼동됨.
    """
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 녹색(55~85) → 주황·빨강(5~25)으로 Hue 이동
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_mask = (h > 55) & (h < 90)

    shift = random.uniform(35, 55)          # Hue 이동량
    sat_boost = random.uniform(1.25, 1.6)   # 채도 강화 (선명한 단풍)
    h[green_mask] = (h[green_mask] - shift) % 180
    s[green_mask] = np.clip(s[green_mask] * sat_boost, 0, 255)

    # 일부 노랑도 추가 (낙엽 다양성)
    yellow_mask = (h > 25) & (h < 40)
    h[yellow_mask] = h[yellow_mask] - random.uniform(5, 15)
    s[yellow_mask] = np.clip(s[yellow_mask] * 1.2, 0, 255)

    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 전체 따뜻한 색조 살짝 추가
    warm = np.zeros_like(result)
    warm[:, :] = (0, 20, 40)   # BGR: 약간 적색 편향
    return cv2.addWeighted(result, 0.9, warm, 0.1, 0)


def sunset_backlight(image: np.ndarray) -> np.ndarray:
    """
    일출·일몰 역광 — 화면 상단 하늘 영역이 주황·빨강으로 물들어
    산림 전체가 화재처럼 보임.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    sky_ratio = random.uniform(0.30, 0.55)
    sky_h = int(h * sky_ratio)

    # 주황 오버레이 색상 (BGR)
    r_shift = random.uniform(0.3, 0.65)
    orange_bgr = np.array([random.randint(10, 40),
                            random.randint(60, 140),
                            random.randint(200, 255)], dtype=np.float32)

    # 하늘 영역에 그라데이션 오버레이 (최상단이 가장 진함)
    for y in range(sky_h):
        t = 1.0 - y / sky_h              # 위쪽일수록 t → 1
        alpha = r_shift * t
        img[y] = img[y] * (1 - alpha) + orange_bgr * alpha

    # 지평선 부근 약한 빛 번짐 (glow)
    glow_h = max(1, int(sky_h * 0.15))
    for y in range(sky_h, min(h, sky_h + glow_h)):
        t = 1.0 - (y - sky_h) / glow_h
        alpha = 0.15 * t
        img[y] = img[y] * (1 - alpha) + orange_bgr * alpha

    return np.clip(img, 0, 255).astype(np.uint8)


def greenhouse_reflection(image: np.ndarray) -> np.ndarray:
    """
    비닐하우스 반사광 — 흰색 직사각형 반사 패턴이 연기·증기처럼 보임.
    주로 평지나 경사면에 위치한 하우스에서 발생.
    """
    img = image.copy()
    h, w = img.shape[:2]

    n_houses = random.randint(2, 5)
    for _ in range(n_houses):
        # 직사각형 반사 패치
        rx = random.randint(0, w - w // 4)
        ry = random.randint(h // 2, h - h // 5)
        rw = random.randint(w // 8, w // 4)
        rh = random.randint(h // 12, h // 6)

        # 밝은 흰색 패치 (반사광)
        patch = np.ones((rh, rw, 3), dtype=np.float32) * 240
        # 가우시안 블러로 흐릿하게 (현실감)
        blur_k = random.choice([5, 9, 13])
        patch = cv2.GaussianBlur(patch, (blur_k, blur_k), 0)

        alpha = random.uniform(0.3, 0.6)
        roi = img[ry:ry+rh, rx:rx+rw].astype(np.float32)
        img[ry:ry+rh, rx:rx+rw] = np.clip(
            roi * (1 - alpha) + patch * alpha, 0, 255
        ).astype(np.uint8)

    return img


def yellow_dust(image: np.ndarray) -> np.ndarray:
    """
    황사·미세먼지 — 전체 화면이 누런 뿌연 색조.
    원거리 산림이 뿌옇게 보여 연기로 혼동.
    """
    img = image.copy().astype(np.float32)

    # 황색 레이어 (BGR)
    dust_b = random.randint(10, 40)
    dust_g = random.randint(130, 180)
    dust_r = random.randint(180, 220)
    dust = np.full_like(img, (dust_b, dust_g, dust_r), dtype=np.float32)

    alpha = random.uniform(0.22, 0.48)
    blended = img * (1 - alpha) + dust * alpha

    # 대비 감소 (시야 탁함)
    contrast = random.uniform(0.75, 0.90)
    brightness = random.uniform(10, 25)
    result = np.clip(blended * contrast + brightness, 0, 255).astype(np.uint8)

    # 가우시안 블러로 경계 흐릿하게
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(result, (k, k), 0)


def steam_extinguish(image: np.ndarray) -> np.ndarray:
    """
    진화 후 수증기 — 흰색 반투명 블롭이 연기와 구분 어려움.
    수증기는 연기보다 밝고 빠르게 확산됨.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    n_blobs = random.randint(1, 3)
    for _ in range(n_blobs):
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        sigma_x = random.randint(30, 100)
        sigma_y = random.randint(40, 130)  # 수직으로 약간 더 길게
        intensity = random.uniform(120, 200)

        Y, X = np.ogrid[:h, :w]
        blob = np.exp(-((X - cx) ** 2 / (2 * sigma_x ** 2) +
                        (Y - cy) ** 2 / (2 * sigma_y ** 2)))
        alpha = random.uniform(0.55, 0.85)

        for c in range(3):
            img[:, :, c] += blob * intensity * alpha

    return np.clip(img, 0, 255).astype(np.uint8)


def valley_fog(image: np.ndarray) -> np.ndarray:
    """
    물안개 — 하단 계곡 부분에 집중된 흰 안개.
    새벽 계곡 물안개가 연기 낮게 깔린 패턴과 혼동됨.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    fog_start = int(h * random.uniform(0.40, 0.70))
    fog_density = random.uniform(0.55, 0.90)

    for y in range(fog_start, h):
        progress = (y - fog_start) / max(1, h - fog_start)
        alpha = fog_density * (progress ** 0.7)  # 비선형: 아래로 갈수록 급격히 진해짐
        img[y] = img[y] * (1 - alpha) + 255 * alpha

    # 수평 경계를 흐릿하게 (자연스러운 안개 경계)
    result = np.clip(img, 0, 255).astype(np.uint8)
    blur_region = result[max(0, fog_start - 15):fog_start + 15]
    if blur_region.shape[0] > 0:
        result[max(0, fog_start - 15):fog_start + 15] = cv2.GaussianBlur(
            blur_region, (1, 15), 0)
    return result


def ridge_fog(image: np.ndarray) -> np.ndarray:
    """
    역무·능선 안개 — 능선을 따라 수평으로 퍼지는 안개.
    산 중턱에 걸린 안개띠가 연기처럼 보임.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    n_bands = random.randint(1, 3)
    for _ in range(n_bands):
        center_y = random.randint(int(h * 0.2), int(h * 0.7))
        band_width = random.randint(h // 10, h // 4)
        density = random.uniform(0.40, 0.80)

        for y in range(max(0, center_y - band_width),
                       min(h, center_y + band_width)):
            dist = abs(y - center_y) / band_width
            alpha = density * (1 - dist ** 1.5)   # 중심이 가장 진함
            if alpha > 0:
                img[y] = img[y] * (1 - alpha) + 250 * alpha

    return np.clip(img, 0, 255).astype(np.uint8)


def blizzard(image: np.ndarray) -> np.ndarray:
    """
    눈보라 — 흰색 분산 패턴 + 밝기 과다.
    눈 날리는 장면이 흰 연기가 확산하는 것처럼 보임.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    # 배경을 밝고 저대비로
    img = img * random.uniform(0.7, 0.9) + random.uniform(30, 70)

    # 눈송이 파티클 (작은 흰 점)
    n_particles = random.randint(200, 600)
    for _ in range(n_particles):
        px = random.randint(0, w - 1)
        py = random.randint(0, h - 1)
        radius = random.randint(1, 4)
        brightness = random.randint(200, 255)
        cv2.circle(img.astype(np.uint8), (px, py), radius,
                   (brightness, brightness, brightness), -1)

    # 전체 블러 (눈보라 효과)
    result = np.clip(img, 0, 255).astype(np.uint8)
    k = random.choice([3, 5])
    return cv2.GaussianBlur(result, (k, k), 0)


def nighttime_light(image: np.ndarray) -> np.ndarray:
    """
    야간 가로등·불빛 — 어두운 배경에 주황빛 점광원.
    소규모 화염처럼 보이는 인공 광원.
    """
    img = image.copy().astype(np.float32)

    # 전체를 어둡게 (야간)
    darkness = random.uniform(0.15, 0.35)
    img *= darkness

    n_lights = random.randint(3, 8)
    h, w = img.shape[:2]

    for _ in range(n_lights):
        lx = random.randint(30, w - 30)
        ly = random.randint(30, h - 30)
        radius = random.randint(6, 22)

        # 광원 색상: 주황~노랑 (BGR)
        b = random.randint(20, 60)
        g = random.randint(100, 180)
        r = random.randint(200, 255)
        color = np.array([b, g, r], dtype=np.float32)

        # 가우시안 glow 합성
        Y, X = np.ogrid[:h, :w]
        sigma = radius * 2.5
        glow = np.exp(-((X - lx) ** 2 + (Y - ly) ** 2) / (2 * sigma ** 2))
        for c in range(3):
            img[:, :, c] += glow * color[c] * random.uniform(0.6, 1.0)

        # 광원 중심 밝은 점
        cv2.circle(img.astype(np.uint8), (lx, ly), radius,
                   (int(b * 2), int(g * 1.2), int(r)), -1)

    return np.clip(img, 0, 255).astype(np.uint8)


def lightning_flash(image: np.ndarray) -> np.ndarray:
    """
    번개 섬광 — 순간 전체 밝기가 급증하여 화재 감지 오류 유발.
    채널별 불균등 밝기 상승으로 파란빛 번개 느낌 추가.
    """
    img = image.copy().astype(np.float32)

    # 전체 밝기 과다 노출
    b_mult = random.uniform(2.0, 3.5)
    g_mult = random.uniform(1.8, 3.0)
    r_mult = random.uniform(1.5, 2.5)

    img[:, :, 0] = np.clip(img[:, :, 0] * b_mult + 50, 0, 255)  # B 강조
    img[:, :, 1] = np.clip(img[:, :, 1] * g_mult + 30, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * r_mult, 0, 255)

    # 번개 방전 경로 (밝은 선)
    h, w = img.shape[:2]
    start = (random.randint(0, w), 0)
    segments = random.randint(3, 6)
    pts = [start]
    y = 0
    for _ in range(segments):
        y += random.randint(h // (segments + 1), h // segments)
        x = pts[-1][0] + random.randint(-60, 60)
        pts.append((max(0, min(w - 1, x)), min(h - 1, y)))

    for i in range(len(pts) - 1):
        cv2.line(img.astype(np.uint8), pts[i], pts[i + 1],
                 (255, 255, 255), random.randint(1, 3))

    return np.clip(img, 0, 255).astype(np.uint8)


def helicopter_spotlight(image: np.ndarray) -> np.ndarray:
    """
    헬기 조명 — 야간 산불 진화 헬기의 스포트라이트 빔.
    이미 감지된 구역에 다시 조명이 비치면 재감지 오류 유발.
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    # 배경 어둡게
    img *= random.uniform(0.2, 0.40)

    # 타원형 스포트라이트
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 3, 2 * h // 3)
    rx = random.randint(w // 6, w // 3)
    ry = random.randint(h // 6, h // 3)

    Y, X = np.ogrid[:h, :w]
    ellipse_dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    spotlight = np.exp(-ellipse_dist * 2.0)

    # 조명 색상: 약간 차가운 흰색 (헬기 서치라이트)
    light_color = np.array([240, 235, 220], dtype=np.float32)  # BGR
    for c in range(3):
        img[:, :, c] += spotlight * light_color[c] * random.uniform(0.7, 1.0)

    return np.clip(img, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────────────────────

# 증강 타입 → (함수, 설명, 저장 접두어)
AUG_REGISTRY = {
    "autumn_foliage":       (autumn_foliage,       "가을 단풍",         "aug_autumn"),
    "sunset_backlight":     (sunset_backlight,     "일출·일몰 역광",    "aug_sunset"),
    "greenhouse_reflect":   (greenhouse_reflection,"비닐하우스 반사광", "aug_greenhouse"),
    "yellow_dust":          (yellow_dust,          "황사·미세먼지",     "aug_dust"),
    "steam_extinguish":     (steam_extinguish,     "진화 후 수증기",    "aug_steam"),
    "valley_fog":           (valley_fog,           "물안개",            "aug_valley_fog"),
    "ridge_fog":            (ridge_fog,            "역무·능선 안개",    "aug_ridge_fog"),
    "blizzard":             (blizzard,             "눈보라",            "aug_blizzard"),
    "nighttime_light":      (nighttime_light,      "야간 불빛",         "aug_night"),
    "lightning_flash":      (lightning_flash,      "번개 섬광",         "aug_lightning"),
    "helicopter_spot":      (helicopter_spotlight, "헬기 조명",         "aug_heli"),
}


class FalsePositiveAugmentor:
    """
    기존 normal 이미지에 11가지 오탐 패턴을 적용하여
    모델이 어려워하는 하드 네거티브(hard negative) 샘플을 생성.

    Usage:
        augmentor = FalsePositiveAugmentor()
        result_bgr = augmentor.augment(image_bgr)               # 랜덤 타입
        result_bgr = augmentor.augment(image_bgr, "autumn_foliage")  # 지정 타입
    """

    def __init__(self, aug_types: Optional[list] = None):
        """
        Args:
            aug_types: 사용할 증강 타입 리스트. None이면 전체 11가지 사용.
        """
        self.aug_types = aug_types or list(AUG_REGISTRY.keys())

    def augment(self, image_bgr: np.ndarray,
                aug_type: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        단일 이미지에 증강 적용.
        Returns:
            (augmented_bgr, aug_type_used)
        """
        if aug_type is None:
            aug_type = random.choice(self.aug_types)
        if aug_type not in AUG_REGISTRY:
            raise ValueError(f"Unknown aug_type: {aug_type}. "
                             f"Valid: {list(AUG_REGISTRY.keys())}")
        fn = AUG_REGISTRY[aug_type][0]
        return fn(image_bgr), aug_type

    def augment_all_types(self, image_bgr: np.ndarray) -> dict:
        """11가지 모든 타입으로 증강한 결과를 반환 (시각화용)."""
        return {
            aug_type: AUG_REGISTRY[aug_type][0](image_bgr)
            for aug_type in self.aug_types
        }

    def save_augmented(self, image_bgr: np.ndarray, save_dir: str,
                       aug_type: Optional[str] = None,
                       index: int = 0) -> Tuple[str, str]:
        """
        증강 이미지를 파일로 저장.
        Returns:
            (saved_path, aug_type_used)
        """
        import os
        augmented, used_type = self.augment(image_bgr, aug_type)
        prefix = AUG_REGISTRY[used_type][2]
        filename = f"{prefix}_{index:05d}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, augmented, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return save_path, used_type

    def visualize_all(self, image_bgr: np.ndarray,
                      save_path: Optional[str] = None):
        """11가지 증강 결과를 그리드로 시각화."""
        import matplotlib.pyplot as plt

        results = self.augment_all_types(image_bgr)
        n = len(results) + 1  # +1 for original
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten()

        # 원본
        axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("원본 (Original)", fontsize=9)
        axes[0].axis("off")

        for i, (aug_type, aug_img) in enumerate(results.items(), start=1):
            desc = AUG_REGISTRY[aug_type][1]
            axes[i].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"{i}. {desc}", fontsize=9)
            axes[i].axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.suptitle("오탐 증강 케이스 11종 (False Positive Augmentation)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[시각화 저장] {save_path}")
        else:
            plt.show()
        plt.close()
