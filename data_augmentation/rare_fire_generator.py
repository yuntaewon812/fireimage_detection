"""
희귀 화재 케이스 합성기
========================
AI Hub 데이터에 없는 2가지 즉각 발화 케이스를 Copy-Paste 합성으로 생성.

  낙뢰 화재 (lightning_fire):
    - 연기 없이 국소 화염만 존재 (연막탄 모의발화와 다름)
    - 섬광 프레임 + 화재 발생 프레임 2장 세트로 생성
    - 발화 지점: 산 중턱·능선 (하늘 경계 부근)

  방화 (arson):
    - 연기 없이 1~3개 지점에서 동시 발화
    - 인접한 두 발화 지점: 방화의 물리적 특성
    - GT bounding box 동시 생성

생성 방식:
  1. 기존 fire 이미지에서 HSV 마스킹으로 화염 패치 자동 추출
  2. 배경 이미지(normal)에 Poisson 블렌딩으로 자연스럽게 합성
  3. 패치가 충분하지 않으면 합성 화염 blob 생성(fallback)
"""

import cv2
import numpy as np
import os
import random
import glob
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 화염 패치 추출
# ─────────────────────────────────────────────────────────────────────────────

class FirePatchExtractor:
    """
    기존 화재(abnormal) 이미지에서 HSV 마스킹으로 화염 패치를 추출.
    추출된 패치는 RareFireGenerator의 합성 재료로 사용.
    """

    # 화염 HSV 범위 (BGR→HSV 변환 기준)
    _FLAME_RANGES = [
        ((0,   130,  80), (12,  255, 255)),   # 빨강(저)
        ((158, 130,  80), (179, 255, 255)),   # 빨강(고)
        ((12,  130,  80), (25,  255, 255)),   # 주황
        ((25,  130, 130), (36,  255, 255)),   # 노랑
    ]
    _KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def extract_from_image(
        self,
        image_bgr: np.ndarray,
        min_area: int = 800,
        max_patches: int = 5,
    ) -> List[np.ndarray]:
        """
        단일 이미지에서 화염 패치(BGRA, alpha=불투명 마스크) 추출.
        Returns: BGRA 패치 리스트
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self._FLAME_RANGES:
            mask |= cv2.inRange(hsv, lo, hi)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._KERNEL)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        patches = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:max_patches]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue

            # BGRA 패치 (alpha = 해당 영역 마스크)
            patch_bgr = image_bgr[y:y+h, x:x+w].copy()
            patch_mask = mask[y:y+h, x:x+w]
            alpha = cv2.dilate(patch_mask, self._KERNEL)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)  # 부드러운 경계
            bgra = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            patches.append(bgra)

        return patches

    def extract_from_dir(
        self,
        fire_dir: str,
        max_images: int = 100,
        min_patches_per_image: int = 1,
    ) -> List[np.ndarray]:
        """
        화재 이미지 디렉터리에서 패치를 일괄 수집.
        Returns: BGRA 패치 리스트 (전체)
        """
        all_patches = []
        exts = ["*.jpg", "*.jpeg", "*.png"]
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(fire_dir, "**", ext), recursive=True)

        random.shuffle(paths)
        for path in paths[:max_images]:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            patches = self.extract_from_image(img)
            if len(patches) >= min_patches_per_image:
                all_patches.extend(patches)

        print(f"[FirePatchExtractor] {len(all_patches)}개 패치 수집 "
              f"(이미지 {min(len(paths), max_images)}장 처리)")
        return all_patches

    @staticmethod
    def make_synthetic_patch(size: int = 60) -> np.ndarray:
        """
        실제 패치 없을 때 사용하는 합성 화염 blob (fallback).
        Returns: BGRA numpy array
        """
        h, w = int(size * 1.3), size
        patch = np.zeros((h, w, 4), dtype=np.uint8)
        cx, cy = w // 2, h * 2 // 3

        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                dist = np.sqrt(dx**2 + (dy * 0.7)**2)  # 세로로 약간 타원
                if dist >= size // 2:
                    continue
                t = 1.0 - dist / (size // 2)
                # 색상: 중심=노랑, 외곽=빨강(BGR)
                b = int(10 * t)
                g = int(200 * (t ** 0.5))
                r = int(255 * t)
                a = int(255 * (t ** 0.4))
                patch[y, x] = [b, g, r, a]

        # 약간의 노이즈로 현실감 추가
        noise = np.random.randint(-20, 20, patch[:, :, :3].shape, dtype=np.int16)
        patch[:, :, :3] = np.clip(patch[:, :, :3].astype(np.int16) + noise,
                                  0, 255).astype(np.uint8)
        return patch


# ─────────────────────────────────────────────────────────────────────────────
# 합성 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _paste_bgra(background: np.ndarray, patch_bgra: np.ndarray,
                cx: int, cy: int, scale: float = 1.0) -> np.ndarray:
    """
    BGRA 패치를 배경에 알파 블렌딩으로 합성.
    (cx, cy): 패치 중심 좌표
    """
    bg = background.copy()
    H, W = bg.shape[:2]

    # 스케일 조정
    ph, pw = patch_bgra.shape[:2]
    new_h, new_w = int(ph * scale), int(pw * scale)
    if new_h < 4 or new_w < 4:
        return bg
    patch_bgra = cv2.resize(patch_bgra, (new_w, new_h))

    # 배치 좌표
    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x2, y2 = x1 + new_w, y1 + new_h

    # 이미지 경계 클리핑
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(W, x2), min(H, y2)
    px1, py1 = bx1 - x1, by1 - y1
    px2, py2 = px1 + (bx2 - bx1), py1 + (by2 - by1)

    if bx2 <= bx1 or by2 <= by1:
        return bg

    alpha = patch_bgra[py1:py2, px1:px2, 3:4].astype(np.float32) / 255.0
    patch_rgb = patch_bgra[py1:py2, px1:px2, :3].astype(np.float32)
    bg_roi = bg[by1:by2, bx1:bx2].astype(np.float32)

    blended = patch_rgb * alpha + bg_roi * (1.0 - alpha)
    bg[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def _try_poisson(background: np.ndarray, patch_bgra: np.ndarray,
                 cx: int, cy: int, scale: float = 1.0) -> np.ndarray:
    """
    Poisson 블렌딩 시도, 실패 시 알파 블렌딩으로 폴백.
    Poisson: 색상이 자연스럽게 주변에 녹아듦.
    """
    bg = background.copy()
    H, W = bg.shape[:2]

    ph, pw = patch_bgra.shape[:2]
    new_h, new_w = max(4, int(ph * scale)), max(4, int(pw * scale))
    patch_bgra = cv2.resize(patch_bgra, (new_w, new_h))

    cx_clip = max(new_w // 2 + 1, min(W - new_w // 2 - 1, cx))
    cy_clip = max(new_h // 2 + 1, min(H - new_h // 2 - 1, cy))

    patch_bgr = patch_bgra[:, :, :3]
    alpha_mask = patch_bgra[:, :, 3]
    bin_mask = (alpha_mask > 127).astype(np.uint8) * 255

    try:
        result = cv2.seamlessClone(
            patch_bgr, bg, bin_mask,
            (cx_clip, cy_clip), cv2.MIXED_CLONE
        )
        return result
    except cv2.error:
        return _paste_bgra(background, patch_bgra, cx, cy, scale=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 희귀 화재 케이스 생성기
# ─────────────────────────────────────────────────────────────────────────────

class RareFireGenerator:
    """
    낙뢰 화재·방화 케이스 합성 생성기.

    Usage:
        extractor = FirePatchExtractor()
        patches = extractor.extract_from_dir("data/fireimage/abnormal", max_images=50)

        gen = RareFireGenerator(patches)
        img, bboxes = gen.lightning_fire(background_bgr)
        img, bboxes = gen.arson(background_bgr)
    """

    def __init__(self, fire_patches: Optional[List[np.ndarray]] = None):
        """
        Args:
            fire_patches: FirePatchExtractor로 추출한 BGRA 패치 리스트.
                          None 또는 비어있으면 합성 패치(fallback) 사용.
        """
        self._patches = fire_patches or []
        self._extractor = FirePatchExtractor()

    def _get_patch(self, size_hint: int = 60) -> np.ndarray:
        """패치 풀에서 하나 선택 (없으면 합성)."""
        if self._patches:
            patch = random.choice(self._patches)
            ph, pw = patch.shape[:2]
            if max(ph, pw) > 200:  # 너무 큰 패치는 축소
                scale = 200 / max(ph, pw)
                patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
            return patch
        return FirePatchExtractor.make_synthetic_patch(size_hint)

    def lightning_fire(
        self,
        background_bgr: np.ndarray,
        use_poisson: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        낙뢰 화재 합성.
        연기 없이 산 중턱~능선에 국소 화염만 발생.

        Returns:
            flash_frame: 섬광 프레임 (낙뢰 순간)
            fire_frame:  화재 발생 프레임
            bboxes:      [{"x","y","w","h","label":"lightning_fire"}]
        """
        H, W = background_bgr.shape[:2]

        # ── 섬광 프레임 ──────────────────────────────────────
        flash = background_bgr.copy().astype(np.float32)
        flash[:, :, 0] = np.clip(flash[:, :, 0] * 2.2 + 40, 0, 255)  # B↑(파란빛)
        flash[:, :, 1] = np.clip(flash[:, :, 1] * 2.5 + 30, 0, 255)
        flash[:, :, 2] = np.clip(flash[:, :, 2] * 2.0, 0, 255)
        flash_frame = np.clip(flash, 0, 255).astype(np.uint8)

        # ── 화염 발생 프레임 ─────────────────────────────────
        fire_frame = background_bgr.copy()

        # 낙뢰 발화 지점: 산 중턱 (화면 1/3~2/3)
        n_points = 1  # 낙뢰는 단일 지점
        strike_x = random.randint(W // 4, 3 * W // 4)
        strike_y = random.randint(H // 3, 2 * H // 3)

        patch = self._get_patch(size_hint=random.randint(40, 90))
        ph, pw = patch.shape[:2]
        scale = random.uniform(0.6, 1.3)

        bboxes = []
        if use_poisson:
            fire_frame = _try_poisson(fire_frame, patch, strike_x, strike_y, scale)
        else:
            fire_frame = _paste_bgra(fire_frame, patch, strike_x, strike_y, scale)

        actual_w, actual_h = int(pw * scale), int(ph * scale)
        bboxes.append({
            "x": max(0, strike_x - actual_w // 2),
            "y": max(0, strike_y - actual_h // 2),
            "w": min(actual_w, W),
            "h": min(actual_h, H),
            "label": "lightning_fire",
        })

        # 약한 연기 없음 확인: 낙뢰는 즉각 발화, 연기 거의 없음
        # (선택) 아주 작은 흰 bloom만 추가
        boom_sigma = int(actual_w * 0.8)
        if boom_sigma > 5:
            Y, X = np.ogrid[:H, :W]
            bloom = np.exp(-((X - strike_x)**2 + (Y - strike_y)**2)
                           / (2 * boom_sigma**2))
            for c in range(3):
                fire_frame[:, :, c] = np.clip(
                    fire_frame[:, :, c].astype(np.float32) + bloom * 30, 0, 255
                ).astype(np.uint8)

        return flash_frame, fire_frame, bboxes

    def arson(
        self,
        background_bgr: np.ndarray,
        use_poisson: bool = True,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        방화 합성.
        연기 전구체 없이 1~3개 지점 동시 발화.
        인접한 두 발화 지점이 있으면 방화 의심 패턴.

        Returns:
            fire_frame: 합성 이미지
            bboxes:     [{"x","y","w","h","label":"arson"}]
        """
        H, W = background_bgr.shape[:2]
        fire_frame = background_bgr.copy()

        n_points = random.randint(1, 3)
        bboxes = []

        if n_points >= 2:
            # 방화: 가까운 두 지점 (100~250px 간격)
            base_x = random.randint(W // 5, 4 * W // 5)
            base_y = random.randint(H // 2, H - H // 5)
            ignition_pts = [(base_x, base_y)]
            for _ in range(n_points - 1):
                offset_x = random.randint(60, 200) * random.choice([-1, 1])
                offset_y = random.randint(-60, 60)
                ignition_pts.append((
                    max(50, min(W - 50, base_x + offset_x)),
                    max(50, min(H - 50, base_y + offset_y)),
                ))
        else:
            ignition_pts = [(
                random.randint(W // 4, 3 * W // 4),
                random.randint(H // 2, H - H // 5),
            )]

        for (px, py) in ignition_pts:
            patch = self._get_patch(size_hint=random.randint(30, 70))
            ph, pw_p = patch.shape[:2]
            scale = random.uniform(0.5, 1.2)

            if use_poisson:
                fire_frame = _try_poisson(fire_frame, patch, px, py, scale)
            else:
                fire_frame = _paste_bgra(fire_frame, patch, px, py, scale)

            actual_w, actual_h = int(pw_p * scale), int(ph * scale)
            bboxes.append({
                "x": max(0, px - actual_w // 2),
                "y": max(0, py - actual_h // 2),
                "w": min(actual_w, W),
                "h": min(actual_h, H),
                "label": "arson",
            })

        return fire_frame, bboxes

    def visualize_cases(self, background_bgr: np.ndarray,
                        save_path: Optional[str] = None):
        """낙뢰·방화 케이스를 나란히 시각화."""
        import matplotlib.pyplot as plt

        flash, lightning_img, lbboxes = self.lightning_fire(background_bgr)
        arson_img, abboxes = self.arson(background_bgr)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def draw_bboxes(ax, img_bgr, bboxes, title):
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            vis = rgb.copy()
            for b in bboxes:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 50, 50), 2)
                cv2.putText(vis, b["label"], (x, max(0, y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1)
            ax.imshow(vis)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        axes[0, 0].imshow(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("배경 (Background)", fontsize=10)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(flash, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("낙뢰 섬광 프레임", fontsize=10)
        axes[0, 1].axis("off")

        draw_bboxes(axes[1, 0], lightning_img, lbboxes,
                    f"낙뢰 화재 (GT bbox {len(lbboxes)}개)")
        draw_bboxes(axes[1, 1], arson_img, abboxes,
                    f"방화 ({len(abboxes)}개 발화 지점)")

        plt.suptitle("희귀 화재 케이스 합성 (Lightning Fire & Arson)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[시각화 저장] {save_path}")
        else:
            plt.show()
        plt.close()
