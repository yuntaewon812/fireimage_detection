"""
데이터셋 구축 파이프라인 (CLI)
================================
기존 AI Hub 데이터에 오탐·희귀화재 증강을 적용하여
완성된 학습 데이터셋을 생성.

최종 데이터 구성 권장안:
  소스                         목표 장수   용도
  ─────────────────────────────────────────────
  AI Hub 원본 (화염·흑연기)    ~50,000    베이스 학습
  AI Hub 네거티브 (구름·안개)  ~30,000    오탐 학습
  오탐 증강 (11가지 타입)      ~40,000    하드 네거티브
  낙뢰·방화 합성               ~3,000     희귀 케이스 (전체 3~5%)
  YouTube 프레임               ~2,000     실제 한국 화재

Usage:
  # 기본 실행 (권장 비율 자동 적용)
  python data_augmentation/dataset_builder.py --class_name fireimage

  # 커스텀 장수 지정
  python data_augmentation/dataset_builder.py \\
      --class_name fireimage \\
      --n_false_positive 5000 \\
      --n_rare_fire 300 \\
      --visualize

  # 시각화 샘플만 생성 (실제 저장 안 함)
  python data_augmentation/dataset_builder.py --dry_run
"""

import os
import glob
import random
import argparse
import json
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from data_augmentation.false_positive_augmentor import FalsePositiveAugmentor, AUG_REGISTRY
from data_augmentation.rare_fire_generator import FirePatchExtractor, RareFireGenerator


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> Optional[np.ndarray]:
    """한글 경로 포함 이미지 로드 (cv2 기본 함수는 한글 경로 미지원)."""
    arr = np.fromfile(path, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _list_images(directory: str) -> list:
    """디렉터리에서 이미지 경로 목록 수집."""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(directory, "**", ext), recursive=True)
    return sorted(paths)


def _next_index(directory: str, prefix: str) -> int:
    """기존 파일 인덱스 충돌 방지를 위해 마지막 인덱스 + 1 반환."""
    existing = glob.glob(os.path.join(directory, f"{prefix}_*.jpg"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def _save_with_manifest(image_bgr: np.ndarray, save_dir: str,
                        prefix: str, index: int,
                        manifest: dict, meta: dict):
    """이미지 저장 + 매니페스트 기록."""
    filename = f"{prefix}_{index:05d}.jpg"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    manifest[filename] = meta
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 오탐 증강 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def build_false_positive_augmentations(
    normal_dir: str,
    output_dir: str,
    n_total: int = 5000,
    aug_types: Optional[list] = None,
    manifest: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    normal 이미지에 11가지 오탐 증강을 적용하여 하드 네거티브 생성.

    각 증강 타입별 균등 분배 (n_total / 11가지 ≈ 각 450장).
    전체 3~5%에 해당하는 희귀 오탐(야간·번개)은 약 절반 분량 생성.

    Args:
        normal_dir:  원본 normal 이미지 경로
        output_dir:  저장 경로 (normal/ 하위에 저장 권장)
        n_total:     생성할 총 이미지 수
        aug_types:   사용할 증강 타입 리스트 (None=전체 11가지)
        manifest:    메타데이터 누적 dict
        verbose:     진행상황 출력
    """
    os.makedirs(output_dir, exist_ok=True)
    if manifest is None:
        manifest = {}

    source_paths = _list_images(normal_dir)
    if not source_paths:
        print(f"[WARNING] {normal_dir} 에 이미지가 없습니다.")
        return manifest

    augmentor = FalsePositiveAugmentor(aug_types=aug_types)
    all_types = augmentor.aug_types

    # 타입별 생성 수 계산 (균등 분배)
    n_per_type = max(1, n_total // len(all_types))
    # 야간·번개는 희귀 → 절반만
    rare_types = {"nighttime_light", "lightning_flash", "helicopter_spot"}
    counts = {t: (n_per_type // 2 if t in rare_types else n_per_type)
              for t in all_types}

    stats = defaultdict(int)
    total_saved = 0

    for aug_type, target_count in counts.items():
        prefix = AUG_REGISTRY[aug_type][2]
        desc = AUG_REGISTRY[aug_type][1]
        idx = _next_index(output_dir, prefix)

        for i in range(target_count):
            src_path = random.choice(source_paths)
            img = _load_image(src_path)
            if img is None:
                continue

            fn = AUG_REGISTRY[aug_type][0]
            augmented = fn(img)

            save_path = _save_with_manifest(
                augmented, output_dir, prefix, idx + i, manifest,
                meta={
                    "aug_type": aug_type,
                    "source": os.path.basename(src_path),
                    "label": "normal",
                    "description": desc,
                }
            )
            stats[aug_type] += 1
            total_saved += 1

        if verbose:
            print(f"  [{aug_type:22s}] {stats[aug_type]:4d}장 → {output_dir}")

    if verbose:
        print(f"\n[오탐 증강 완료] 총 {total_saved}장 생성")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# 희귀 화재 케이스 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def build_rare_fire_cases(
    fire_dir: str,
    normal_dir: str,
    output_dir: str,
    n_lightning: int = 150,
    n_arson: int = 150,
    manifest: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    낙뢰 화재 · 방화 케이스를 합성 생성.
    비율: 전체 학습 데이터의 3~5% 이내 권장.

    Args:
        fire_dir:    abnormal(화재) 이미지 경로 (패치 추출용)
        normal_dir:  normal 이미지 경로 (배경 이미지)
        output_dir:  저장 경로 (abnormal/ 하위에 저장 권장)
        n_lightning: 낙뢰 화재 생성 수 (flash + fire → 각 n_lightning장)
        n_arson:     방화 생성 수
    """
    os.makedirs(output_dir, exist_ok=True)
    if manifest is None:
        manifest = {}

    # 화염 패치 수집
    if verbose:
        print("[FirePatchExtractor] 화염 패치 추출 중...")
    extractor = FirePatchExtractor()
    patches = extractor.extract_from_dir(fire_dir, max_images=80)

    if len(patches) < 5:
        if verbose:
            print(f"  패치 {len(patches)}개 — 합성 패치(fallback) 사용")
    generator = RareFireGenerator(patches if patches else None)

    bg_paths = _list_images(normal_dir)
    if not bg_paths:
        print(f"[WARNING] 배경 이미지 없음: {normal_dir}")
        return manifest

    stats = {"lightning_flash": 0, "lightning_fire": 0, "arson": 0}

    # ── 낙뢰 화재 ─────────────────────────────────────────────
    flash_idx = _next_index(output_dir, "aug_lightning_flash")
    fire_idx = _next_index(output_dir, "aug_lightning_fire")

    for i in range(n_lightning):
        bg = _load_image(random.choice(bg_paths))
        if bg is None:
            continue
        bg = cv2.resize(bg, (224, 224))

        flash_frame, fire_frame, bboxes = generator.lightning_fire(bg)

        _save_with_manifest(flash_frame, output_dir,
                            "aug_lightning_flash", flash_idx + i, manifest,
                            {"aug_type": "lightning_flash",
                             "label": "normal",
                             "description": "낙뢰 섬광 프레임 (화재 전)",
                             "bboxes": []})
        _save_with_manifest(fire_frame, output_dir,
                            "aug_lightning_fire", fire_idx + i, manifest,
                            {"aug_type": "lightning_fire",
                             "label": "abnormal",
                             "description": "낙뢰 화재 발생 (연기 없는 즉각 발화)",
                             "bboxes": bboxes})
        stats["lightning_flash"] += 1
        stats["lightning_fire"] += 1

    # ── 방화 ──────────────────────────────────────────────────
    arson_idx = _next_index(output_dir, "aug_arson")
    for i in range(n_arson):
        bg = _load_image(random.choice(bg_paths))
        if bg is None:
            continue
        bg = cv2.resize(bg, (224, 224))

        fire_frame, bboxes = generator.arson(bg)
        _save_with_manifest(fire_frame, output_dir,
                            "aug_arson", arson_idx + i, manifest,
                            {"aug_type": "arson",
                             "label": "abnormal",
                             "description": f"방화 ({len(bboxes)}개 발화 지점, 연기 없음)",
                             "bboxes": bboxes})
        stats["arson"] += 1

    total = sum(stats.values())
    if verbose:
        print(f"  낙뢰 섬광(normal): {stats['lightning_flash']}장")
        print(f"  낙뢰 화재(abnormal): {stats['lightning_fire']}장")
        print(f"  방화(abnormal):    {stats['arson']}장")
        print(f"[희귀 화재 완료] 총 {total}장 생성")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 통계 리포트
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(data_root: str):
    """클래스별·증강 타입별 현황 출력."""
    normal_dir = os.path.join(data_root, "normal")
    abnormal_dir = os.path.join(data_root, "abnormal")

    print("\n" + "═" * 55)
    print("  데이터셋 현황")
    print("═" * 55)

    for split, d in [("normal", normal_dir), ("abnormal", abnormal_dir)]:
        paths = _list_images(d)
        prefix_counts = defaultdict(int)
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            # 접두어 추출 (마지막 숫자 부분 제거)
            parts = stem.rsplit("_", 1)
            prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            prefix_counts[prefix] += 1

        print(f"\n  [{split}] 총 {len(paths):,}장")
        for prefix, cnt in sorted(prefix_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(30, cnt // max(1, len(paths) // 30))
            print(f"    {prefix:25s} {cnt:6,}장  {bar}")

    print("\n" + "═" * 55)

    # 불균형 경고
    n_normal = len(_list_images(normal_dir)) if os.path.exists(normal_dir) else 0
    n_abnormal = len(_list_images(abnormal_dir)) if os.path.exists(abnormal_dir) else 0
    if n_normal + n_abnormal > 0:
        ratio = n_abnormal / (n_normal + n_abnormal)
        print(f"  클래스 비율: normal={n_normal:,}  abnormal={n_abnormal:,}  "
              f"(fire ratio={ratio:.1%})")
        if ratio < 0.15:
            print("  ⚠  화재 비율이 낮습니다 (권장 20~35%). "
                  "--n_rare_fire 를 늘리거나 class_weight 조정 권장.")
        elif ratio > 0.50:
            print("  ⚠  화재 비율이 높습니다. 오탐 증강을 더 추가하세요.")
        else:
            print("  ✓  클래스 비율 양호")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_augmentation_pipeline(
    class_name: str = "fireimage",
    data_root: Optional[str] = None,
    n_false_positive: int = 5000,
    n_rare_fire: int = 300,
    aug_types: Optional[list] = None,
    dry_run: bool = False,
    visualize: bool = False,
    verbose: bool = True,
):
    """
    전체 증강 파이프라인 실행.

    Args:
        class_name:       데이터 클래스 이름 (data/<class_name>/ 구조)
        data_root:        데이터 루트 경로 (None → ./data/<class_name>)
        n_false_positive: 생성할 오탐 증강 이미지 수
        n_rare_fire:      생성할 희귀 화재 이미지 수 (낙뢰+방화 합산)
        dry_run:          True이면 파일 저장 없이 시각화만
        visualize:        True이면 증강 샘플 그리드 저장
    """
    if data_root is None:
        data_root = os.path.join(".", "data", class_name)

    normal_dir = os.path.join(data_root, "normal")
    abnormal_dir = os.path.join(data_root, "abnormal")

    if not os.path.exists(normal_dir):
        print(f"[ERROR] normal 디렉터리 없음: {normal_dir}")
        return

    print(f"\n{'═'*55}")
    print(f"  화재 감지 데이터 증강 파이프라인")
    print(f"  class: {class_name}  |  data_root: {data_root}")
    print(f"{'═'*55}\n")

    # ── 시각화 샘플 (dry_run 또는 visualize 시) ──────────────
    if dry_run or visualize:
        sample_paths = _list_images(normal_dir)
        bg_paths = _list_images(abnormal_dir)

        if sample_paths:
            sample_img = _load_image(random.choice(sample_paths))
            if sample_img is not None:
                sample_img = cv2.resize(sample_img, (224, 224))

                os.makedirs("results_augmentation", exist_ok=True)

                augmentor = FalsePositiveAugmentor()
                augmentor.visualize_all(
                    sample_img,
                    save_path="results_augmentation/false_positive_grid.png"
                )
                print("[시각화] results_augmentation/false_positive_grid.png 저장")

                if bg_paths:
                    fire_paths = _list_images(abnormal_dir)
                    extractor = FirePatchExtractor()
                    patches = extractor.extract_from_dir(
                        abnormal_dir, max_images=30)
                    gen = RareFireGenerator(patches or None)
                    bg = _load_image(random.choice(sample_paths))
                    if bg is not None:
                        bg = cv2.resize(bg, (224, 224))
                        gen.visualize_cases(
                            bg,
                            save_path="results_augmentation/rare_fire_grid.png"
                        )
                        print("[시각화] results_augmentation/rare_fire_grid.png 저장")

        if dry_run:
            print("\n[Dry Run] 파일 저장 없이 종료.")
            return

    manifest = {}

    # ── 오탐 증강 ─────────────────────────────────────────────
    print(f"[1/2] 오탐 증강 시작  (목표: {n_false_positive}장)")
    manifest = build_false_positive_augmentations(
        normal_dir=normal_dir,
        output_dir=normal_dir,
        n_total=n_false_positive,
        aug_types=aug_types,
        manifest=manifest,
        verbose=verbose,
    )

    # ── 희귀 화재 합성 ─────────────────────────────────────────
    n_lightning = n_rare_fire // 2
    n_arson = n_rare_fire - n_lightning
    print(f"\n[2/2] 희귀 화재 합성  (낙뢰:{n_lightning} / 방화:{n_arson})")
    manifest = build_rare_fire_cases(
        fire_dir=abnormal_dir,
        normal_dir=normal_dir,
        output_dir=abnormal_dir,
        n_lightning=n_lightning,
        n_arson=n_arson,
        manifest=manifest,
        verbose=verbose,
    )

    # ── 매니페스트 저장 ───────────────────────────────────────
    manifest_path = os.path.join(data_root, "augmentation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[매니페스트] {manifest_path} ({len(manifest):,}개 항목)")

    # ── 최종 통계 ─────────────────────────────────────────────
    print_dataset_stats(data_root)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="화재 감지 데이터 증강 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행
  python data_augmentation/dataset_builder.py --class_name fireimage

  # 커스텀 장수
  python data_augmentation/dataset_builder.py \\
      --n_false_positive 8000 --n_rare_fire 400

  # 시각화 샘플만 확인 (저장 안 함)
  python data_augmentation/dataset_builder.py --dry_run

  # 특정 증강 타입만 사용
  python data_augmentation/dataset_builder.py \\
      --aug_types autumn_foliage sunset_backlight yellow_dust
        """
    )
    p.add_argument("--class_name", default="fireimage",
                   help="데이터 클래스 이름 (기본: fireimage)")
    p.add_argument("--data_root", default=None,
                   help="데이터 루트 경로 (기본: ./data/<class_name>)")
    p.add_argument("--n_false_positive", type=int, default=5000,
                   help="생성할 오탐 증강 이미지 수 (기본: 5000)")
    p.add_argument("--n_rare_fire", type=int, default=300,
                   help="생성할 희귀 화재 이미지 수 (기본: 300)")
    p.add_argument("--aug_types", nargs="+", default=None,
                   choices=list(AUG_REGISTRY.keys()),
                   help="사용할 증강 타입 (기본: 전체 11가지)")
    p.add_argument("--dry_run", action="store_true",
                   help="파일 저장 없이 시각화 샘플만 생성")
    p.add_argument("--visualize", action="store_true",
                   help="증강 결과 그리드 이미지 저장")
    p.add_argument("--stats_only", action="store_true",
                   help="증강 없이 현재 데이터셋 통계만 출력")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.stats_only:
        data_root = args.data_root or os.path.join(".", "data", args.class_name)
        print_dataset_stats(data_root)
    else:
        run_augmentation_pipeline(
            class_name=args.class_name,
            data_root=args.data_root,
            n_false_positive=args.n_false_positive,
            n_rare_fire=args.n_rare_fire,
            aug_types=args.aug_types,
            dry_run=args.dry_run,
            visualize=args.visualize,
        )
