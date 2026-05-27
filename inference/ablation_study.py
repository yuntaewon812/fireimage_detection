"""
Ablation Study — 파이프라인 구성 요소별 기여도 측정
====================================================
논문용 Ablation 표를 자동 생성.

실험 설계:
  Config                        제거 모듈
  ─────────────────────────────────────────────────
  Full System                   없음 (기준)
  w/o SAM                       SAM
  w/o Optical Flow              Optical Flow
  w/o HSV                       HSV 라우팅
  w/o Wind Gate                 풍향 게이트
  w/o SAM + Optical Flow        SAM + Optical Flow
  GradCAM Only                  SAM + OF + HSV + Wind Gate

사용법:
  python inference/ablation_study.py \\
      --class_name fireimage \\
      --model_name efficientnetv2 \\
      --fold 0

  # 전체 fold 평균
  python inference/ablation_study.py \\
      --class_name fireimage \\
      --model_name efficientnetv2 \\
      --all_folds
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.fire_pipeline import (
    FireDetectionPipeline, PipelineConfig, MockVLMAnalyzer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 구성 정의
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, Dict] = {
    "Full System": dict(
        use_hsv=True, use_optical_flow=True, use_wind_gate=True,
        use_sam=True, use_uncertainty=True,
    ),
    "w/o SAM": dict(
        use_hsv=True, use_optical_flow=True, use_wind_gate=True,
        use_sam=False, use_uncertainty=True,
    ),
    "w/o Optical Flow": dict(
        use_hsv=True, use_optical_flow=False, use_wind_gate=True,
        use_sam=True, use_uncertainty=True,
    ),
    "w/o HSV": dict(
        use_hsv=False, use_optical_flow=True, use_wind_gate=True,
        use_sam=True, use_uncertainty=True,
    ),
    "w/o Wind Gate": dict(
        use_hsv=True, use_optical_flow=True, use_wind_gate=False,
        use_sam=True, use_uncertainty=True,
    ),
    "w/o SAM + Optical Flow": dict(
        use_hsv=True, use_optical_flow=False, use_wind_gate=True,
        use_sam=False, use_uncertainty=True,
    ),
    "GradCAM Only": dict(
        use_hsv=False, use_optical_flow=False, use_wind_gate=False,
        use_sam=False, use_uncertainty=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로더
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, class_name: str, fold: int) -> torch.nn.Module:
    """저장된 .pt 로드."""
    save_path = PROJECT_ROOT / "model_save" / class_name / f"fold{fold}" / f"{model_name}.pt"
    if not save_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {save_path}")

    model = _build_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(str(save_path), map_location=device))
    model.eval()
    return model


def _build_model(model_name: str) -> torch.nn.Module:
    n = model_name.lower()
    if "efficientnetv2_proposal" in n:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        return EfficientNetV2ForImageClassification_v2(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="s")
    if "efficientnetv2" in n:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification
        return EfficientNetV2ForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="s")
    if "nextvit" in n:
        from models.nextvit import NextViTForImageClassification
        return NextViTForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="small")
    if "maxvit" in n:
        from models.maxvit import MaxViTForImageClassification
        return MaxViTForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="tiny")
    if "internimage" in n:
        from models.internimage import InternImageForImageClassification
        return InternImageForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="tiny")
    if "resnet" in n:
        from models.deep_model import DeepModel
        return DeepModel("ResNet50")
    if "densenet" in n:
        from models.deep_model import DeepModel
        return DeepModel("DenseNet121")
    raise ValueError(f"지원하지 않는 모델명: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 데이터 로더
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(class_name: str, fold: int,
                   with_prev_frame: bool = False) -> List[Dict]:
    """
    data/<class_name>/normal|abnormal 에서 이미지를 로드해
    test_cases 리스트 반환.

    with_prev_frame=True: 연속 이미지 쌍 구성 (Optical Flow용)
    """
    import cv2
    import glob
    from sklearn.model_selection import StratifiedKFold, train_test_split

    data_root = PROJECT_ROOT / "data" / class_name
    X_paths, y = [], []

    for label, subfolder in [(0, "normal"), (1, "abnormal")]:
        folder = data_root / subfolder
        if not folder.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(str(folder / "**" / ext), recursive=True):
                X_paths.append(p)
                y.append(label)

    if not X_paths:
        print(f"[WARNING] 이미지 없음: {data_root}")
        return []

    X_paths = np.array(X_paths)
    y       = np.array(y)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1004)
    splits = list(skf.split(X_paths, y))
    _, test_idx = splits[fold]

    test_cases = []
    for idx in test_idx:
        path = X_paths[idx]
        label = int(y[idx])

        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)

        case: Dict[str, Any] = {
            "image": img_rgb,
            "label": label,
            "path": path,
            "case_type": "fire" if label == 1 else "normal",
        }

        if with_prev_frame:
            # 같은 이미지를 prev로 사용 (실제 영상 없을 때 폴백)
            case["prev_frame"] = img_rgb

        test_cases.append(case)

    print(f"[데이터] fold{fold} 테스트셋: {len(test_cases)}장 "
          f"(fire={sum(c['label']==1 for c in test_cases)}, "
          f"normal={sum(c['label']==0 for c in test_cases)})")
    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """F1, Precision, Recall, AUC, Balanced Accuracy."""
    metrics: Dict[str, float] = {}
    metrics["f1"]       = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision"]= float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall"]   = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["bacc"]     = float(balanced_accuracy_score(y_true, y_pred))
    try:
        metrics["auc"]  = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"]  = float("nan")
    return metrics


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray,
                 n_iter: int = 500) -> Dict[str, List[float]]:
    """각 메트릭의 [mean, lower95, upper95] 부트스트랩 CI."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    records = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        try:
            records.append(compute_metrics(y_true[idx], y_pred[idx], y_prob[idx]))
        except Exception:
            pass

    ci: Dict[str, List[float]] = {}
    for key in records[0]:
        vals = [r[key] for r in records if not np.isnan(r[key])]
        ci[key] = [
            float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        ]
    return ci


# ─────────────────────────────────────────────────────────────────────────────
# 단일 Ablation 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(
    config_name: str,
    flags: Dict,
    model: torch.nn.Module,
    model_name: str,
    test_cases: List[Dict],
    wind_direction: Optional[float] = None,
    verbose: bool = False,
) -> Dict:
    """
    단일 Ablation 설정으로 전체 테스트셋 추론 후 메트릭 반환.
    """
    cfg = PipelineConfig(**flags)
    pipeline = FireDetectionPipeline(
        model=model, model_name=model_name,
        vlm=MockVLMAnalyzer(), config=cfg,
    )

    y_true, y_pred, y_prob = [], [], []
    latencies = []

    for case in test_cases:
        prev = case.get("prev_frame") if flags.get("use_optical_flow", True) else None
        try:
            result = pipeline.run(
                case["image"],
                wind_direction=wind_direction,
                prev_frame=prev,
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] {case['path']}: {e}")
            continue

        y_true.append(case["label"])
        y_pred.append(1 if result.fire_detected else 0)
        y_prob.append(result.fire_confidence)
        latencies.append(result.latency_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics  = compute_metrics(y_true, y_pred, y_prob)
    ci       = bootstrap_ci(y_true, y_pred, y_prob)
    avg_lat  = float(np.mean(latencies)) if latencies else 0.0

    print(f"  [{config_name:35s}]  "
          f"F1={metrics['f1']:.4f}  "
          f"Prec={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  "
          f"AUC={metrics['auc']:.4f}  "
          f"Lat={avg_lat:.1f}ms")

    return {
        "config":    config_name,
        "flags":     flags,
        "metrics":   metrics,
        "ci":        ci,
        "avg_latency_ms": avg_lat,
        "n_samples": len(y_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 전체 Ablation Study 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    model_name: str,
    class_name: str,
    fold: int,
    wind_direction: Optional[float] = None,
    n_bootstrap: int = 500,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    7가지 ablation 설정 전체 실행 후 결과 반환.
    """
    print(f"\n{'═'*65}")
    print(f"  Ablation Study — {model_name}  |  fold {fold}  |  class: {class_name}")
    print(f"{'═'*65}")

    # 모델 로드
    try:
        model = load_model(model_name, class_name, fold)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return []

    # 테스트 데이터 로드 (Optical Flow 있는 설정을 위해 prev_frame 포함)
    test_cases = load_test_data(class_name, fold, with_prev_frame=True)
    if not test_cases:
        return []

    # 전체 설정 순회
    results = []
    for config_name, flags in ABLATION_CONFIGS.items():
        row = run_one_config(
            config_name=config_name,
            flags=flags,
            model=model,
            model_name=model_name,
            test_cases=test_cases,
            wind_direction=wind_direction,
            verbose=verbose,
        )
        results.append(row)

    # 결과 저장
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "results" / class_name / "ablation")
    os.makedirs(save_dir, exist_ok=True)

    _save_results(results, model_name, class_name, fold, save_dir)
    _print_summary_table(results)
    _plot_ablation(results, model_name, class_name, fold, save_dir)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 저장 / 출력 / 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(results: List[Dict], model_name: str,
                  class_name: str, fold: int, save_dir: str):
    """JSON + CSV 저장."""
    import csv

    # JSON
    json_path = os.path.join(save_dir, f"ablation_{model_name}_fold{fold}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = os.path.join(save_dir, f"ablation_{model_name}_fold{fold}.csv")
    fieldnames = ["config", "F1", "F1_lower", "F1_upper",
                  "Precision", "Recall", "AUC", "Balanced_Acc",
                  "Latency_ms", "n_samples"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ci = r["ci"]
            writer.writerow({
                "config":       r["config"],
                "F1":           f"{ci['f1'][0]:.4f}",
                "F1_lower":     f"{ci['f1'][1]:.4f}",
                "F1_upper":     f"{ci['f1'][2]:.4f}",
                "Precision":    f"{ci['precision'][0]:.4f}",
                "Recall":       f"{ci['recall'][0]:.4f}",
                "AUC":          f"{ci['auc'][0]:.4f}",
                "Balanced_Acc": f"{ci['bacc'][0]:.4f}",
                "Latency_ms":   f"{r['avg_latency_ms']:.1f}",
                "n_samples":    r["n_samples"],
            })

    print(f"\n[저장] {json_path}")
    print(f"[저장] {csv_path}")


def _print_summary_table(results: List[Dict]):
    """터미널 요약 표 출력."""
    full_f1 = results[0]["metrics"]["f1"] if results else 0.0

    print(f"\n{'─'*80}")
    print(f"{'Config':35s} {'F1':>7} {'Δ F1':>8} {'Prec':>7} {'Rec':>7} "
          f"{'AUC':>7} {'Lat(ms)':>9}")
    print(f"{'─'*80}")
    for r in results:
        f1  = r["metrics"]["f1"]
        delta = f1 - full_f1
        sign  = "+" if delta >= 0 else ""
        print(f"{r['config']:35s} "
              f"{f1:7.4f} "
              f"{sign}{delta:7.4f} "
              f"{r['metrics']['precision']:7.4f} "
              f"{r['metrics']['recall']:7.4f} "
              f"{r['metrics']['auc']:7.4f} "
              f"{r['avg_latency_ms']:9.1f}")
    print(f"{'─'*80}")


def _plot_ablation(results: List[Dict], model_name: str,
                   class_name: str, fold: int, save_dir: str):
    """F1 + CI 막대 그래프 저장."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    configs = [r["config"] for r in results]
    f1_means  = [r["ci"]["f1"][0] for r in results]
    f1_lowers = [r["ci"]["f1"][0] - r["ci"]["f1"][1] for r in results]
    f1_uppers = [r["ci"]["f1"][2] - r["ci"]["f1"][0] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(configs))

    colors = ["#2196F3" if c == "Full System" else
              "#F44336" if "GradCAM" in c else
              "#FF9800" for c in configs]

    bars = ax.bar(x, f1_means, color=colors, alpha=0.85, width=0.6,
                  yerr=[f1_lowers, f1_uppers], capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})

    # 수치 표기
    for bar, mean in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    # Full System 기준선
    ax.axhline(f1_means[0], color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"Ablation Study — {model_name} (fold {fold})\n"
                 f"95% Bootstrap CI", fontsize=11)
    ax.set_ylim(max(0, min(f1_means) - 0.08), min(1.02, max(f1_means) + 0.06))
    ax.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color="#2196F3", label="Full System"),
        mpatches.Patch(color="#FF9800", label="Component Removed"),
        mpatches.Patch(color="#F44336", label="Baseline (GradCAM Only)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"ablation_{model_name}_fold{fold}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 전체 Fold 평균 집계
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_folds(model_name: str, class_name: str,
                    n_folds: int = 3, save_dir: Optional[str] = None, **kwargs):
    """
    3 fold 결과를 평균 내어 최종 Ablation 표 출력.
    """
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "results" / class_name / "ablation")

    all_results: Dict[str, List[Dict]] = {}

    for fold in range(n_folds):
        fold_results = run_ablation_study(
            model_name=model_name,
            class_name=class_name,
            fold=fold,
            save_dir=save_dir,
            **kwargs,
        )
        for row in fold_results:
            all_results.setdefault(row["config"], []).append(row)

    if not all_results:
        return

    print(f"\n{'═'*65}")
    print(f"  Ablation Study 최종 요약 ({n_folds}-Fold 평균) — {model_name}")
    print(f"{'═'*65}")
    print(f"{'Config':35s} {'F1 mean':>8} {'F1 std':>8} {'Lat(ms)':>9}")
    print(f"{'─'*65}")

    summary = []
    for config_name, rows in all_results.items():
        f1s  = [r["metrics"]["f1"] for r in rows]
        lats = [r["avg_latency_ms"] for r in rows]
        entry = {
            "config": config_name,
            "f1_mean": float(np.mean(f1s)),
            "f1_std":  float(np.std(f1s)),
            "lat_mean": float(np.mean(lats)),
        }
        summary.append(entry)
        print(f"{config_name:35s}  {entry['f1_mean']:8.4f}  "
              f"{entry['f1_std']:8.4f}  {entry['lat_mean']:9.1f}")

    print(f"{'─'*65}")

    # 요약 JSON 저장
    out_path = os.path.join(save_dir, f"ablation_{model_name}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ablation Study 실행")
    p.add_argument("--class_name",  default="fireimage")
    p.add_argument("--model_name",  default="efficientnetv2",
                   help="평가할 모델명 (model_save/ 내 .pt 파일명 기준)")
    p.add_argument("--fold",        type=int, default=0,
                   help="단일 fold (--all_folds 없을 때)")
    p.add_argument("--all_folds",   action="store_true",
                   help="3-fold 전체 평균 집계")
    p.add_argument("--wind_dir",    type=float, default=None,
                   help="풍향 (도). None이면 풍향 게이트 스킵")
    p.add_argument("--n_bootstrap", type=int, default=500)
    p.add_argument("--save_dir",    default=None,
                   help="결과 저장 경로 (기본: results/<class_name>/ablation/)")
    args = p.parse_args()

    if args.all_folds:
        aggregate_folds(
            model_name=args.model_name,
            class_name=args.class_name,
            n_folds=3,
            save_dir=args.save_dir,
            wind_direction=args.wind_dir,
            n_bootstrap=args.n_bootstrap,
        )
    else:
        run_ablation_study(
            model_name=args.model_name,
            class_name=args.class_name,
            fold=args.fold,
            wind_direction=args.wind_dir,
            n_bootstrap=args.n_bootstrap,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
