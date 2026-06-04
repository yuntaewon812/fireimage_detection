"""
model_loader.py — 학습된 .pt 가중치를 로드해 FireDetectionPipeline에 연결.

모델 인스턴스 생성 규약은 main_v2.py 의 model_dicts 와 동일하게 맞춘다.
가중치 탐색 순서:
  1) model_save/fireimage/fold{fold}/{name}.pt
  2) valid_models_backup/fold{fold}/{name}.pt   (백업본)
가중치가 없으면 무작위 초기화 상태로 반환(UI 테스트용)하고 경고를 남긴다.
"""
import os
import functools
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 지연 임포트 — DeepModel(ResNet/DenseNet)만 쓸 때 timm 등 미설치여도 동작
def _build_deep(variant):
    from models.deep_model import DeepModel
    return DeepModel(variant)

def _build_effv2():
    from models.efficientnetv2 import EfficientNetV2ForImageClassification
    return EfficientNetV2ForImageClassification(num_labels=2, img_size=160,
        patch_size=16, hidden_dim=512, model_variant="s")

def _build_effv2_prop():
    from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
    return EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=160,
        patch_size=16, hidden_dim=512, model_variant="s")

def _build_nextvit():
    from models.nextvit import NextViTForImageClassification
    return NextViTForImageClassification(num_labels=2, img_size=160,
        patch_size=16, hidden_dim=512, model_variant="small")

def _build_maxvit():
    from models.maxvit import MaxViTForImageClassification
    return MaxViTForImageClassification(num_labels=2, img_size=160,
        patch_size=16, hidden_dim=512, model_variant="tiny")

def _build_internimage():
    from models.internimage import InternImageForImageClassification
    return InternImageForImageClassification(num_labels=2, img_size=160,
        patch_size=16, hidden_dim=512, model_variant="tiny")

# main_v2.py 와 동일한 생성자 (img_size=160 학습 기준)
MODEL_BUILDERS = {
    "Resnet50":                lambda: _build_deep("ResNet50"),
    "DenseNet121":             lambda: _build_deep("DenseNet121"),
    "efficientnetv2":          _build_effv2,
    "efficientnetv2_proposal": _build_effv2_prop,
    "nextvit":                 _build_nextvit,
    "maxvit":                  _build_maxvit,
    "internimage":             _build_internimage,
}

AVAILABLE_MODELS = list(MODEL_BUILDERS.keys())

# ──────────────────────────────────────────────────────────────
# 데모에 노출할 상위 모델 (성능평가 결과로 결정)
# v25 GPU 재학습 분포내(fold0+fold2) 성능평가 결과로 선정:
#   1위 maxvit         F1 1.000 / Loss 0.000
#   2위 efficientnetv2 F1 0.998 / Loss 0.002 (internimage는 동급이나 DCNv3로 추론 느려 제외)
# ──────────────────────────────────────────────────────────────
TOP_MODELS = ["maxvit", "efficientnetv2"]


def rank_top_models(k: int = 2) -> list[str]:
    """metrics.csv 에서 fold 평균 F1 상위 k개 모델명을 자동 산출.
    파일이 없으면 TOP_MODELS 기본값 반환."""
    import csv
    from collections import defaultdict
    path = os.path.join(PROJECT_ROOT, "results", "fireimage", "metrics.csv")
    if not os.path.exists(path):
        return TOP_MODELS[:k]
    f1 = defaultdict(list)
    with open(path, encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            name = row["model name"].rsplit("_", 1)[0]   # Resnet50_0 → Resnet50
            try:
                f1[name].append(float(row["F1 score"].split("(")[0]))
            except (KeyError, ValueError):
                continue
    # MODEL_BUILDERS 에 있고 + 가중치가 실제 존재하는 모델만 후보로
    def has_w(n):
        return _weight_path(n, 0) is not None
    if not f1:
        avail = [n for n in TOP_MODELS if has_w(n)]
        return (avail or TOP_MODELS)[:k]
    ranked = sorted(f1, key=lambda n: sum(f1[n]) / len(f1[n]), reverse=True)
    ranked = [n for n in ranked if n in MODEL_BUILDERS and has_w(n)]
    if len(ranked) < k:   # 가중치 있는 모델로 보충
        for n in TOP_MODELS:
            if n not in ranked and has_w(n):
                ranked.append(n)
    return ranked[:k] if ranked else TOP_MODELS[:k]


def _weight_path(name: str, fold: int) -> str | None:
    candidates = [
        os.path.join(PROJECT_ROOT, "model_save", "fireimage", f"fold{fold}", f"{name}.pt"),
        os.path.join(PROJECT_ROOT, "valid_models_backup", f"fold{fold}", f"{name}.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@functools.lru_cache(maxsize=8)
def load_model(name: str, fold: int = 0):
    """모델 인스턴스 + 가중치 로드. (캐시됨)"""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"지원하지 않는 모델: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_BUILDERS[name]().to(device)

    wpath = _weight_path(name, fold)
    if wpath:
        state = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        loaded = True
    else:
        print(f"[경고] 가중치 없음: {name} fold{fold} → 무작위 초기화 (UI 테스트용)")
        loaded = False

    # DeepModel(ResNet/DenseNet)은 forward 끝에 Softmax를 적용 → 파이프라인의
    # softmax와 이중 적용되어 신뢰도가 뭉개짐. 추론 시 logit 반환하도록 제거.
    if hasattr(model, "activation") and isinstance(model.activation, torch.nn.Softmax):
        model.activation = torch.nn.Identity()

    model.eval()
    return model, loaded


def weight_status() -> dict[str, dict[int, bool]]:
    """각 모델/폴드의 가중치 존재 여부 맵."""
    return {n: {f: _weight_path(n, f) is not None for f in (0, 1, 2)}
            for n in AVAILABLE_MODELS}
