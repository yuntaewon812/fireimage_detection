"""
loo_xai.py — LOO run 직후 GradCAM Sensitivity/Stability 측정
============================================================
학습된 모델(가중치 로드 상태)에서 추론만 추가로 수행:
  - Sensitivity : E(x) vs E(x+noise) 의 평균 L2 거리 (낮을수록 robust)
  - Stability   : N회 perturbation 설명의 픽셀별 std 평균 (낮을수록 일관)
GradCAM 타겟 = 모델 내 마지막 nn.Conv2d (pretrained/커스텀 백본 모두 호환)
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM


def _last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def _normalize(m):
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def _gradcam(model, x, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam(input_tensor=x, targets=None)[0]


def eval_gradcam_sens_stab(model, X_eval, device, n_perturb=5, sigma=0.05):
    """X_eval: (N,3,H,W) tensor in [0,1]. returns (sensitivity, stability)."""
    model.eval()
    tl = _last_conv(model)
    if tl is None:
        return float('nan'), float('nan'), 0

    sens_list, stab_list = [], []
    for xi in X_eval:
        x = xi.unsqueeze(0).to(device)
        try:
            base = _normalize(_gradcam(model, x, tl))
            pmaps = []
            for _ in range(n_perturb):
                xp = (x + torch.randn_like(x) * sigma).clamp(0, 1)
                pmaps.append(_normalize(_gradcam(model, xp, tl)))
            sens = float(np.mean([np.sqrt(np.mean((base - p) ** 2)) for p in pmaps]))
            stab = float(np.mean(np.std(np.stack(pmaps, 0), axis=0)))
            sens_list.append(sens)
            stab_list.append(stab)
        except Exception:
            continue

    if not sens_list:
        return float('nan'), float('nan'), 0
    return float(np.mean(sens_list)), float(np.mean(stab_list)), len(sens_list)
