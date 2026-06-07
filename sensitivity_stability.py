"""
sensitivity_stability.py
XAI Sensitivity & Stability 정량 평가 + 실행 시간 측정
  - Sensitivity : E(x)와 E(x+noise)의 평균 L2 거리 (낮을수록 robust)
  - Stability   : N회 perturbation 설명의 픽셀별 std 평균 (낮을수록 안정)
결과: results_SENS_STAB/sens_stab_{cpu|cuda}.csv
"""
import os, sys, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from pytorch_grad_cam import GradCAM
from lime.lime_image import LimeImageExplainer

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from utils.utils import load_data_with_groups

# ── Config ──────────────────────────────────────────────────────────────────
N_IMAGES  = 20
N_PERTURB = 5
SIGMA     = 0.05
LIME_SAMPLES = 20
SEED      = 1004
IMG_SIZE  = 160
FOLD      = 0
WEIGHT_DIR = os.path.join(BASE, 'model_save', 'fireimage_abl_E', 'fold0')
OUT_DIR    = os.path.join(BASE, 'results_SENS_STAB')
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── 모델 빌드 ────────────────────────────────────────────────────────────────
def build_efficientnetv2(path, dev):
    import timm
    bb = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0, global_pool='avg')
    nf = bb.num_features
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.classifier = nn.Sequential(
                nn.LayerNorm(nf), nn.Linear(nf, 512),
                nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2))
        def forward(self, x):
            return self.classifier(self.backbone(x))
    m = M()
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
    return m.to(dev).eval()

def build_maxvit(path, dev):
    import timm
    bb = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=False, num_classes=0, global_pool='avg')
    nf = bb.num_features
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.classifier = nn.Sequential(
                nn.LayerNorm(nf), nn.Linear(nf, 512),
                nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2))
        def forward(self, x):
            x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
            return self.classifier(self.backbone(x))
    m = M()
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
    return m.to(dev).eval()

# ── XAI 함수 ─────────────────────────────────────────────────────────────────
def xai_gradcam(model, x, model_name):
    if 'efficient' in model_name:
        tl = [model.backbone.conv_head]
    else:
        tl = [model.backbone.stages[-1].blocks[-1].conv.conv3_1x1]
    cam = GradCAM(model=model, target_layers=tl)
    return cam(input_tensor=x, targets=None)[0]

def xai_saliency(model, x):
    x_ = x.clone().detach().requires_grad_(True)
    out = model(x_)
    model.zero_grad()
    out[0, out.argmax(1).item()].backward()
    return torch.max(x_.grad.data.abs(), dim=1)[0].squeeze().detach().cpu().numpy()

def xai_ig(model, x):
    """Integrated Gradients (SHAP equivalent, zero baseline, 20 steps)"""
    baseline = torch.zeros_like(x)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    acc = torch.zeros_like(x)
    for k in range(1, 21):
        interp = (baseline + k / 20 * (x - baseline)).requires_grad_(True)
        out = model(interp)
        model.zero_grad()
        out[0, pred].backward()
        acc = acc + interp.grad.data.detach()
    ig = (x - baseline) * (acc / 20)
    return ig.abs().squeeze(0).mean(0).cpu().numpy()

def xai_lime(model, x, dev):
    np_img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float64)
    def clf_fn(imgs):
        with torch.no_grad():
            b = torch.tensor(imgs).permute(0, 3, 1, 2).float().to(dev)
            return torch.softmax(model(b), dim=1).cpu().numpy()
    exp = LimeImageExplainer().explain_instance(
        np_img, clf_fn, top_labels=1, hide_color=0, num_samples=LIME_SAMPLES)
    lbl = exp.top_labels[0]
    hmap = np.zeros(np_img.shape[:2], dtype=np.float32)
    for seg_id, w in dict(exp.local_exp[lbl]).items():
        hmap[exp.segments == seg_id] = w
    return np.abs(hmap)

def normalize(m):
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)

# ── 핵심 계산 ─────────────────────────────────────────────────────────────────
def compute_sens_stab(model, xi, map_fn):
    base = normalize(map_fn(model, xi))
    pmaps = []
    for _ in range(N_PERTURB):
        xp = (xi + torch.randn_like(xi) * SIGMA).clamp(0, 1)
        pmaps.append(normalize(map_fn(model, xp)))
    sensitivity = float(np.mean([np.sqrt(np.mean((base - p) ** 2)) for p in pmaps]))
    stability   = float(np.mean(np.std(np.stack(pmaps, 0), axis=0)))
    return sensitivity, stability

# ── 데이터 ───────────────────────────────────────────────────────────────────
print('\n[데이터 로드]')
X, y, groups = load_data_with_groups('fireimage', img_size=(IMG_SIZE, IMG_SIZE))
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
_, test_idx = list(sgkf.split(X, y, groups))[FOLD]
sel = np.random.default_rng(SEED).choice(test_idx, min(N_IMAGES, len(test_idx)), replace=False)
X_eval = X[sel].to(device)
print(f'  평가 이미지: {len(sel)}장  (fold{FOLD} test set)')

# ── 평가 ─────────────────────────────────────────────────────────────────────
MODELS = {
    'efficientnetv2': (build_efficientnetv2, 'efficientnetv2.pt'),
    'maxvit':         (build_maxvit,          'maxvit.pt'),
}

rows = []
for model_name, (build_fn, fname) in MODELS.items():
    model = build_fn(os.path.join(WEIGHT_DIR, fname), device)
    print(f'\n[{model_name}]')

    methods = {
        'GradCAM':  lambda m, x, mn=model_name: xai_gradcam(m, x, mn),
        'Saliency': xai_saliency,
        'SHAP(IG)': xai_ig,
        'LIME':     lambda m, x, d=device: xai_lime(m, x, d),
    }

    for method_name, fn in methods.items():
        s_list, st_list = [], []
        t0 = time.time()
        for i, xi in enumerate(X_eval):
            try:
                s, st = compute_sens_stab(model, xi.unsqueeze(0), fn)
                s_list.append(s); st_list.append(st)
            except Exception as e:
                print(f'  [{method_name}] img{i} 에러: {e}')
        elapsed = time.time() - t0

        s_mean  = float(np.mean(s_list))  if s_list  else float('nan')
        st_mean = float(np.mean(st_list)) if st_list else float('nan')
        print(f'  {method_name:10s}  Sensitivity={s_mean:.4f}  Stability={st_mean:.4f}'
              f'  Time={elapsed:.1f}s  ({len(s_list)}/{len(X_eval)})')

        rows.append({
            'Model': model_name, 'Method': method_name,
            'Sensitivity': round(s_mean, 6), 'Stability': round(st_mean, 6),
            'N_images': len(s_list), 'Time_sec': round(elapsed, 2),
            'Device': str(device),
        })

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, f'sens_stab_{device.type}.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['Model','Method','Sensitivity','Stability','N_images','Time_sec','Device'])
    w.writeheader(); w.writerows(rows)
print(f'\n[저장] {out_csv}')
