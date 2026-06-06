"""
quantize_fast.py — 경량화 (속도측정 없음, 크기변환만)
대상: efficientnetv2, maxvit  /  설정 E  /  fold 0~2
"""
import os, sys, csv, zipfile, glob

import torch
import torch.nn as nn

BASE = '/content/fireimage_detection'
if not os.path.isdir(BASE):
    BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

CLASS = 'fireimage_abl_E'
TARGET = ['efficientnetv2', 'maxvit']
FOLDS  = [0, 1, 2]

# ── 가중치 zip 자동 탐색 ─────────────────────────────────────────
zip_candidates = [
    '/content/weights_E.zip',
    os.path.join(BASE, 'weights_E.zip'),
    os.path.expanduser('~/Downloads/weights_E.zip'),
]
zip_path = next((z for z in zip_candidates if os.path.exists(z)), None)
model_save = os.path.join(BASE, 'model_save')

if zip_path and not os.path.isdir(os.path.join(model_save, CLASS)):
    print(f'[압축해제] {zip_path}')
    os.makedirs(model_save, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(model_save)
    print('  완료')

# ── 모델 빌드 (state_dict 키 보고 timm 여부 자동 판단) ──────────────
import torch.nn as nn

def build(name, state_dict):
    """state_dict 키를 보고 timm 백본(pretrained=True) vs 자체 구현 판단."""
    keys = list(state_dict.keys())
    is_timm = any('conv_stem' in k or 'stem.0' in k or 'patch_embed' in k
                  for k in keys[:10])

    if name == 'efficientnetv2':
        if is_timm:
            # pretrained=True로 학습 → timm tf_efficientnetv2_s 구조 재현
            import timm
            backbone = timm.create_model('tf_efficientnetv2_s',
                                         pretrained=False, num_classes=0, global_pool='avg')
            num_features = backbone.num_features  # 1280

            class _EffNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(num_features),
                        nn.Linear(num_features, 512),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 2),
                    )
                def forward(self, x):
                    return self.classifier(self.backbone(x))
            return _EffNet()
        else:
            from models.efficientnetv2 import EfficientNetV2ForImageClassification
            return EfficientNetV2ForImageClassification(
                num_labels=2, img_size=160, patch_size=16,
                hidden_dim=512, model_variant='s', pretrained=False)

    elif name == 'maxvit':
        # maxvit pretrained=True는 timm maxvit_tiny_tf_224.in1k 사용
        # timm 키: stem.conv1, stages.0.blocks.0.conv.pre_norm (stem.0 아님)
        # → is_timm 로직과 무관하게 항상 timm 구조 재현
        import timm as _timm
        backbone = _timm.create_model('maxvit_tiny_tf_224.in1k',
                                      pretrained=False, num_classes=0, global_pool='avg')
        num_features = backbone.num_features

        class _MaxViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Sequential(
                    nn.LayerNorm(num_features),
                    nn.Linear(num_features, 512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 2),
                )
            def forward(self, x):
                import torch.nn.functional as F
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                return self.classifier(self.backbone(x))
        return _MaxViT()

    raise ValueError(f'알 수 없는 모델: {name}')

# ── 실행 ─────────────────────────────────────────────────────────
rows = []
print(f'\n{"모델":<22} {"fold":<5} {"원본MB":>7} {"INT8MB":>7} {"압축율":>7}')
print('-' * 50)

for name in TARGET:
    for fold in FOLDS:
        w = os.path.join(model_save, CLASS, f'fold{fold}', f'{name}.pt')
        if not os.path.exists(w):
            print(f'  {name} fold{fold}: 가중치 없음 ({w})')
            continue

        print(f'  {name} fold{fold} 처리 중...', flush=True)

        # 원본 크기
        orig_mb = os.path.getsize(w) / 1024 / 1024

        # state_dict 로드 → 키 확인 → 맞는 모델 빌드
        state = torch.load(w, map_location='cpu', weights_only=True)

        # INT8 변환
        m = build(name, state)
        m.load_state_dict(state)
        m.eval()
        m = torch.quantization.quantize_dynamic(m.cpu(), {nn.Linear}, dtype=torch.qint8)

        # 저장
        out_dir = os.path.join(model_save, CLASS + '_quant', f'fold{fold}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.pt')
        torch.save(m.state_dict(), out_path)

        quant_mb = os.path.getsize(out_path) / 1024 / 1024
        ratio = orig_mb / quant_mb

        print(f'  {name:<22} fold{fold}  {orig_mb:>6.1f}  {quant_mb:>6.1f}  {ratio:>6.2f}x')
        rows.append({'model': name, 'fold': fold,
                     'orig_mb': round(orig_mb,2), 'quant_mb': round(quant_mb,2),
                     'ratio': round(ratio,3)})

print('-' * 50)

# CSV 저장
results_dir = os.path.join(BASE, 'results')
os.makedirs(results_dir, exist_ok=True)
report = os.path.join(results_dir, 'quantization_report.csv')
with open(report, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['model','fold','orig_mb','quant_mb','ratio'])
    w.writeheader(); w.writerows(rows)

print(f'\n[완료] {len(rows)}개 모델 경량화')
print(f'  저장: model_save/{CLASS}_quant/')
print(f'  리포트: results/quantization_report.csv')
