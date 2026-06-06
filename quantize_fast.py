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

# ── 모델 임포트 ───────────────────────────────────────────────────
from models.efficientnetv2 import EfficientNetV2ForImageClassification
from models.maxvit import MaxViTForImageClassification

def build(name):
    if name == 'efficientnetv2':
        return EfficientNetV2ForImageClassification(
            num_labels=2, img_size=160, patch_size=16,
            hidden_dim=512, model_variant='s', pretrained=False)
    return MaxViTForImageClassification(
        num_labels=2, img_size=160, patch_size=16,
        hidden_dim=512, model_variant='tiny', pretrained=False)

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

        # INT8 변환
        m = build(name)
        m.load_state_dict(torch.load(w, map_location='cpu', weights_only=True))
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
