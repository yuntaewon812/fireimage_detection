"""
main_ablation.py — OOD 성능 개선 Ablation 실험
================================================
목적: pretrained 백본 + online 증강 + mixup의 OOD(처음 보는 영상) 기여도 측정.

실험 설정 (누적 ablation):
  설정 A: baseline      (weights=None, 증강X)       ← v25에 이미 있음, 재사용
  설정 B: +pretrained   (ImageNet 사전학습)
  설정 C: +pretrained   + 증강 (ColorJitter/Flip/Rotation/Cutout)
  설정 D: +pretrained   + 증강 + Mixup

각 설정은 grouped 분할(StratifiedGroupKFold, random_state=1004)로 학습 →
  OOD fold(YouTube test 전용 fold)의 F1을 측정해 기법별 기여도를 비교.

결과 저장:
  model_save/fireimage_abl_{설정}/
  results/fireimage_abl_{설정}/metrics.csv

사용법:
  python main_ablation.py --class_name fireimage --setting B   # pretrained만
  python main_ablation.py --class_name fireimage --setting C   # pretrained+증강
  python main_ablation.py --class_name fireimage --setting D   # +mixup
  python main_ablation.py --class_name fireimage --setting all  # B→C→D 순차

설정 A(baseline)는 v25 results/fireimage/metrics.csv 그대로 참조 (재학습 X).
"""
import os
import argparse
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from utils.utils import load_data_with_groups, get_args as _get_base_args
from utils.train_loop import train_full_loop

from models.deep_model import DeepModel
from models.efficientnetv2 import (EfficientNetV2ForImageClassification,
                                    EfficientNetV2ForImageClassification_v2)
from models.nextvit import NextViTForImageClassification
from models.maxvit import MaxViTForImageClassification
from models.internimage import InternImageForImageClassification

os.makedirs('./model_save', exist_ok=True)
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# ──────────────────────────────────────────────────────────────
SETTINGS = {
    # 설정 A = baseline (v25), 여기선 B/C/D만 실행
    'B': dict(pretrained=True,  augment=False, mixup=False,
              desc='사전학습(ImageNet) 백본'),
    'C': dict(pretrained=True,  augment=True,  mixup=False,
              desc='사전학습 + Online 증강(ColorJitter/Flip/Rotation/Cutout)'),
    'D': dict(pretrained=True,  augment=True,  mixup=True,
              desc='사전학습 + 증강 + Mixup(α=0.4)'),
}

PER_MODEL_LR = {
    'Resnet50': 1e-4, 'DenseNet121': 1e-4,
    'efficientnetv2': 3e-4, 'efficientnetv2_proposal': 3e-4,
    'nextvit': 1e-4,   # nextvit 안정화 LR 유지
    'maxvit': 3e-4, 'internimage': 5e-5,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--class_name', default='fireimage')
    p.add_argument('--setting', default='B', choices=['B','C','D','all'],
                   help='ablation 설정: B=pretrained, C=+증강, D=+mixup, all=B→C→D')
    return p.parse_args()


def make_models(pretrained: bool) -> dict:
    return {
        'Resnet50':                DeepModel('ResNet50', pretrained=pretrained),
        'DenseNet121':             DeepModel('DenseNet121', pretrained=pretrained),
        'efficientnetv2':          EfficientNetV2ForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='s',
                                       pretrained=pretrained),
        'efficientnetv2_proposal': EfficientNetV2ForImageClassification_v2(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='s',
                                       pretrained=pretrained),
        'nextvit':                 NextViTForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='small',
                                       pretrained=pretrained),
        'maxvit':                  MaxViTForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='tiny',
                                       pretrained=pretrained),
        'internimage':             InternImageForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='tiny',
                                       pretrained=pretrained),
    }


def run_setting(setting_key: str, class_name: str, X, y, groups):
    cfg = SETTINGS[setting_key]
    save_name = f'{class_name}_abl_{setting_key}'
    print(f'\n{"="*55}')
    print(f'  설정 {setting_key}: {cfg["desc"]}')
    print(f'  저장: model_save/{save_name}/  results/{save_name}/')
    print(f'{"="*55}')

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1004)
    for fold_idx, (train_index, test_index) in enumerate(
            sgkf.split(X.cpu().numpy(), y.cpu().numpy(), groups)):
        train_idx, val_idx = train_test_split(
            train_index, test_size=0.25, random_state=1004,
            stratify=y[train_index].numpy())

        train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                                  batch_size=8, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X[val_idx],   y[val_idx]),
                                  batch_size=8, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X[test_index], y[test_index]),
                                  batch_size=8, shuffle=False)

        train_full_loop(
            train_loader, val_loader, test_loader,
            make_models(cfg['pretrained']),
            save_name, fold_idx,
            lr=3e-4, epochs=30, patience=8,
            per_model_lr=PER_MODEL_LR,
            force_retrain=set(make_models(True).keys()),  # 항상 재학습
            augment=cfg['augment'],
            use_mixup=cfg['mixup'],
        )
    print(f'\n  ✓ 설정 {setting_key} 완료. results/{save_name}/metrics.csv 확인')


def main():
    args = parse_args()
    now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    print(f'main_ablation.py 시작: {now:%Y-%m-%d %H:%M:%S}')
    print(f'설정: {args.setting}')
    print('\n[데이터 로드] grouped(StratifiedGroupKFold) — v25와 동일 분할')

    X, y, groups = load_data_with_groups(args.class_name, img_size=(160, 160))
    n_normal, n_abnormal = int((y==0).sum()), int((y==1).sum())
    print(f'  총 {len(X):,}  | normal={n_normal:,} / abnormal={n_abnormal:,}')
    assert n_normal > 0 and n_abnormal > 0

    to_run = list(SETTINGS.keys()) if args.setting == 'all' else [args.setting]
    for s in to_run:
        run_setting(s, args.class_name, X, y, groups)

    print('\n[전체 완료] ablation 결과 비교:')
    import csv
    models = ['Resnet50','DenseNet121','efficientnetv2','efficientnetv2_proposal',
              'nextvit','maxvit','internimage']
    # baseline(v25) OOD fold
    print('\n기법별 OOD fold F1 (가장 붕괴한 fold 기준, YouTube test fold)')
    header = f"{'모델':<26}  A(baseline)  " + "  ".join(f"  {s}({SETTINGS[s]['desc'][:10]})" for s in to_run)
    print(header)
    print('-' * len(header))

    def get_ood_f1(csv_path, model_name):
        if not os.path.exists(csv_path):
            return None
        rows = {}
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows[r['model name']] = float(r['F1 score'].split('(')[0])
        # OOD fold = F1이 가장 낮은 fold
        vals = [rows.get(f'{model_name}_{i}') for i in range(3)]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    baseline_csv = f'./results/{args.class_name}/metrics.csv'
    for m in models:
        b = get_ood_f1(baseline_csv, m)
        row = f'{m:<26}  {b:.3f if b else "  -  ":>11}'
        for s in to_run:
            p = f'./results/{args.class_name}_abl_{s}/metrics.csv'
            v = get_ood_f1(p, m)
            row += f'  {v:.3f if v else "  -  ":>10}'
        print(row)

if __name__ == '__main__':
    main()
