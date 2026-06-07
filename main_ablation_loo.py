"""
main_ablation_loo.py — Leave-One-Out Ablation (efficientnetv2 + maxvit)
=======================================================================
목적: 설정 E(Full = pretrained + 강증강 + mixup)에서 요소를 하나씩 제거해
      각 기법의 '필요성'을 다중 시드로 통계적으로 검증.

LOO variant:
  full          : pretrained + 강증강(youtube) + mixup        ← 기준
  no_pretrained : (random init) + 강증강 + mixup
  no_augment    : pretrained + (증강 없음) + mixup
  no_mixup      : pretrained + 강증강 + (mixup 없음)
  no_strongaug  : pretrained + 기본증강(standard) + mixup

다중 시드: --seeds 1004,2024,777  (각 시드마다 모델 초기화/증강 난수 변경)
대상 모델: efficientnetv2, maxvit  (--models 로 변경 가능)

저장:
  model_save/{class}_loo_{variant}_s{seed}/fold{k}/{model}.pt
  results/{class}_loo_{variant}_s{seed}/metrics.csv

사용법:
  python main_ablation_loo.py --seeds 1004,2024,777 --epochs 15
  python main_ablation_loo.py --variants full,no_pretrained --seeds 1004
"""
import os, csv, argparse, random, datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from utils.utils import load_data_with_groups
from utils.train_loop import train_full_loop
from loo_xai import eval_gradcam_sens_stab

from models.efficientnetv2 import EfficientNetV2ForImageClassification
from models.maxvit import MaxViTForImageClassification

N_XAI = 15  # fold당 XAI 평가에 쓸 test 이미지 수

os.makedirs('./model_save', exist_ok=True)
torch.set_num_threads(8)

# ── LOO 설정 매트릭스 ─────────────────────────────────────────────────────────
VARIANTS = {
    'full':          dict(pretrained=True,  augment=True,  mixup=True,  augment_type='youtube'),
    'no_pretrained': dict(pretrained=False, augment=True,  mixup=True,  augment_type='youtube'),
    'no_augment':    dict(pretrained=True,  augment=False, mixup=True,  augment_type='standard'),
    'no_mixup':      dict(pretrained=True,  augment=True,  mixup=False, augment_type='youtube'),
    'no_strongaug':  dict(pretrained=True,  augment=True,  mixup=True,  augment_type='standard'),
}

PER_MODEL_LR = {'efficientnetv2': 3e-4, 'maxvit': 3e-4}


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_models(pretrained: bool, which: list) -> dict:
    pool = {
        'efficientnetv2': lambda: EfficientNetV2ForImageClassification(
            num_labels=2, img_size=160, patch_size=16, hidden_dim=512,
            model_variant='s', pretrained=pretrained),
        'maxvit': lambda: MaxViTForImageClassification(
            num_labels=2, img_size=160, patch_size=16, hidden_dim=512,
            model_variant='tiny', pretrained=pretrained),
    }
    return {m: pool[m]() for m in which}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--class_name', default='fireimage')
    p.add_argument('--variants', default='full,no_pretrained,no_augment,no_mixup,no_strongaug',
                   help='쉼표 구분 LOO variant')
    p.add_argument('--models', default='efficientnetv2,maxvit')
    p.add_argument('--seeds', default='1004,2024,777', help='쉼표 구분 시드')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--patience', type=int, default=5)
    return p.parse_args()


def run_one(variant, seed, class_name, X, y, groups, which_models, epochs, patience):
    cfg = VARIANTS[variant]
    save_name = f'{class_name}_loo_{variant}_s{seed}'
    print(f'\n{"="*60}\n  [{variant}]  seed={seed}  {cfg}\n  저장: {save_name}\n{"="*60}')

    # 분할은 고정(random_state=1004) — 시드는 모델/증강 난수에만 영향
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1004)
    for fold_idx, (train_index, test_index) in enumerate(
            sgkf.split(X.cpu().numpy(), y.cpu().numpy(), groups)):
        set_all_seeds(seed)  # 매 fold마다 동일 시드로 모델 초기화/증강 결정

        train_idx, val_idx = train_test_split(
            train_index, test_size=0.25, random_state=1004,
            stratify=y[train_index].numpy())

        train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                                  batch_size=8, shuffle=True, drop_last=True)
        val_loader   = DataLoader(TensorDataset(X[val_idx], y[val_idx]),
                                  batch_size=8, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X[test_index], y[test_index]),
                                  batch_size=8, shuffle=False)

        train_full_loop(
            train_loader, val_loader, test_loader,
            make_models(cfg['pretrained'], which_models),
            save_name, fold_idx,
            lr=3e-4, epochs=epochs, patience=patience,
            per_model_lr=PER_MODEL_LR,
            force_retrain=set(),
            augment=cfg['augment'],
            use_mixup=cfg['mixup'],
            augment_type=cfg['augment_type'],
        )

        # ── 학습 직후 GradCAM Sensitivity/Stability (추론만 추가) ──
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rng = np.random.default_rng(seed + fold_idx)
        n_eval = min(N_XAI, len(test_index))
        local = rng.choice(len(test_index), n_eval, replace=False)
        X_xai = X[test_index][local].to(device)

        for model_name in which_models:
            wpath = f'./model_save/{save_name}/fold{fold_idx}/{model_name}.pt'
            if not os.path.exists(wpath):
                print(f'  [XAI skip] 가중치 없음: {wpath}')
                continue
            m = make_models(cfg['pretrained'], [model_name])[model_name]
            m.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
            m.to(device).eval()
            sens, stab, n_ok = eval_gradcam_sens_stab(m, X_xai, device)
            print(f'  [XAI] {model_name} fold{fold_idx}  '
                  f'Sensitivity={sens:.4f}  Stability={stab:.4f}  ({n_ok}/{n_eval})')

            xcsv = f'./results/{save_name}/xai_sens_stab.csv'
            os.makedirs(os.path.dirname(xcsv), exist_ok=True)
            new = not os.path.exists(xcsv)
            with open(xcsv, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['model', 'fold', 'seed', 'sensitivity', 'stability', 'n_images'])
                w.writerow([model_name, fold_idx, seed, round(sens, 6), round(stab, 6), n_ok])


def main():
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    models   = [m.strip() for m in args.models.split(',') if m.strip()]
    seeds    = [int(s) for s in args.seeds.split(',') if s.strip()]

    now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    print(f'main_ablation_loo.py 시작: {now:%Y-%m-%d %H:%M:%S}')
    print(f'variants={variants}  models={models}  seeds={seeds}')
    print(f'총 run = {len(variants)} × {len(models)} × 3fold × {len(seeds)}seed '
          f'= {len(variants)*len(models)*3*len(seeds)}')

    X, y, groups = load_data_with_groups(args.class_name, img_size=(160, 160))
    print(f'  데이터 {len(X):,}  normal={int((y==0).sum()):,}  abnormal={int((y==1).sum()):,}')

    for seed in seeds:
        for variant in variants:
            run_one(variant, seed, args.class_name, X, y, groups,
                    models, args.epochs, args.patience)

    print('\n[전체 완료] 집계: python aggregate_loo.py')


if __name__ == '__main__':
    main()
