"""
main_nextvit_fix.py — nextvit fold0/fold1 재학습 (학습 불안정 수정)
====================================================================
distributed 실험에서 nextvit가 fold0·fold1만 acc 0.500(한 클래스 고정)으로
학습 실패 → 학습률을 3e-4 → 1e-4 로 낮춰 안정화 후 해당 fold만 재학습.

- fold2(nextvit) 및 다른 모델은 건드리지 않음
- force_retrain={'nextvit'} → 기존 불량 nextvit 가중치 덮어씀
- 결과는 model_save/fireimage_dist/, results/fireimage_dist/metrics.csv 갱신

사용법:
  python main_nextvit_fix.py --class_name fireimage
"""
import os
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.utils import load_data_distribute, get_args
from utils.train_loop import train_full_loop
from models.nextvit import NextViTForImageClassification

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('=' * 50)
print(f'  nextvit fold0/fold1 재학습 시작: {now:%Y-%m-%d %H:%M:%S}')
print('=' * 50)

args = get_args()
class_name = args.class_name
SAVE_NAME = f'{class_name}_dist'

REDO_FOLDS = {0, 1}          # 실패한 fold만
NEXTVIT_LR = 1e-4            # 3e-4 → 1e-4 안정화

print('\n[1/2] distributed 데이터 로드...')
X, y = load_data_distribute(class_name, img_size=(160, 160), target_per_video=400)
print(f'  총 {len(X):,}  | normal={int((y==0).sum()):,} / abnormal={int((y==1).sum()):,}')

print('\n[2/2] StratifiedKFold(동일 split, random_state=1004)...')
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1004)

for fold_idx, (train_index, test_index) in enumerate(skf.split(X.cpu().numpy(), y.cpu().numpy())):
    if fold_idx not in REDO_FOLDS:
        continue
    print(f'\n=== nextvit Fold {fold_idx} 재학습 (lr={NEXTVIT_LR:.0e}) ===')

    train_idx, val_idx = train_test_split(
        train_index, test_size=0.25, random_state=1004,
        stratify=y[train_index].numpy())

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_index], y[test_index]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=8, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=8, shuffle=False)

    model_dicts = {
        'nextvit': NextViTForImageClassification(
            num_labels=2, img_size=160, patch_size=16,
            hidden_dim=512, model_variant='small'),
    }

    train_full_loop(
        train_loader, val_loader, test_loader,
        model_dicts, SAVE_NAME, fold_idx,
        lr=3e-4, epochs=30, patience=8,
        per_model_lr={'nextvit': NEXTVIT_LR},
        force_retrain={'nextvit'},   # 불량 가중치 덮어쓰기
    )

print('\n완료. results/fireimage_dist/metrics.csv 의 nextvit_0 / nextvit_1 갱신됨.')
