"""
main_distribute.py — distributed 분할 실험 (fold3)
=================================================
main_v2.py(grouped, OOD) 대비 차이:

  - load_data_distribute: YouTube 프레임을 영상별로 듬성 샘플링(중복 제거)한 뒤
    모든 샘플을 독립적으로 사용 (영상을 한 fold에 묶지 않음)
  - StratifiedKFold(shuffle=True): YouTube 프레임이 train/test에 무작위 분산
    → 특정 fold만 OOD로 붕괴하는 문제 없음
  - 2개 영상(의성 9UONERkDWq0 + eEP8a2u5PbA) 소스로 장면 다양성 확보

  결과 저장: model_save/fireimage_dist/, results/fireimage_dist/
            (기존 grouped 결과 fireimage/ 와 분리)

사용법:
  python main_distribute.py --class_name fireimage
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

from models.deep_model import DeepModel
from models.nextvit import NextViTForImageClassification
from models.efficientnetv2 import (EfficientNetV2ForImageClassification,
                                    EfficientNetV2ForImageClassification_v2)
from models.maxvit import MaxViTForImageClassification
from models.internimage import InternImageForImageClassification

os.makedirs('./model_save', exist_ok=True)
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('=' * 50)
print(f'  main_distribute.py (fold3/distributed) 시작: {now:%Y-%m-%d %H:%M:%S}')
print('=' * 50)

args = get_args()
class_name = args.class_name
SAVE_NAME = f'{class_name}_dist'      # 결과 분리 저장용 이름

# ─────────────────────────────────────────────────────
# 데이터 로드 (YouTube 솎기 + 독립 샘플)
# ─────────────────────────────────────────────────────
print('\n[1/3] distributed 데이터 로드 중 (YouTube 영상별 솎기)...')
X, y = load_data_distribute(class_name, img_size=(160, 160), target_per_video=400)

n_normal, n_abnormal = int((y == 0).sum()), int((y == 1).sum())
print(f'  총 샘플: {len(X):,}  |  normal={n_normal:,} / abnormal={n_abnormal:,}')
assert n_normal > 0 and n_abnormal > 0, '데이터 검증 실패'

PER_MODEL_LR = {
    'Resnet50': 1e-4, 'DenseNet121': 1e-4,
    'efficientnetv2': 3e-4, 'efficientnetv2_proposal': 3e-4,
    'nextvit': 3e-4, 'maxvit': 3e-4, 'internimage': 5e-5,
}
FORCE_RETRAIN = set()   # 중간 끊김 시 resume 가능 (기존 가중치 있으면 평가만)

# ─────────────────────────────────────────────────────
# StratifiedKFold (shuffle) — YouTube가 train/test에 무작위 분산
# ─────────────────────────────────────────────────────
print('\n[2/3] StratifiedKFold(shuffle=True) 분할...')
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1004)

for fold_idx, (train_index, test_index) in enumerate(skf.split(X.cpu().numpy(), y.cpu().numpy())):
    print(f'\n  Fold {fold_idx}: train={len(train_index):,} / test={len(test_index):,}')
    print(f'  device: {"cuda" if torch.cuda.is_available() else "cpu"}')

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
        'Resnet50':                DeepModel('ResNet50'),
        'DenseNet121':             DeepModel('DenseNet121'),
        'efficientnetv2':          EfficientNetV2ForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='s'),
        'efficientnetv2_proposal': EfficientNetV2ForImageClassification_v2(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='s'),
        'nextvit':                 NextViTForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='small'),
        'maxvit':                  MaxViTForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='tiny'),
        'internimage':             InternImageForImageClassification(
                                       num_labels=2, img_size=160, patch_size=16,
                                       hidden_dim=512, model_variant='tiny'),
    }

    train_full_loop(
        train_loader, val_loader, test_loader,
        model_dicts, SAVE_NAME, fold_idx,
        lr=3e-4, epochs=30, patience=8,
        per_model_lr=PER_MODEL_LR, force_retrain=FORCE_RETRAIN,
    )

print('\n[3/3] distributed 학습 완료.')
print(f'  결과: results/{SAVE_NAME}/metrics.csv  |  가중치: model_save/{SAVE_NAME}/')
