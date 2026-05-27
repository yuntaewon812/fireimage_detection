"""
main_v2.py — 데이터 누수 수정 + internimage 재학습 + ECE 평가
=================================================================
main_proposal.py 대비 변경 사항:

  1. StratifiedKFold  →  StratifiedGroupKFold
     - YouTube 프레임은 video_id 단위로 묶어 같은 fold에 배치
     - 같은 영상의 연속 프레임이 train/test에 동시에 들어가지 않음

  2. internimage 재학습
     - 기존 불량 가중치 삭제 (force_retrain)
     - 학습률 1e-3 → 5e-5 (DCNv3 구조에 맞는 낮은 LR)
     - CosineAnnealingLR 스케줄러 자동 적용

  3. 모델별 학습률 설정
     - CNN 계열  (Resnet50, DenseNet121): 1e-4
     - Transformer 계열 (maxvit, nextvit, efficientnetv2 등): 3e-4
     - internimage: 5e-5

  4. ECE (Expected Calibration Error) 출력
     - 학습 후 각 fold 평가 시 콘솔에 출력
     - F1 + Loss + ECE 3가지를 종합해 모델 품질 판단

사용법:
  python main_v2.py --class_name fireimage
"""

import os
import datetime
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from utils.utils import load_data_with_groups, get_args
from utils.train_loop import train_full_loop

from models.deep_model import DeepModel
from models.nextvit import NextViTForImageClassification
from models.efficientnetv2 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v2
from models.maxvit import MaxViTForImageClassification
from models.internimage import InternImageForImageClassification


os.makedirs('./model_save', exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('=' * 50)
print(f'  main_v2.py 시작: {now.strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 50)

args = get_args()
class_name = args.class_name
print(f'class: {class_name}')

# ─────────────────────────────────────────────────────
# 데이터 로드 (groups 포함)
# ─────────────────────────────────────────────────────
print('\n[1/3] 데이터 로드 중...')
X, y, groups = load_data_with_groups(class_name, img_size=(160, 160))

import numpy as np
unique_groups = np.unique(groups)
youtube_groups = [g for g in unique_groups if not g.startswith('other_')]
print(f'  총 샘플: {len(X):,}  |  클래스 분포: normal={int((y==0).sum()):,} / abnormal={int((y==1).sum()):,}')
print(f'  YouTube 영상 소스: {len(youtube_groups)}개 {youtube_groups}')
print(f'  그룹 수: {len(unique_groups):,}  (YouTube 프레임은 같은 그룹으로 묶임)')

# ─────────────────────────────────────────────────────
# 모델별 학습률 설정
# ─────────────────────────────────────────────────────
PER_MODEL_LR = {
    'Resnet50':               1e-4,   # CNN — 보수적
    'DenseNet121':            1e-4,   # CNN — 보수적
    'efficientnetv2':         3e-4,   # EfficientNet
    'efficientnetv2_proposal':3e-4,   # EfficientNet variant
    'nextvit':                3e-4,   # Hybrid Transformer
    'maxvit':                 3e-4,   # Hybrid Transformer
    'internimage':            5e-5,   # DCNv3 — 매우 낮은 LR 필요
}

# internimage 와 maxvit 은 기존 불량 가중치를 삭제하고 재학습
FORCE_RETRAIN = {'internimage', 'maxvit'}

# ─────────────────────────────────────────────────────
# StratifiedGroupKFold — 3-fold
# ─────────────────────────────────────────────────────
print('\n[2/3] StratifiedGroupKFold 분할...')
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1004)

for fold_idx, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):

    # YouTube 그룹이 이번 fold의 test에 들어갔는지 확인
    test_groups = set(groups[test_index])
    yt_in_test = [g for g in test_groups if not g.startswith('other_')]
    print(f'\n  Fold {fold_idx}: train={len(train_index):,} / test={len(test_index):,}  '
          f'| YouTube in test: {yt_in_test if yt_in_test else "없음"}')

    # 학습셋의 25% 를 검증셋으로 분리
    train_idx, val_idx = train_test_split(
        train_index, test_size=0.25, random_state=1004,
        stratify=y[train_index].numpy()
    )

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
        model_dicts, class_name, fold_idx,
        lr=3e-4,
        epochs=60,
        patience=12,
        per_model_lr=PER_MODEL_LR,
        force_retrain=FORCE_RETRAIN,
    )

print('\n[3/3] 학습 완료. compare_models.py 로 결과 확인:')
print('  python compare_models.py')
