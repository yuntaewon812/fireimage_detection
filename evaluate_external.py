"""
evaluate_external.py — 외부 검증셋 평가
=========================================
학습에 사용하지 않은 별도 데이터로 저장된 모델 가중치를 평가합니다.

외부 검증셋 디렉터리 구조:
  ./data/external_val/<class_name>/normal/   ← 비화재 이미지
  ./data/external_val/<class_name>/abnormal/ ← 화재 이미지

결과 저장:
  ./results/<class_name>/external_val_metrics.csv
  ./results/<class_name>/external_cm/  ← 혼동 행렬 이미지

사용법:
  python evaluate_external.py --class_name fireimage
"""

import os
import argparse
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from utils.utils import get_args
from utils.train_loop import compute_ece, evaluate_model, bootstrap_ci, bootstrap_ci_loss

from models.deep_model import DeepModel
from models.nextvit import NextViTForImageClassification
from models.efficientnetv2 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v2
from models.maxvit import MaxViTForImageClassification
from models.internimage import InternImageForImageClassification


# ─────────────────────────────────────────────────────
# 모델 정의 (main_v2.py 와 동일하게 유지)
# ─────────────────────────────────────────────────────
def build_model_dict():
    return {
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


# ─────────────────────────────────────────────────────
# 외부 검증셋 로드
# ─────────────────────────────────────────────────────
def load_external_data(class_name: str, img_size=(160, 160)):
    import cv2

    data_path = os.path.join('.', 'data', 'external_val', class_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"외부 검증셋 디렉터리가 없습니다: {data_path}\n"
            "아래 구조로 이미지를 준비해 주세요:\n"
            f"  {data_path}/normal/   ← 비화재\n"
            f"  {data_path}/abnormal/ ← 화재"
        )

    X, y = [], []
    found = {'normal': 0, 'abnormal': 0}

    for top_folder, label in [('normal', 0), ('abnormal', 1)]:
        top_path = os.path.join(data_path, top_folder)
        if not os.path.exists(top_path):
            continue
        for root, _dirs, files in os.walk(top_path):
            for file_name in files:
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                image_path = os.path.join(root, file_name)
                try:
                    arr = np.fromfile(image_path, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        X.append(img)
                        y.append(label)
                        found[top_folder] += 1
                except Exception as e:
                    print(f"  경고: {file_name} 로드 실패 — {e}")

    if not X:
        raise ValueError(f"외부 검증셋에 유효한 이미지가 없습니다: {data_path}")

    print(f"  외부 검증셋 로드 완료: normal={found['normal']} / abnormal={found['abnormal']} "
          f"/ 합계={len(X)}")

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.array(y, dtype=np.int64)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


# ─────────────────────────────────────────────────────
# 혼동 행렬 저장
# ─────────────────────────────────────────────────────
def save_cm(y_true, y_pred, model_name, class_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Normal', class_name], columns=['Normal', class_name])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 28})
    plt.title(f'External Val — {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}.png'))
    plt.close()


# ─────────────────────────────────────────────────────
# 결과 CSV 저장
# ─────────────────────────────────────────────────────
def save_external_csv(rows: list, class_name: str):
    out_dir = os.path.join('.', 'results', class_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'external_val_metrics.csv')

    header = ['model', 'Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'ECE',
              'folds_evaluated', 'timestamp']
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"\n  결과 저장: {path}")
    return df


# ─────────────────────────────────────────────────────
# 단일 모델 평가 (가용 fold 평균)
# ─────────────────────────────────────────────────────
def evaluate_one_model(model_name, model, ext_loader, class_name, device, cm_dir):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    fold_results = []
    for fold_idx in range(3):
        weight_path = os.path.join('.', 'model_save', class_name, f'fold{fold_idx}', f'{model_name}.pt')
        if not os.path.exists(weight_path):
            print(f"    Fold {fold_idx}: 가중치 없음 — 건너뜀 ({weight_path})")
            continue
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"    Fold {fold_idx}: 가중치 로드 실패 — {e}")
            continue

        loss, acc, y_true, y_pred, y_prob, all_losses = evaluate_model(
            model, ext_loader, criterion, device)
        fold_results.append((y_true, y_pred, y_prob, all_losses))
        print(f"    Fold {fold_idx}: Acc={acc:.4f}  Loss={loss:.4f}")

    if not fold_results:
        print(f"  [{model_name}] 사용 가능한 가중치 없음 — 건너뜀")
        return None

    # fold 결과 합산하여 Bootstrap CI 계산
    all_y_true  = np.concatenate([r[0] for r in fold_results])
    all_y_pred  = np.concatenate([r[1] for r in fold_results])
    all_y_prob  = np.concatenate([r[2] for r in fold_results])
    all_losses  = np.concatenate([r[3] for r in fold_results])

    acc_ci    = bootstrap_ci(balanced_accuracy_score, all_y_true, all_y_pred)
    loss_ci   = bootstrap_ci_loss(all_losses)
    recall_ci = bootstrap_ci(lambda y, yh: recall_score(y, yh, average='macro'), all_y_true, all_y_pred)
    prec_ci   = bootstrap_ci(lambda y, yh: precision_score(y, yh, average='macro'), all_y_true, all_y_pred)
    f1_ci     = bootstrap_ci(lambda y, yh: f1_score(y, yh, average='macro'), all_y_true, all_y_pred)
    try:
        auc_ci = bootstrap_ci(lambda y, p: roc_auc_score(y, p, multi_class='ovr'),
                               all_y_true, all_y_pred, probs=all_y_prob)
    except Exception:
        auc_ci = [np.nan, np.nan, np.nan]

    ece = compute_ece(all_y_true, all_y_prob)

    def ci_str(ci): return f'{ci[0]:.3f}({ci[1]:.3f}-{ci[2]:.3f})'

    print(f"  [{model_name}]  Acc={acc_ci[0]:.4f}  F1={f1_ci[0]:.4f}  "
          f"AUROC={auc_ci[0]:.4f}  ECE={ece:.4f}")

    save_cm(all_y_true, all_y_pred, model_name, class_name, cm_dir)

    now = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    return [model_name, ci_str(acc_ci), ci_str(loss_ci), ci_str(recall_ci),
            ci_str(prec_ci), ci_str(f1_ci), ci_str(auc_ci), f'{ece:.4f}',
            len(fold_results), now]


# ─────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='fireimage')
    args = parser.parse_args()
    class_name = args.class_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*55}')
    print(f'  외부 검증셋 평가  |  class: {class_name}  |  device: {device}')
    print(f'{"="*55}')

    # 외부 검증셋 로드
    print('\n[1/3] 외부 검증셋 로드...')
    try:
        X_ext, y_ext = load_external_data(class_name)
    except FileNotFoundError as e:
        print(f'\n[오류] {e}')
        return

    ext_loader = DataLoader(TensorDataset(X_ext, y_ext), batch_size=8, shuffle=False)

    cm_dir = os.path.join('.', 'results', class_name, 'external_cm')
    rows = []

    print('\n[2/3] 모델별 평가...')
    model_dict = build_model_dict()

    for model_name, model in model_dict.items():
        print(f'\n  ── {model_name}')
        row = evaluate_one_model(model_name, model, ext_loader, class_name, device, cm_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print('\n평가할 수 있는 모델이 없습니다. main_v2.py를 먼저 실행하세요.')
        return

    print('\n[3/3] 결과 저장...')
    df = save_external_csv(rows, class_name)

    print('\n' + '=' * 80)
    print(f'  외부 검증셋 최종 결과  (기준: F1 Score)')
    print('=' * 80)
    print(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'AUROC':>8} {'ECE':>8}")
    print('-' * 80)
    for _, row in df.sort_values('F1 score', ascending=False).iterrows():
        acc_m  = float(row['Accuracy'].split('(')[0])
        f1_m   = float(row['F1 score'].split('(')[0])
        auc_m  = float(row['AUROC'].split('(')[0]) if 'nan' not in row['AUROC'].lower() else float('nan')
        ece_m  = float(row['ECE'])
        print(f"  {row['model']:<28} {acc_m:>8.4f} {f1_m:>8.4f} {auc_m:>8.4f} {ece_m:>8.4f}")
    print('=' * 80)


if __name__ == '__main__':
    main()
