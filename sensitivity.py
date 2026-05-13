import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zoneinfo import ZoneInfo
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             balanced_accuracy_score)

from utils.utils import get_args, load_data
from utils.train_loop import evaluate_model

SEED = 1004
TOP2_MODELS = ["efficientnetv2", "nextvit"]
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]


def setSeed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setSeed()

os.makedirs("./model_save", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print("=" * 30)
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 30)

args = get_args()
class_name = args.class_name
print(f"class: {class_name}")

X, y = load_data(class_name)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
test_sets = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_test = X[test_index]
    y_test = y[test_index]
    test_sets.append((X_test, y_test))

X_test1, y_test1 = test_sets[0]
X_test2, y_test2 = test_sets[1]
X_test3, y_test3 = test_sets[2]


def build_model(model_name):
    name = model_name.lower()
    if "nextvit" in name:
        from models.nextvit import NextViTForImageClassification
        return NextViTForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='small')
    elif "efficient" in name and "proposal" not in name:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification
        return EfficientNetV2ForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    elif "efficient" in name and "proposal" in name:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        return EfficientNetV2ForImageClassification_v2(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    elif "mamba" in name:
        from models.mambavision import MambaVisionForImageClassification_v2
        return MambaVisionForImageClassification_v2(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='tiny')
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def make_noise_loaders(X_test, y_test, noise_levels=NOISE_LEVELS):
    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test)

    dev = X_test.device
    dtype = X_test.dtype
    loaders = []
    for nl in noise_levels:
        noise = torch.randn_like(X_test, dtype=dtype, device=dev) * nl
        X_noisy = torch.clamp(X_test + noise, 0, 1)
        ds = TensorDataset(X_noisy, y_test)
        loaders.append(DataLoader(ds, batch_size=32, shuffle=False))
    return loaders


def sensitivity_test(loaders, model, noise_levels=NOISE_LEVELS, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    results = []

    for nl, test_loader in zip(noise_levels, loaders):
        test_loss, test_acc, y_true, y_pred, y_prob, _ = evaluate_model(
            model, test_loader, criterion, device)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_prob).flatten()

        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        ba = balanced_accuracy_score(y_true, y_pred)

        try:
            auroc = roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"  [noise={nl}] AUROC error: {e}")
            auroc = np.nan

        try:
            aupr = average_precision_score(y_true, y_score)
        except Exception as e:
            print(f"  [noise={nl}] AUPR error: {e}")
            aupr = np.nan

        results.append({
            "NoiseLevel": float(nl),
            "Acc": test_acc,
            "BA": float(ba),
            "Loss": float(test_loss),
            "AUROC": float(auroc) if not np.isnan(auroc) else np.nan,
            "AUPR": float(aupr) if not np.isnan(aupr) else np.nan,
            "F1 Score": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
        })

    return pd.DataFrame(results)


save_root = "./results_sensitivity"
os.makedirs(save_root, exist_ok=True)

for model_name in TOP2_MODELS:
    print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")

    model1 = build_model(model_name).to(device)
    model2 = build_model(model_name).to(device)
    model3 = build_model(model_name).to(device)

    model1.load_state_dict(torch.load(
        f'./model_save/{class_name}/fold0/{model_name}.pt', map_location=device, weights_only=False))
    model2.load_state_dict(torch.load(
        f'./model_save/{class_name}/fold1/{model_name}.pt', map_location=device, weights_only=False))
    model3.load_state_dict(torch.load(
        f'./model_save/{class_name}/fold2/{model_name}.pt', map_location=device, weights_only=False))

    loaders1 = make_noise_loaders(X_test1, y_test1)
    loaders2 = make_noise_loaders(X_test2, y_test2)
    loaders3 = make_noise_loaders(X_test3, y_test3)

    df1 = sensitivity_test(loaders1, model1, device=device)
    df2 = sensitivity_test(loaders2, model2, device=device)
    df3 = sensitivity_test(loaders3, model3, device=device)

    df_all = pd.concat(
        [df1.assign(Fold=0), df2.assign(Fold=1), df3.assign(Fold=2)],
        ignore_index=True
    )
    df_all.insert(0, "Class", class_name)
    df_all.insert(1, "Model", model_name)

    numeric_cols = ["Acc", "BA", "Loss", "AUROC", "AUPR", "F1 Score", "Precision", "Recall"]
    cv_mean = df_all.groupby("NoiseLevel")[numeric_cols].mean().reset_index()
    cv_mean.insert(0, "Class", class_name)
    cv_mean.insert(1, "Model", model_name)
    cv_mean["Fold"] = "CV_MEAN"
    df_all = pd.concat([df_all, cv_mean], ignore_index=True)

    out_path = os.path.join(save_root, f"{class_name}_{model_name}.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_path}")

    print(f"Fold 0:\n{df1}\nFold 1:\n{df2}\nFold 2:\n{df3}")
