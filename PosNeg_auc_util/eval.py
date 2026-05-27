import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from .auc import perturbation_hit_curve
from utils.utils import load_images_at_indices


def run_xai_pos_neg_auc_cv(
    *,
    cfg,
    load_data_fn,
    model_name: str,
    model_loader,
    methods,
    device,
    print_errors: bool = True,
    out_suffix: str = "xai_pos_neg_auc.csv",
    write_cv_mean: bool = True,
    write_all: bool = True,
):
    os.makedirs(cfg.out_dir, exist_ok=True)

    erase_ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    for class_name in cfg.class_names:
        print(f"=== Processing Class: {class_name} ===")

        all_paths, y_array = load_data_fn(class_name)
        if len(all_paths) == 0:
            print("  No samples found.\n")
            continue

        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

        fold_results = {
            fold: {
                m: {
                    "pos_auc": [],
                    "neg_auc": [],
                    "pos_curve": [],
                    "neg_curve": [],
                } for m in methods.keys()
            }
            for fold in cfg.folds
        }

        for i, (_, test_index) in enumerate(skf.split(all_paths, y_array)):
            if i not in cfg.folds:
                continue

            test_paths = [all_paths[j] for j in test_index]
            y_test_all = y_array[test_index]
            rng = np.random.default_rng(cfg.seed + i)
            n_eval = min(100, len(test_paths))
            eval_local = rng.choice(len(test_paths), n_eval, replace=False)
            if len(eval_local) == 0:
                print(f"  Fold {i}: no test samples")
                continue

            X_test, valid = load_images_at_indices(test_paths, eval_local, (cfg.img_size, cfg.img_size))
            if len(valid) == 0:
                print(f"  Fold {i}: no images loaded")
                continue
            y_test = torch.tensor(y_test_all[eval_local][valid], dtype=torch.long)

            model_path = os.path.join(cfg.base_model_path, class_name, f"fold{i}", f"{model_name}.pt")
            if not os.path.exists(model_path):
                print(f"  Fold {i}: model not found -> {model_path}")
                continue

            model = model_loader(model_name, model_path, cfg.img_size, device)

            for local_idx in tqdm(range(len(X_test)), desc=f"{class_name} fold{i}", leave=False):
                x = X_test[local_idx].unsqueeze(0).to(device)
                y_true = int(y_test[local_idx].item())

                for method_name, map_fn in methods.items():
                    try:
                        exp_map = map_fn(model, x)
                        _, pos_curve, pos_auc = perturbation_hit_curve(
                            model=model, img_tensor=x, explanation_map=exp_map,
                            target_class=y_true, erase_ratios=erase_ratios, neg=False,
                        )
                        _, neg_curve, neg_auc = perturbation_hit_curve(
                            model=model, img_tensor=x, explanation_map=exp_map,
                            target_class=y_true, erase_ratios=erase_ratios, neg=True,
                        )
                        fold_results[i][method_name]["pos_auc"].append(pos_auc)
                        fold_results[i][method_name]["neg_auc"].append(neg_auc)
                        fold_results[i][method_name]["pos_curve"].append(pos_curve)
                        fold_results[i][method_name]["neg_curve"].append(neg_curve)
                    except Exception as e:
                        if print_errors:
                            print(f"[{method_name} ERROR] {class_name} fold{i} {type(e).__name__} {e}")

            counts = "  ".join(f"{m}={len(fold_results[i][m]['pos_auc'])}" for m in methods.keys())
            print(f"[COUNT] {class_name} fold{i}  {counts}")

        out_csv = os.path.join(cfg.out_dir, f"{class_name}_{out_suffix}")
        all_buf = {m: {"pos_auc": [], "neg_auc": []} for m in methods.keys()}
        fold_means = {m: {"pos_auc": [], "neg_auc": []} for m in methods.keys()}

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Fold","Method","AUC_XRange","PosAUC","NegAUC","PosAUC_x100","NegAUC_x100","PosCurveMean","NegCurveMean"])
            for fold in cfg.folds:
                for method_name in methods.keys():
                    pos_list = fold_results[fold][method_name]["pos_auc"]
                    neg_list = fold_results[fold][method_name]["neg_auc"]
                    pos_curve_list = fold_results[fold][method_name]["pos_curve"]
                    neg_curve_list = fold_results[fold][method_name]["neg_curve"]
                    if len(pos_list) == 0:
                        continue
                    pos_m = float(np.mean(pos_list))
                    neg_m = float(np.mean(neg_list))
                    pos_curve_m = np.mean(np.stack(pos_curve_list, axis=0), axis=0)
                    neg_curve_m = np.mean(np.stack(neg_curve_list, axis=0), axis=0)
                    all_buf[method_name]["pos_auc"].extend(pos_list)
                    all_buf[method_name]["neg_auc"].extend(neg_list)
                    fold_means[method_name]["pos_auc"].append(pos_m)
                    fold_means[method_name]["neg_auc"].append(neg_m)
                    writer.writerow([fold, method_name, "0.1-0.9", pos_m, neg_m, pos_m*100, neg_m*100,
                                     ";".join(f"{v:.6f}" for v in pos_curve_m.tolist()),
                                     ";".join(f"{v:.6f}" for v in neg_curve_m.tolist())])
            if write_cv_mean:
                for method_name in methods.keys():
                    if not fold_means[method_name]["pos_auc"]:
                        continue
                    pm = float(np.mean(fold_means[method_name]["pos_auc"]))
                    nm = float(np.mean(fold_means[method_name]["neg_auc"]))
                    writer.writerow(["CV_MEAN", method_name, "0.1-0.9", pm, nm, pm*100, nm*100, "", ""])
            if write_all:
                for method_name in methods.keys():
                    pl = all_buf[method_name]["pos_auc"]
                    nl = all_buf[method_name]["neg_auc"]
                    if not pl:
                        continue
                    writer.writerow(["ALL", method_name, "0.1-0.9", float(np.mean(pl)), float(np.mean(nl)),
                                     float(np.mean(pl))*100, float(np.mean(nl))*100, "", ""])
        print(f"  Saved -> {out_csv}\n")
