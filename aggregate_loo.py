"""
aggregate_loo.py — Leave-One-Out Ablation 결과 집계 + 통계 검정
==============================================================
results/{class}_loo_{variant}_s{seed}/metrics.csv 들을 모아:
  - variant별 OOD-F1 / 평균-F1 의 mean±std (시드×fold 표본)
  - full 대비 각 LOO variant 의 차이(ΔF1)
  - paired t-test, Wilcoxon signed-rank (p-value)
  - 효과크기 Cohen's d
출력: results_LOO/loo_summary.csv  +  콘솔 표
"""
import os, csv, glob, re
import numpy as np
from scipy import stats

CLASS = 'fireimage'
MODELS = ['efficientnetv2', 'maxvit']
VARIANTS = ['full', 'no_pretrained', 'no_augment', 'no_mixup', 'no_strongaug']
OUT_DIR = './results_LOO'
os.makedirs(OUT_DIR, exist_ok=True)


def parse_f1(cell):
    return float(cell.split('(')[0])


def load_metrics(path):
    """return {model_name_fold: f1}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['model name']] = parse_f1(r['F1 score'])
    return out


def collect(model, variant):
    """모든 시드의 (fold별 F1) 수집 → {'ood':[...], 'mean':[...]} (표본=시드×집계단위)"""
    pattern = f'./results/{CLASS}_loo_{variant}_s*/metrics.csv'
    ood_samples, mean_samples = [], []
    per_fold = {0: [], 1: [], 2: []}  # fold별 F1 (시드 표본)
    for path in sorted(glob.glob(pattern)):
        m = load_metrics(path)
        folds = [m.get(f'{model}_{i}') for i in range(3)]
        folds = [v for v in folds if v is not None]
        if len(folds) < 3:
            continue
        ood_samples.append(min(folds))          # OOD fold = 최저 F1
        mean_samples.append(float(np.mean(folds)))
        for i in range(3):
            per_fold[i].append(folds[i])
    return ood_samples, mean_samples, per_fold


def collect_xai(model, variant):
    """variant의 모든 시드/fold GradCAM Sensitivity·Stability 수집"""
    pattern = f'./results/{CLASS}_loo_{variant}_s*/xai_sens_stab.csv'
    sens, stab = [], []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['model'] != model:
                    continue
                try:
                    sens.append(float(r['sensitivity']))
                    stab.append(float(r['stability']))
                except ValueError:
                    pass
    return sens, stab


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp > 1e-12 else float('nan')


def fmt(xs):
    return f'{np.mean(xs):.3f}±{np.std(xs):.3f}' if xs else '  -  '


def main():
    rows = []
    for model in MODELS:
        full_ood, full_mean, full_pf = collect(model, 'full')
        if not full_ood:
            print(f'[{model}] full 결과 없음 — 학습 먼저 실행')
            continue

        print(f'\n{"="*78}\n  {model}\n{"="*78}')
        print(f'{"variant":<15}{"OOD-F1(m±s)":<16}{"평균F1(m±s)":<16}'
              f'{"ΔOOD":<9}{"t-test p":<11}{"Wilcoxon p":<12}{"Cohen d":<9}'
              f'{"GC-Sens":<10}{"GC-Stab":<10}')
        print('-'*98)

        for variant in VARIANTS:
            ood, mean, pf = collect(model, variant)
            if not ood:
                print(f'{variant:<15}(결과 없음)')
                continue

            if variant == 'full':
                d_ood = 0.0; p_t = p_w = float('nan'); d = 0.0
            else:
                # full 대비 fold별 paired 비교 (같은 fold끼리 시드 평균을 페어로)
                full_vec, var_vec = [], []
                for i in range(3):
                    n = min(len(full_pf[i]), len(pf[i]))
                    full_vec += full_pf[i][:n]
                    var_vec  += pf[i][:n]
                full_vec, var_vec = np.array(full_vec), np.array(var_vec)
                d_ood = np.mean(ood) - np.mean(full_ood)
                try:
                    p_t = stats.ttest_rel(var_vec, full_vec).pvalue
                except Exception:
                    p_t = float('nan')
                try:
                    if np.allclose(var_vec, full_vec):
                        p_w = float('nan')
                    else:
                        p_w = stats.wilcoxon(var_vec, full_vec).pvalue
                except Exception:
                    p_w = float('nan')
                d = cohens_d(var_vec, full_vec)

            star = ''
            if not np.isnan(p_t):
                star = '***' if p_t < 0.001 else '**' if p_t < 0.01 else '*' if p_t < 0.05 else ''

            xs, xst = collect_xai(model, variant)
            gc_sens = f'{np.mean(xs):.3f}' if xs else '  -  '
            gc_stab = f'{np.mean(xst):.3f}' if xst else '  -  '

            print(f'{variant:<15}{fmt(ood):<16}{fmt(mean):<16}'
                  f'{d_ood:+.3f}   {p_t:<11.4f}{p_w:<12.4f}{d:<9.2f}{star:<5}'
                  f'{gc_sens:<10}{gc_stab:<10}')

            rows.append(dict(
                model=model, variant=variant,
                ood_f1_mean=round(float(np.mean(ood)), 4),
                ood_f1_std=round(float(np.std(ood)), 4),
                mean_f1_mean=round(float(np.mean(mean)), 4),
                mean_f1_std=round(float(np.std(mean)), 4),
                delta_ood=round(float(d_ood), 4),
                ttest_p=round(float(p_t), 5) if not np.isnan(p_t) else '',
                wilcoxon_p=round(float(p_w), 5) if not np.isnan(p_w) else '',
                cohens_d=round(float(d), 3) if not np.isnan(d) else '',
                gradcam_sens=round(float(np.mean(xs)), 4) if xs else '',
                gradcam_stab=round(float(np.mean(xst)), 4) if xst else '',
                n_seeds=len(ood),
            ))

    out_csv = os.path.join(OUT_DIR, 'loo_summary.csv')
    if rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'\n저장 → {out_csv}')
        print('\n해석: ΔOOD가 음수이고 p<0.05(*)면 "그 요소를 빼면 OOD-F1이 유의하게 하락"'
              ' = 해당 기법이 통계적으로 유의미하게 기여.')
    else:
        print('\n집계할 결과 없음 — main_ablation_loo.py 먼저 실행')


if __name__ == '__main__':
    main()
