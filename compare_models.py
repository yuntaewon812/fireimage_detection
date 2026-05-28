import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METRICS_CSV = './results/fireimage/metrics.csv'
OUT_DIR = './results/fireimage'

METRIC_COLS = ['Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'ECE']
# Higher is better for all except Loss and ECE
HIGHER_IS_BETTER = {'Accuracy': True, 'Loss': False, 'Recall': True,
                    'Precision': True, 'F1 score': True, 'AUROC': True, 'ECE': False}

PRIMARY_METRIC = 'F1 score'   # 최종 순위 기준 지표


def parse_ci(value: str):
    """'mean(lower-upper)' 형식에서 (mean, lower, upper) 추출"""
    m = re.match(r'([\d.]+)\(([\d.]+)-([\d.]+)\)', str(value).strip())
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    try:
        v = float(value)
        return v, v, v
    except Exception:
        return np.nan, np.nan, np.nan


def extract_model_name(row_name: str) -> str:
    """'ModelName_foldN' → 'ModelName'"""
    return re.sub(r'_\d+$', '', row_name)


def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['model'] = df['model name'].apply(extract_model_name)

    # ECE가 없는 기존 CSV와의 하위 호환
    if 'ECE' not in df.columns:
        df['ECE'] = np.nan

    for col in METRIC_COLS:
        df[[f'{col}_mean', f'{col}_lower', f'{col}_upper']] = df[col].apply(
            lambda x: pd.Series(parse_ci(x))
        )
    return df


def aggregate_folds(df: pd.DataFrame) -> pd.DataFrame:
    """fold 평균 집계"""
    agg_cols = {f'{c}_mean': 'mean' for c in METRIC_COLS}
    agg_cols.update({f'{c}_lower': 'mean' for c in METRIC_COLS})
    agg_cols.update({f'{c}_upper': 'mean' for c in METRIC_COLS})
    agg_cols['model name'] = 'count'

    grouped = df.groupby('model').agg(agg_cols).rename(columns={'model name': 'n_folds'})
    return grouped.reset_index()


def rank_models(agg: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for _, row in agg.iterrows():
        scores.append(row[f'{PRIMARY_METRIC}_mean'])
    agg = agg.copy()
    agg['rank_score'] = scores
    agg = agg.sort_values('rank_score', ascending=not HIGHER_IS_BETTER[PRIMARY_METRIC])
    agg['rank'] = range(1, len(agg) + 1)
    return agg


def print_summary(ranked: pd.DataFrame):
    print('\n' + '=' * 80)
    print(f'  Model Comparison  (기준 지표: {PRIMARY_METRIC})')
    print('=' * 80)
    header = f"{'Rank':<5} {'Model':<30} {'Acc':>7} {'F1':>7} {'AUROC':>7} {'Loss':>7} {'ECE':>7}"
    print(header)
    print('-' * 80)
    for _, row in ranked.iterrows():
        marker = '  ** TOP' if row['rank'] <= 2 else ''
        ece_val = row.get('ECE_mean', np.nan)
        ece_str = f'{ece_val:>7.4f}' if not np.isnan(ece_val) else '    N/A'
        print(f"{int(row['rank']):<5} {row['model']:<30} "
              f"{row['Accuracy_mean']:>7.4f} "
              f"{row['F1 score_mean']:>7.4f} "
              f"{row['AUROC_mean']:>7.4f} "
              f"{row['Loss_mean']:>7.4f} "
              f"{ece_str}"
              f"{marker}")
    print('=' * 80)

    top2 = ranked[ranked['rank'] <= 2]['model'].tolist()
    print(f'\n  [TOP2] {top2[0]}, {top2[1]}')
    print()


def plot_comparison(ranked: pd.DataFrame, df_raw: pd.DataFrame):
    plot_metrics = ['Accuracy', 'F1 score', 'AUROC', 'Recall', 'Precision']
    models = ranked['model'].tolist()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
              '#00BCD4', '#795548', '#607D8B']
    top2 = ranked[ranked['rank'] <= 2]['model'].tolist()

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 6))
    fig.suptitle('Model Performance Comparison (3-Fold CV Mean ± CI)', fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, plot_metrics):
        for i, model in enumerate(models):
            row = ranked[ranked['model'] == model].iloc[0]
            mean = row[f'{metric}_mean']
            lower = row[f'{metric}_lower']
            upper = row[f'{metric}_upper']
            yerr = [[mean - lower], [upper - mean]]

            color = colors[i % len(colors)]
            edge = 'gold' if model in top2 else color
            lw = 3 if model in top2 else 1

            ax.bar(i, mean, color=color, edgecolor=edge, linewidth=lw,
                   yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5})

        ax.set_title(metric, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Score')
        ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')

    patches = [mpatches.Patch(color=colors[i % len(colors)], label=m)
               for i, m in enumerate(models)]
    fig.legend(handles=patches, loc='lower center', ncol=len(models),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUT_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {out_path}')


def plot_radar(ranked: pd.DataFrame):
    radar_metrics = ['Accuracy', 'F1 score', 'AUROC', 'Recall', 'Precision']
    models = ranked['model'].tolist()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
              '#00BCD4', '#795548', '#607D8B']
    top2 = ranked[ranked['rank'] <= 2]['model'].tolist()

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title('Radar Chart — Model Comparison', fontsize=13, fontweight='bold', pad=20)

    for i, model in enumerate(models):
        row = ranked[ranked['model'] == model].iloc[0]
        values = [row[f'{m}_mean'] for m in radar_metrics]
        values += values[:1]
        lw = 3 if model in top2 else 1.5
        ls = '-' if model in top2 else '--'
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=lw,
                linestyle=ls, label=model)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.05)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

    out_path = os.path.join(OUT_DIR, 'model_radar.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  레이더 차트 저장: {out_path}')


def save_summary_csv(ranked: pd.DataFrame):
    cols = ['rank', 'model', 'n_folds'] + [f'{c}_mean' for c in METRIC_COLS]
    out = ranked[cols].copy()
    out.columns = ['Rank', 'Model', 'Folds'] + [f'{c} (mean)' for c in METRIC_COLS]
    path = os.path.join(OUT_DIR, 'model_comparison_summary.csv')
    out.to_csv(path, index=False)
    print(f'  요약 CSV 저장: {path}')
    return out


if __name__ == '__main__':
    df = load_metrics(METRICS_CSV)
    agg = aggregate_folds(df)
    ranked = rank_models(agg)

    print_summary(ranked)
    save_summary_csv(ranked)
    plot_comparison(ranked, df)
    plot_radar(ranked)

    top2 = ranked[ranked['rank'] <= 2]['model'].tolist()
    print(f'\n최종 추천 Top-2: {top2}')
