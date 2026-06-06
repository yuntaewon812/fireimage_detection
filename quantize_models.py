"""
quantize_models.py — 상위 2개 모델 INT8 Dynamic Quantization
============================================================
대상: efficientnetv2 (OOD F1 0.922), maxvit (OOD F1 0.916)
방법: torch.quantization.quantize_dynamic (Linear 레이어 INT8)
     → CPU 추론 기준 ~1.5–2x 속도, ~25% 크기 감소, F1 거의 유지

사용법:
  python quantize_models.py                    # Colab 경로 자동
  python quantize_models.py --base /path/to/fireimage_detection

출력:
  model_save/fireimage_abl_E_quant/fold{n}/{model_name}.pt
  results/quantization_report.csv
"""
import os
import sys
import time
import csv
import argparse
import zipfile
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=None,
                   help='프로젝트 루트 (기본: Colab /content/fireimage_detection)')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--class_name', default='fireimage_abl_E')
    return p.parse_args()


def setup_base(base):
    """Colab이면 자동 탐색, 아니면 지정 경로."""
    if base:
        return base
    candidates = [
        '/content/fireimage_detection',
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise SystemExit('[오류] 프로젝트 루트를 찾을 수 없습니다. --base 옵션으로 지정하세요.')


# ── 가중치 압축 해제 (Downloads/weights_E.zip이 있는 경우) ────────────────────
def maybe_extract_weights(base):
    zip_candidates = [
        os.path.join(base, 'weights_E.zip'),
        '/content/weights_E.zip',
        os.path.expanduser('~/Downloads/weights_E.zip'),
    ]
    zip_path = next((z for z in zip_candidates if os.path.exists(z)), None)
    if zip_path is None:
        return  # 이미 해제되어 있거나 없으면 pass

    model_save = os.path.join(base, 'model_save')
    if os.path.isdir(os.path.join(model_save, 'fireimage_abl_E')):
        print(f'[가중치] model_save/fireimage_abl_E 이미 존재 → 압축 해제 건너뜀')
        return

    print(f'[가중치] {zip_path} 압축 해제 중...')
    os.makedirs(model_save, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(model_save)
    print(f'  완료: {model_save}')


# ── 모델 클래스 임포트 ────────────────────────────────────────────────────────
def import_models(base):
    if base not in sys.path:
        sys.path.insert(0, base)
    from models.efficientnetv2 import EfficientNetV2ForImageClassification
    from models.maxvit import MaxViTForImageClassification
    return EfficientNetV2ForImageClassification, MaxViTForImageClassification


def build_model(name, EfficientNetV2, MaxViT):
    if name == 'efficientnetv2':
        return EfficientNetV2(num_labels=2, img_size=160, patch_size=16,
                              hidden_dim=512, model_variant='s', pretrained=False)
    elif name == 'maxvit':
        return MaxViT(num_labels=2, img_size=160, patch_size=16,
                      hidden_dim=512, model_variant='tiny', pretrained=False)
    raise ValueError(f'알 수 없는 모델: {name}')


# ── INT8 Dynamic Quantization ─────────────────────────────────────────────────
def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Linear 레이어를 INT8로 변환 (Transformer 계열에 특히 효과적)."""
    return torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear},
        dtype=torch.qint8
    )


# ── 추론 속도 측정 (CPU, warmup 포함) ─────────────────────────────────────────
def measure_latency(model, dummy_input, n_runs=50):
    model.eval()
    model.cpu()
    dummy = dummy_input.cpu()
    # warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    # 측정
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    return (time.perf_counter() - t0) / n_runs * 1000  # ms per batch


# ── F1 평가 (테스트 데이터 없으면 None 반환) ─────────────────────────────────
def evaluate_f1(model, loader):
    if loader is None:
        return None
    model.eval()
    model.cpu()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.cpu())
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    return f1_score(all_labels, all_preds, zero_division=0)


# ── 파일 크기 (MB) ────────────────────────────────────────────────────────────
def model_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / 1024 / 1024
    return None


def model_param_mb(model):
    """파라미터 메모리 (FP32 기준 MB)."""
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / 1024 / 1024


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    BASE = setup_base(args.base)
    print(f'프로젝트 루트: {BASE}')

    maybe_extract_weights(BASE)

    EfficientNetV2, MaxViT = import_models(BASE)
    TARGET_MODELS = ['efficientnetv2', 'maxvit']

    # 결과 저장
    results_dir = os.path.join(BASE, 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'quantization_report.csv')

    # 더미 입력 (batch=1, 160×160 RGB)
    dummy_input = torch.randn(1, 3, 160, 160)

    rows = []
    print('\n' + '='*65)
    print(f'  {"모델":<22} {"Fold":<5} {"원본MB":>7} {"INT8MB":>7} {"압축율":>7} '
          f'{"원본ms":>7} {"INT8ms":>7} {"속도향상":>8}')
    print('-'*65)

    for model_name in TARGET_MODELS:
        for fold in args.folds:
            weight_path = os.path.join(
                BASE, 'model_save', args.class_name, f'fold{fold}', f'{model_name}.pt')

            if not os.path.exists(weight_path):
                print(f'  [{model_name}] fold{fold} 가중치 없음: {weight_path}')
                continue

            # ── 원본 모델 로드 ─────────────────────────────────────────
            model_fn = 'efficientnetv2' if model_name == 'efficientnetv2' else 'maxvit'
            model_orig = build_model(model_fn, EfficientNetV2, MaxViT)
            state = torch.load(weight_path, map_location='cpu', weights_only=True)
            model_orig.load_state_dict(state)
            model_orig.eval()
            model_orig.cpu()

            orig_size = model_size_mb(weight_path)
            orig_lat = measure_latency(model_orig, dummy_input)

            # ── INT8 Quantization ──────────────────────────────────────
            model_quant = build_model(model_fn, EfficientNetV2, MaxViT)
            model_quant.load_state_dict(
                torch.load(weight_path, map_location='cpu', weights_only=True))
            model_quant.eval()
            model_quant = quantize_dynamic(model_quant)

            # 경량화 모델 저장
            save_dir = os.path.join(BASE, 'model_save',
                                    args.class_name + '_quant', f'fold{fold}')
            os.makedirs(save_dir, exist_ok=True)
            quant_path = os.path.join(save_dir, f'{model_name}.pt')
            torch.save(model_quant.state_dict(), quant_path)

            quant_size = model_size_mb(quant_path)
            quant_lat = measure_latency(model_quant, dummy_input)

            ratio = orig_size / quant_size if quant_size else float('nan')
            speedup = orig_lat / quant_lat if quant_lat else float('nan')

            print(f'  {model_name:<22} fold{fold}  '
                  f'{orig_size:>6.1f}  {quant_size:>6.1f}  {ratio:>6.2f}x  '
                  f'{orig_lat:>6.1f}  {quant_lat:>6.1f}  {speedup:>7.2f}x')

            rows.append({
                'model': model_name,
                'fold': fold,
                'orig_size_mb': round(orig_size, 2),
                'quant_size_mb': round(quant_size, 2),
                'size_ratio': round(ratio, 3),
                'orig_latency_ms': round(orig_lat, 2),
                'quant_latency_ms': round(quant_lat, 2),
                'speedup': round(speedup, 3),
                'quant_path': quant_path,
            })

    print('='*65)

    # CSV 저장
    if rows:
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n[결과] {report_path} 저장 완료')

        # 요약 출력
        print('\n[요약]')
        for name in TARGET_MODELS:
            subset = [r for r in rows if r['model'] == name]
            if not subset:
                continue
            avg_ratio = np.mean([r['size_ratio'] for r in subset])
            avg_speedup = np.mean([r['speedup'] for r in subset])
            print(f'  {name}: 크기 {avg_ratio:.2f}x 압축, 추론 {avg_speedup:.2f}x 가속 (CPU)')
    else:
        print('\n[경고] 경량화된 모델이 없습니다. 가중치 경로를 확인하세요.')
        print(f'  예상 경로: {BASE}/model_save/{args.class_name}/fold{{n}}/{{model}}.pt')

    print('\n[경량화 완료]')
    print('  경량화 가중치 위치: model_save/' + args.class_name + '_quant/')
    print('  Colab 다운로드: files.download("results/quantization_report.csv")')


if __name__ == '__main__':
    main()
