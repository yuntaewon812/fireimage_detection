"""
colab_results.py — 학습 결과 요약 표시
설정 E의 fold별 F1 + OOD(최저 fold) F1을 모델별로 출력.
"""
import os, csv

CSV = '/content/fireimage_detection/results/fireimage_abl_E/metrics.csv'
MODELS = ['Resnet50', 'DenseNet121', 'efficientnetv2', 'efficientnetv2_proposal',
          'nextvit', 'maxvit', 'internimage']

if not os.path.exists(CSV):
    raise SystemExit(f'결과 없음: {CSV}\n  학습 셀이 끝나야 생성됩니다.')

rows = {}
with open(CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        try:
            rows[r['model name']] = float(r['F1 score'].split('(')[0])
        except Exception:
            pass

print('=' * 60)
print(f'{"모델":<24} {"fold0":>7} {"fold1":>7} {"fold2":>7} {"OOD(최저)":>10}')
print('-' * 60)
ood_vals = []
for m in MODELS:
    f = [rows.get(f'{m}_{i}') for i in range(3)]
    cells = [f'{v:.3f}' if v is not None else '  -  ' for v in f]
    valid = [v for v in f if v is not None]
    ood = min(valid) if valid else None
    if ood is not None:
        ood_vals.append(ood)
    ood_s = f'{ood:.3f}' if ood is not None else '  -  '
    print(f'{m:<24} {cells[0]:>7} {cells[1]:>7} {cells[2]:>7} {ood_s:>10}')
print('-' * 60)
if ood_vals:
    print(f'{"7모델 OOD 평균":<24} {"":>7} {"":>7} {"":>7} {sum(ood_vals)/len(ood_vals):>10.3f}')
print('=' * 60)
print('\n비교: 옛 데이터(영상 2개) OOD ≈ 0.40 → 재설계 데이터(영상 30개) 위 수치')
