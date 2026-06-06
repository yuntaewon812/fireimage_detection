"""
colab_setup.py — Colab 데이터 셋업 (Drive 선택적)
하는 일: fireimage_clean.zip 자동 탐색 → 압축해제. Drive 있으면 결과 영구저장.
"""
import os, glob, zipfile, shutil

REPO = '/content/fireimage_detection'
BASE = f'{REPO}/data/fireimage'

# 1. (선택) Drive 마운트돼 있으면 결과 영구저장 심볼릭
CKPT = '/content/drive/MyDrive/fireimage_ablation'
if os.path.isdir('/content/drive/MyDrive'):
    for sub in ['model_save', 'results']:
        os.makedirs(f'{CKPT}/{sub}', exist_ok=True)
        link = f'{REPO}/{sub}'
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            shutil.rmtree(link, ignore_errors=True)
        os.symlink(f'{CKPT}/{sub}', link)
    print('[1/2] 결과 Drive 영구저장 설정 완료')
else:
    print('[1/2] Drive 미마운트 → 결과는 로컬 저장 (끊기면 재학습 필요)')

# 2. zip 자동 탐색 (업로드 / Drive / 어디든)
candidates = [
    '/content/fireimage_clean.zip',
    '/content/drive/MyDrive/fireimage_clean.zip',
    f'{REPO}/fireimage_clean.zip',
]
candidates += glob.glob('/content/**/fireimage_clean.zip', recursive=True)
ZIP = next((z for z in candidates if os.path.exists(z)), None)
if ZIP is None:
    raise SystemExit('[오류] fireimage_clean.zip 못 찾음 — 업로드 셀을 먼저 실행하세요')
print(f'  zip 발견: {ZIP}')

if os.path.exists(BASE):
    shutil.rmtree(BASE)
os.makedirs(BASE, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(BASE)

n = len(glob.glob(f'{BASE}/normal/**/*', recursive=True))
a = len(glob.glob(f'{BASE}/abnormal/**/*', recursive=True))
print(f'[2/2] 압축해제 완료 — normal {n} / abnormal {a}')
assert n > 0 and a > 0, '데이터 비어있음'
print('\n✅ 셋업 완료. 다음 셀(학습) 실행하세요.')
