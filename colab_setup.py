"""
colab_setup.py — Colab 셋업 (Drive 마운트는 노트북에서 먼저 해야 함)
실행 전제: /content/drive 마운트됨 + 이 repo가 /content/fireimage_detection 에 클론됨
하는 일: 결과 Drive 영구저장 심볼릭 + fireimage_clean.zip 압축해제
"""
import os, glob, zipfile, shutil

REPO = '/content/fireimage_detection'
CKPT = '/content/drive/MyDrive/fireimage_ablation'
ZIP  = '/content/drive/MyDrive/fireimage_clean.zip'
BASE = f'{REPO}/data/fireimage'

# 1. 가중치/결과 Drive 영구저장 (끊겨도 이어학습)
for sub in ['model_save', 'results']:
    os.makedirs(f'{CKPT}/{sub}', exist_ok=True)
    link = f'{REPO}/{sub}'
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.exists(link):
        shutil.rmtree(link, ignore_errors=True)
    os.symlink(f'{CKPT}/{sub}', link)
print('[1/2] 결과 Drive 영구저장 설정 완료')

# 2. 데이터 압축해제 — zip 위치 자동 탐색 (Drive / 업로드 / 어디든)
candidates = [
    '/content/drive/MyDrive/fireimage_clean.zip',
    '/content/fireimage_clean.zip',
    f'{REPO}/fireimage_clean.zip',
]
candidates += glob.glob('/content/**/fireimage_clean.zip', recursive=True)
ZIP = next((z for z in candidates if os.path.exists(z)), None)
if ZIP is None:
    raise SystemExit('[오류] fireimage_clean.zip 못 찾음\n'
                     '  방법1) files.upload()로 PC에서 직접 업로드\n'
                     '  방법2) Drive(MyDrive)에 업로드 (계정 일치 필요)')
print(f'  zip 발견: {ZIP}')
if os.path.exists(BASE):
    shutil.rmtree(BASE)
os.makedirs(BASE, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(BASE)

IMG = ('.jpg', '.jpeg', '.png', '.bmp')
n = len(glob.glob(f'{BASE}/normal/**/*', recursive=True))
a = len(glob.glob(f'{BASE}/abnormal/**/*', recursive=True))
print(f'[2/2] 데이터 압축해제 완료 — normal {n} / abnormal {a}')
assert n > 0 and a > 0, '데이터가 비어있음'
print('\n✅ 셋업 완료. 이제 학습 셀 실행하세요.')
