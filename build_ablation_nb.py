import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# OOD 개선 Ablation 실험 — Colab L4/T4\n",
    "\n",
    "**목적**: pretrained 백본 + 증강 + Mixup의 OOD F1 기여도 측정\n",
    "\n",
    "```\n",
    "설정 A (baseline) = v25 결과 재사용  OOD F1 ≈ 0.40\n",
    "설정 B = 사전학습(ImageNet)           OOD F1 = ?\n",
    "설정 C = 사전학습 + Online 증강       OOD F1 = ?\n",
    "설정 D = 사전학습 + 증강 + Mixup      OOD F1 = ?\n",
    "```\n",
    "\n",
    "**실행 전**: 런타임 → T4 또는 L4 GPU 선택.\n",
    "Drive 저장으로 끊겨도 이어학습 가능. 예상 시간: ~15-24h"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: GPU + 코드 클론\n",
    "import torch\n",
    "assert torch.cuda.is_available(), 'GPU 런타임 설정 필요'\n",
    "print('GPU:', torch.cuda.get_device_name(0))\n",
    "!git clone https://github.com/yuntaewon812/fireimage_detection.git /content/fireimage_detection\n",
    "%cd /content/fireimage_detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Drive 연결 (가중치·결과 영구저장)\n",
    "from google.colab import drive\n",
    "import os, shutil\n",
    "drive.mount('/content/drive')\n",
    "CKPT = '/content/drive/MyDrive/fireimage_ablation'\n",
    "for sub in ['model_save', 'results']:\n",
    "    os.makedirs(f'{CKPT}/{sub}', exist_ok=True)\n",
    "    link = f'/content/fireimage_detection/{sub}'\n",
    "    if os.path.islink(link): os.unlink(link)\n",
    "    elif os.path.exists(link): shutil.rmtree(link, ignore_errors=True)\n",
    "    os.symlink(f'{CKPT}/{sub}', link)\n",
    "import glob\n",
    "done = glob.glob(f'{CKPT}/model_save/**/*.pt', recursive=True)\n",
    "print(f'기존 완료 가중치: {len(done)}개')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: 패키지\n",
    "!pip install timm einops transformers kaggle -q\n",
    "print('완료')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: Kaggle 인증\n",
    "import os, getpass, re\n",
    "raw = getpass.getpass('Kaggle API Token (KGAT_...): ')\n",
    "token = re.sub(r'[^A-Za-z0-9_\\-]', '', raw)\n",
    "print('토큰 길이:', len(token))\n",
    "os.environ['KAGGLE_API_TOKEN'] = token\n",
    "os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)\n",
    "open(os.path.expanduser('~/.kaggle/access_token'), 'w').write(token)\n",
    "os.chmod(os.path.expanduser('~/.kaggle/access_token'), 0o600)\n",
    "!kaggle datasets list --user yuntarwon"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: 데이터 다운로드\n",
    "import os, glob, zipfile\n",
    "BASE = '/content/fireimage_detection/data/fireimage'\n",
    "def dl(ds, dst):\n",
    "    os.makedirs(dst, exist_ok=True)\n",
    "    os.system(f'kaggle datasets download {ds} -p {dst} --unzip')\n",
    "    for _ in range(2):\n",
    "        zs = glob.glob(f'{dst}/**/*.zip', recursive=True)\n",
    "        if not zs: break\n",
    "        for z in zs:\n",
    "            try:\n",
    "                with zipfile.ZipFile(z) as zf: zf.extractall(dst)\n",
    "                os.remove(z)\n",
    "            except: pass\n",
    "dl('yuntarwon/fireimage-abnormal',         f'{BASE}/abnormal')\n",
    "dl('yuntarwon/fireimage-abnormal-youtube', f'{BASE}/abnormal/youtube')\n",
    "dl('yuntarwon/fireimage-normal',           f'{BASE}/normal')\n",
    "imgs = ('.jpg', '.jpeg', '.png', '.bmp')\n",
    "n = sum(1 for f in glob.glob(f'{BASE}/normal/**/*',  recursive=True) if f.lower().endswith(imgs))\n",
    "a = sum(1 for f in glob.glob(f'{BASE}/abnormal/**/*', recursive=True) if f.lower().endswith(imgs))\n",
    "print(f'normal {n:,} / abnormal {a:,}')\n",
    "assert n > 0 and a > 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: Ablation B->C->D 순차 실행 (끊겨도 Drive 덕에 이어학습)\n",
    "%cd /content/fireimage_detection\n",
    "!python main_ablation.py --class_name fireimage --setting all"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 7: OOD F1 비교표\n",
    "import pandas as pd, os\n",
    "CSVS = {\n",
    "    'A(baseline)': 'results/fireimage/metrics.csv',\n",
    "    'B(pretrained)': 'results/fireimage_abl_B/metrics.csv',\n",
    "    'C(+augment)': 'results/fireimage_abl_C/metrics.csv',\n",
    "    'D(+mixup)': 'results/fireimage_abl_D/metrics.csv',\n",
    "}\n",
    "dfs = {k: pd.read_csv(v) for k, v in CSVS.items() if os.path.exists(v)}\n",
    "models = ['Resnet50','DenseNet121','efficientnetv2','efficientnetv2_proposal',\n",
    "          'nextvit','maxvit','internimage']\n",
    "rows = []\n",
    "for m in models:\n",
    "    row = {'model': m}\n",
    "    for k, df in dfs.items():\n",
    "        vals = []\n",
    "        for i in range(3):\n",
    "            r = df[df['model name'] == f'{m}_{i}']\n",
    "            if not r.empty:\n",
    "                vals.append(float(r['F1 score'].iloc[0].split('(')[0]))\n",
    "        row[f'{k}_OOD'] = f'{min(vals):.3f}' if vals else '-'\n",
    "    rows.append(row)\n",
    "comp = pd.DataFrame(rows).set_index('model')\n",
    "print('=== OOD fold F1 (작을수록 취약) ===')\n",
    "print(comp.to_string())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 8: 결과 다운로드 (가중치는 Drive에)\n",
    "import shutil\n",
    "shutil.make_archive('/content/ablation_results', 'zip',\n",
    "                    '/content/fireimage_detection', 'results')\n",
    "from google.colab import files\n",
    "files.download('/content/ablation_results.zip')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

with open(r'C:\fireimage_detection\demo\colab_ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("노트북 생성 완료")
