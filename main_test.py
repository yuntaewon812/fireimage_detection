"""
main_test.py — GPU 작동 확인용 스모크 테스트
ResNet50 1개 모델, fold0, 데이터 소량(클래스당 300장), 3 epoch만 학습.
목적: 짧은 시간 안에 GPU/CPU 여부와 epoch 속도를 확인.
"""
import os, time, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.deep_model import DeepModel

result = []
def log(m):
    print(m, flush=True)
    result.append(str(m))

log("=" * 50)
log("  GPU 스모크 테스트 (ResNet50 / 소량 / 3 epoch)")
log("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    log(f"device: cuda")
    log(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    log(f"torch {torch.__version__} | arch_list: {torch.cuda.get_arch_list()}")
else:
    log(f"device: cpu")
    log(f"torch {torch.__version__}  ← GPU 미사용!")

# ── 데이터 소량 로드 (클래스당 최대 300장) ──
img_size = (160, 160)
N = 300
base = './data/fireimage'
X, y = [], []
for label, top in [(0, 'normal'), (1, 'abnormal')]:
    files = [f for f in glob.glob(f'{base}/{top}/**/*', recursive=True)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:N]
    for fp in files:
        arr = np.fromfile(fp, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        X.append(img)
        y.append(label)

log(f"로드: {len(X)}장 (normal+abnormal subset)")
if len(X) == 0:
    log("데이터 없음! 경로 확인 필요")
    with open('./test_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    raise SystemExit(1)

X = np.transpose(np.array(X, dtype=np.float32) / 255.0, (0, 3, 1, 2))
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)
loader = DataLoader(TensorDataset(Xt, yt), batch_size=8, shuffle=True)

# ── ResNet50 3 epoch 학습 ──
model = DeepModel('ResNet50').to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
model.train()

total_t0 = time.time()
for ep in range(3):
    t0 = time.time()
    tot = correct = 0
    losssum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        losssum += loss.item() * len(yb)
        tot += len(yb)
        correct += (out.argmax(1) == yb).sum().item()
    dt = time.time() - t0
    log(f"Epoch {ep+1}/3 | loss {losssum/tot:.4f} | acc {correct/tot:.4f} | {dt:.1f}s/epoch")

total = time.time() - total_t0
log("=" * 50)
log(f"테스트 완료 | 총 {total:.1f}s | device={device}")
log(f"판정: {'GPU 정상 (빠름)' if total < 60 else 'CPU 의심 (느림)'}")
log("=" * 50)

with open('./test_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(result))
