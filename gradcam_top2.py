"""
gradcam_top2.py — efficientnetv2 + maxvit Grad-CAM 시각화
==========================================================
목적: "모델이 불/연기 영역을 보고 판단하는가?" 검증
대상: 설정 E fold0 원본 FP32 가중치 (경량화 모델은 gradient 계산 불가)

출력: results/gradcam_top2/{normal,abnormal}/{이미지명}/
  - original.jpg
  - efficientnetv2_gradcam.png
  - maxvit_gradcam.png
  - compare.png  (원본 + 두 모델 나란히)
"""
import os, sys, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DATA_DIR    = os.path.join(BASE, 'data', 'fireimage')
WEIGHT_DIR  = os.path.join(BASE, 'model_save', 'fireimage_abl_E', 'fold0')
SAVE_DIR    = os.path.join(BASE, 'results', 'gradcam_top2')
IMG_SIZE    = 160   # 학습 해상도
N_SAMPLES   = 10    # 클래스당 샘플 수
DEVICE      = torch.device('cpu')  # CPU (로컬, MPS/CUDA 없으면 CPU)

# ── 모델 빌드 (원본 FP32, pretrained 구조 재현) ────────────────────
def build_efficientnetv2():
    import timm
    backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False,
                                 num_classes=0, global_pool='avg')
    num_features = backbone.num_features

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.LayerNorm(num_features), nn.Linear(num_features, 512),
                nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2))
        def forward(self, x):
            return self.classifier(self.backbone(x))
    return _M()


def build_maxvit():
    import timm
    backbone = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=False,
                                 num_classes=0, global_pool='avg')
    num_features = backbone.num_features

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.LayerNorm(num_features), nn.Linear(num_features, 512),
                nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2))
        def forward(self, x):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return self.classifier(self.backbone(x))
    return _M()


def load_model(build_fn, weight_name):
    w = os.path.join(WEIGHT_DIR, weight_name)
    if not os.path.exists(w):
        raise FileNotFoundError(f'가중치 없음: {w}')
    m = build_fn()
    m.load_state_dict(torch.load(w, map_location='cpu', weights_only=True))
    m.to(DEVICE).eval()
    return m


# ── 이미지 전처리 ──────────────────────────────────────────────────
def preprocess(img_path, size=IMG_SIZE):
    img = Image.open(img_path).convert('RGB').resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, arr   # tensor: (1,3,H,W), arr: (H,W,3) 0~1


# ── Grad-CAM (efficientnetv2 — CNN) ────────────────────────────────
def gradcam_effnet(model, tensor):
    # conv_head: 마지막 1×1 Conv2d, 4D 출력 확실
    target_layer = [model.backbone.conv_head]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale = cam(input_tensor=tensor, targets=None)[0]
    cam.activations_and_grads.release()
    return grayscale   # (H,W) 0~1


# ── Grad-CAM (maxvit — MBConv 마지막 projection Conv) ─────────────
def gradcam_maxvit(model, tensor):
    # stages[-1].blocks[-1].conv.conv3_1x1: 마지막 MBConv의 projection (1×1 Conv2d)
    # 공간 feature map(4D)을 보존하므로 reshape 불필요
    target_layer = [model.backbone.stages[-1].blocks[-1].conv.conv3_1x1]
    cam = GradCAM(model=model, target_layers=target_layer)
    # maxvit 내부에서 224로 보간 → GradCAM도 224 기준으로 계산
    tensor_224 = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    grayscale = cam(input_tensor=tensor_224, targets=None)[0]
    cam.activations_and_grads.release()
    return grayscale   # (H,W) 0~1


# ── 예측 결과 텍스트 ───────────────────────────────────────────────
def predict_label(model, tensor):
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    label = '화재(abnormal)' if pred == 1 else '정상(normal)'
    conf = float(probs[pred])
    return label, conf


# ── 시각화 저장 ───────────────────────────────────────────────────
def save_compare(img_arr, cam_eff, cam_max, save_dir, base_name,
                 label_eff, conf_eff, label_max, conf_max):
    os.makedirs(save_dir, exist_ok=True)

    # 원본 (160x160) → numpy RGB 0~1
    img_small = img_arr  # (160,160,3)

    # Grad-CAM 오버레이 (show_cam_on_image는 0~1 float RGB 요구)
    cam_img_eff = show_cam_on_image(img_small, cam_eff, use_rgb=True)
    # maxvit은 224 → 160으로 리사이즈
    cam_max_resized = np.array(Image.fromarray(
        (cam_max * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    cam_img_max = show_cam_on_image(img_small, cam_max_resized, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(base_name, fontsize=11)

    axes[0].imshow(img_small)
    axes[0].set_title('원본')
    axes[0].axis('off')

    axes[1].imshow(cam_img_eff)
    axes[1].set_title(f'EfficientNetV2\n{label_eff} ({conf_eff:.1%})')
    axes[1].axis('off')

    axes[2].imshow(cam_img_max)
    axes[2].set_title(f'MaxViT\n{label_max} ({conf_max:.1%})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{base_name}_compare.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # 개별 저장
    plt.imsave(os.path.join(save_dir, f'{base_name}_original.jpg'), img_small)
    plt.imsave(os.path.join(save_dir, f'{base_name}_effnet_gradcam.png'), cam_img_eff)
    plt.imsave(os.path.join(save_dir, f'{base_name}_maxvit_gradcam.png'), cam_img_max)


# ── 메인 ─────────────────────────────────────────────────────────
def main():
    print('[모델 로드]')
    eff_model = load_model(build_efficientnetv2, 'efficientnetv2.pt')
    max_model = load_model(build_maxvit,         'maxvit.pt')
    print('  efficientnetv2, maxvit 로드 완료\n')

    for label_name in ['normal', 'abnormal']:
        paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            paths.extend(glob.glob(
                os.path.join(DATA_DIR, label_name, '**', ext), recursive=True))

        if not paths:
            print(f'[건너뜀] {label_name} 이미지 없음')
            continue

        # 랜덤 샘플링
        random.seed(1004)
        sampled = random.sample(paths, min(N_SAMPLES, len(paths)))
        print(f'[{label_name}] {len(sampled)}장 처리 중...')

        save_sub = os.path.join(SAVE_DIR, label_name)

        for i, img_path in enumerate(sampled):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f'  [{i+1}/{len(sampled)}] {base_name}', end=' ', flush=True)

            try:
                tensor, img_arr = preprocess(img_path)
                tensor = tensor.to(DEVICE)

                # 예측
                label_eff, conf_eff = predict_label(eff_model, tensor)
                label_max, conf_max = predict_label(max_model, tensor)

                # Grad-CAM
                cam_eff = gradcam_effnet(eff_model, tensor)
                cam_max = gradcam_maxvit(max_model, tensor)

                save_compare(img_arr, cam_eff, cam_max, save_sub, base_name,
                             label_eff, conf_eff, label_max, conf_max)
                print('✓')
            except Exception as e:
                print(f'오류: {e}')

    print(f'\n[완료] 결과: {SAVE_DIR}')
    # 파일 수 출력
    total = len(glob.glob(os.path.join(SAVE_DIR, '**', '*_compare.png'), recursive=True))
    print(f'  compare 이미지 {total}개 생성됨')


if __name__ == '__main__':
    main()
