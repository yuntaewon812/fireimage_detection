import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shap
import warnings

# 1. 경로 및 환경 설정
# 윈도우 환경에서 가장 안전한 ./ 상대 경로 방식을 유지합니다.
class_names = ["fireimage"]
base_model_path = "./model_save"
base_result_path = "./results_shap"
data_path = "./data"

# 결과 저장 폴더가 없으면 생성
os.makedirs(base_result_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def compute_shap(model, input_tensor):
    # SHAP용 배경 데이터 (노이즈가 섞인 5개 샘플)
    background_samples = []
    for _ in range(5):
        noise = torch.randn_like(input_tensor) * 0.1
        background_samples.append(input_tensor + noise)
    background = torch.cat(background_samples, dim=0).to(device)

    # GradientExplainer 설정 및 값 계산
    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(input_tensor)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    # SHAP 결과 배열화 및 차원 조정
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[pred_class])
    else:
        shap_array = np.array(shap_values)

    # (Batch, C, H, W) -> (H, W, C) 변환 로직
    if len(shap_array.shape) == 4:
        shap_array = shap_array[0]
    
    if shap_array.shape[0] <= 3: 
        shap_array = np.transpose(shap_array, (1, 2, 0))

    # 채널 평균 히트맵 생성
    shap_img = np.mean(shap_array, axis=2)

    # 시각화 정규화 (RdBu_r: 빨간색은 정적 기여, 파란색은 부적 기여)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vmin, vmax = shap_img.min(), shap_img.max()
    abs_max = max(abs(vmin), abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max) if abs_max > 0 else mcolors.Normalize(0, 1)
    
    np_img = input_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    shap_colored = cm.get_cmap('RdBu_r')(norm(shap_img))[:, :, :3]

    # 원본과 히트맵 합성
    alpha = 0.5
    result = alpha * np_img + (1 - alpha) * shap_colored
    return np.clip(result, 0, 1)

def build_model(model_name, model_path):
    if model_name == "DenseNet121":
        from torchvision import models
        import torch.nn as nn
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == "efficientnetv2_proposal":
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # 가중치 로딩 (Key 불일치 방지)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    return model

def get_top_confident_samples(model_name, target_class, img_paths, num_samples):
    model_path = os.path.join(base_model_path, target_class, "fold0", f"{model_name}.pt")
    if not os.path.exists(model_path):
        print(f"      [Warning] Model not found at {model_path}. Using random samples.")
        return img_paths[:num_samples]

    model = build_model(model_name, model_path)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    scores = []
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                scores.append((img_path, probs.max().item()))
        except:
            continue
    
    del model
    torch.cuda.empty_cache()
    scores.sort(key=lambda x: x[1], reverse=True)
    return [path for path, score in scores[:num_samples]]

def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"\n[1] Processing class: {target_class}")
        subclasses = ["normal", "abnormal", "labelling"]
        selected_paths = []

        # 이미지 수집 (대소문자 확장자 모두 포함)
        for sub in subclasses:
            paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                paths.extend(glob.glob(os.path.join(data_path, target_class, sub, ext)))
            
            if paths:
                print(f"    - Found {len(paths)} images in '{sub}' folder.")
                top_samples = get_top_confident_samples(model_name, target_class, paths, num_samples_per_subclass)
                selected_paths.extend([(p, sub) for p in top_samples])

        if not selected_paths:
            print(f"    [Error] No images found. Check your path: {os.path.abspath(data_path)}")
            continue

        print(f"[2] Starting SHAP calculation for {len(selected_paths)} images...")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"    ({idx+1}/{len(selected_paths)}) Analyzing: {os.path.basename(img_path)}")
            try:
                shap_imgs = []
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if os.path.exists(model_path):
                        model = build_model(model_name, model_path)
                        shap_img = compute_shap(model, input_tensor.clone())
                        shap_imgs.append(shap_img)
                        del model
                        torch.cuda.empty_cache()

                if shap_imgs:
                    # 결과물 저장 (폴더 자동 생성 포함)
                    base_name = os.path.basename(img_path).split('.')[0]
                    img_dir = os.path.join(base_result_path, target_class, subclass, base_name)
                    os.makedirs(img_dir, exist_ok=True)

                    # 1. 원본 저장
                    img.resize((224, 224)).save(os.path.join(img_dir, f"{base_name}_original.png"))
                    
                    # 2. Fold 비교샷 저장
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for i in range(len(shap_imgs)):
                        axes[i].imshow(shap_imgs[i])
                        axes[i].set_title(f"Fold {i}")
                        axes[i].axis('off')
                    plt.savefig(os.path.join(img_dir, f"{base_name}_compare.png"), bbox_inches='tight')
                    plt.close()

                    # 3. 평균 결과 저장
                    avg_shap = np.mean(np.stack(shap_imgs), axis=0)
                    plt.imsave(os.path.join(img_dir, f"{base_name}_average.png"), avg_shap)
                    print(f"      => Saved to {img_dir}")
            except Exception as e:
                print(f"      [Error] {e}")

if __name__ == "__main__":
    print(f"Current Directory: {os.getcwd()}")
    # 캡스톤 프로젝트에서 사용 중인 모델명으로 설정
    target_model = "DenseNet121" 
    run_all_classes(target_model, 10)
    print("\n" + "="*30 + "\nSHAP Processing Complete!\n" + "="*30)