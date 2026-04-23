import os
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 경로 및 설정 (Windows 환경에 맞게 ./ 사용)
class_names = ["fireimage"]
base_model_path = "./model_save"
base_result_path = "./results_saliency"
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_saliency_map(model, input_tensor):
    # Gradient를 계산하기 위해 설정
    input_tensor.requires_grad = True
    
    # 모델 예측
    output = model(input_tensor)
    
    # 가장 높은 확률을 가진 클래스 선택
    if output.shape[1] > 1:
        class_idx = output.argmax(dim=1).item()
    else:
        class_idx = 0
        
    model.zero_grad()
    # 해당 클래스 점수에 대해 역전파
    output[0, class_idx].backward()
    
    # 입력 이미지의 gradient 절댓값 중 채널(RGB) 최대값 선택
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    return saliency

def build_model(model_name, model_path):
    # 2. DenseNet121 지원 추가
    if model_name == "DenseNet121":
        from torchvision import models
        import torch.nn as nn
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == "efficientnetv2_proposal":
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    # 필요한 다른 모델들 (convnextv2 등)이 있다면 여기에 추가 가능
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # 3. Key 불일치 해결을 위한 로딩 (strict=False)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device).eval()
    return model

def get_top_confident_samples(model_name, target_class, img_paths, num_samples):
    print(f"  Calculating confidence scores for {len(img_paths)} images...")
    # Fold0 모델을 기준으로 신뢰도 측정
    model_path = os.path.join(base_model_path, target_class, "fold0", f"{model_name}.pt")
    if not os.path.exists(model_path):
        return img_paths[:num_samples]

    model = build_model(model_name, model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

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
            scores.append((img_path, 0.0))

    del model
    torch.cuda.empty_cache()
    scores.sort(key=lambda x: x[1], reverse=True)
    return [path for path, score in scores[:num_samples]]

def save_fold_visualizations(image_path, saliencies, save_dir, model_name, subclass):
    base_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.join(save_dir, subclass, base_name)
    os.makedirs(img_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img) / 255.0
    img.save(os.path.join(img_dir, f"{base_name}_original.png"))

    # 4. Fold별 비교 이미지 생성
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"{model_name} Saliency Comparison (Fold 0~2)", fontsize=16)

    for i in range(len(saliencies)):
        saliency_np = saliencies[i].squeeze().cpu().numpy()
        # 정규화: 시각화를 위해 0~1 사이로 조정
        saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
        
        axes[i].imshow(img_np)
        axes[i].imshow(saliency_np, cmap='hot', alpha=0.6) # hot 컬러맵이 saliency에 적합
        axes[i].set_title(f"Fold {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{base_name}_compare.png"))
    plt.close()

    # 5. 평균 Saliency 생성
    avg_saliency = torch.mean(torch.stack(saliencies), dim=0).squeeze().cpu().numpy()
    avg_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.max() - avg_saliency.min() + 1e-8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(avg_saliency, cmap='hot', alpha=0.6)
    plt.title("Average Saliency Map")
    plt.axis('off')
    plt.savefig(os.path.join(img_dir, f"{base_name}_average.png"))
    plt.close()

def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"Processing class: {target_class}")
        # 6. labelling 폴더 포함 탐색
        subclasses = ["normal", "abnormal", "labelling"]
        selected_paths = []

        for sub in subclasses:
            paths = []
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                paths.extend(glob.glob(os.path.join(data_path, target_class, sub, ext)))
            
            if paths:
                top_samples = get_top_confident_samples(model_name, target_class, paths, num_samples_per_subclass)
                selected_paths.extend([(p, sub) for p in top_samples])

        if not selected_paths:
            print(f"  No images found for {target_class}")
            continue

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  [{idx+1}/{len(selected_paths)}] Processing: {os.path.basename(img_path)}")
            try:
                saliencies = []
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if os.path.exists(model_path):
                        model = build_model(model_name, model_path)
                        saliency = compute_saliency_map(model, input_tensor.clone())
                        saliencies.append(saliency)
                        del model
                        torch.cuda.empty_cache()

                if saliencies:
                    save_fold_visualizations(img_path, saliencies, base_result_path, model_name, subclass)
                    print(f"    Saved results")
            except Exception as e:
                print(f"    Error: {e}")

if __name__ == "__main__":
    # 실제 학습한 모델 이름인 DenseNet121로 설정
    model_name = "DenseNet121" 
    run_all_classes(model_name, 10)
    print("\nSaliency Map 생성 완료!")