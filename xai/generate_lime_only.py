import os
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_image import LimeImageExplainer

# 1. 시드 설정 (결과 재현성)
SEED = 1004
def setSeed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setSeed(SEED)

# 2. 경로 및 설정 (사용자님 환경에 맞게 수정)
class_names = ["fireimage"]
base_model_path = "./model_save"
base_result_path = "./results_lime"
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_lime(model, input_tensor, original_img):
    import warnings
    warnings.filterwarnings('ignore')

    explainer = LimeImageExplainer()
    # LIME은 Numpy 기반이므로 변환
    np_img = np.array(original_img.resize((224, 224))).astype(np.float64) / 255.0

    def classifier_fn(x):
        model.eval()
        with torch.no_grad():
            # LIME이 생성한 샘플들을 모델이 예측하도록 변환
            batch = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.detach().cpu().numpy()

    # 샘플 수를 늘리면 정확하지만 시간이 오래 걸립니다 (num_samples)
    explanation = explainer.explain_instance(
        np_img, 
        classifier_fn=classifier_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=500, 
        batch_size=10
    )

    label = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[label])
    segments = explanation.segments
    heatmap = np.zeros((224, 224))

    for segment_id, weight in dict_heatmap.items():
        mask = (segments == segment_id)
        heatmap[mask] = weight

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vmin, vmax = heatmap.min(), heatmap.max()
    abs_max = max(abs(vmin), abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max) if abs_max > 0 else mcolors.Normalize(0, 1)

    cmap = cm.get_cmap('RdBu_r') # 붉은색: 긍정적 기여, 파란색: 부정적 기여
    heatmap_colored = cmap(norm(heatmap))[:, :, :3]

    alpha = 0.4
    lime_img = alpha * np_img + (1 - alpha) * heatmap_colored
    return np.clip(lime_img, 0, 1)

def build_model(model_name, model_path):
    # 이전에 발생한 DenseNet121 지원 추가
    if model_name == "DenseNet121":
        from torchvision import models
        import torch.nn as nn
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == "efficientnetv2_proposal":
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    # ... 필요한 다른 모델들 추가 ...
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Key 불일치 해결을 위한 strict=False 로딩
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device).eval()
    return model

def get_top_confident_samples(model_name, target_class, img_paths, num_samples):
    print(f"  Calculating confidence scores for {len(img_paths)} images...")
    model_path = os.path.join(base_model_path, target_class, "fold0", f"{model_name}.pt")
    if not os.path.exists(model_path):
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
            scores.append((img_path, 0.0))
    
    del model
    torch.cuda.empty_cache()
    scores.sort(key=lambda x: x[1], reverse=True)
    return [path for path, score in scores[:num_samples]]

def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"Processing class: {target_class}")
        # labelling 폴더 포함 데이터 구조 반영
        subclasses = ["normal", "abnormal", "labelling"]
        selected_paths = []

        for sub in subclasses:
            paths = []
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                paths.extend(glob.glob(os.path.join(data_path, target_class, sub, ext)))
            
            if paths:
                top_samples = get_top_confident_samples(model_name, target_class, paths, num_samples_per_subclass)
                selected_paths.extend([(p, sub) for p in top_samples])

        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  [{idx+1}/{len(selected_paths)}] Processing: {os.path.basename(img_path)}")
            try:
                lime_imgs = []
                img = Image.open(img_path).convert("RGB")
                
                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if os.path.exists(model_path):
                        model = build_model(model_name, model_path)
                        lime_img = compute_lime(model, None, img) # input_tensor 대신 원본 img 사용
                        lime_imgs.append(lime_img)
                        del model
                        torch.cuda.empty_cache()

                if lime_imgs:
                    save_fold_visualizations(img_path, lime_imgs, os.path.join(base_result_path, target_class), model_name, subclass)
            except Exception as e:
                print(f"    Error: {e}")

def save_fold_visualizations(image_path, lime_imgs, save_dir, model_name, subclass):
    base_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.join(save_dir, subclass, base_name)
    os.makedirs(img_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img.save(os.path.join(img_dir, f"{base_name}_original.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(len(lime_imgs)):
        axes[i].imshow(lime_imgs[i])
        axes[i].set_title(f"Fold {i}")
        axes[i].axis('off')
    plt.savefig(os.path.join(img_dir, f"{base_name}_lime_compare.png"))
    plt.close()

    avg_lime = np.mean(np.stack(lime_imgs), axis=0)
    plt.imsave(os.path.join(img_dir, f"{base_name}_lime_average.png"), avg_lime)

if __name__ == "__main__":
    # 아까 성공했던 DenseNet121로 설정
    model_name = "DenseNet121" 
    run_all_classes(model_name, 10)
    print("LIME 시각화 완료!")