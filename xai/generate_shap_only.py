import os
import sys
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP2_MODELS = ["Resnet50", "DenseNet121"]

class_names = ["fireimage"]
base_model_path = "./model_save"
base_result_path = "./results_shap"
data_path = "./data"

os.makedirs(base_result_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, model_path):
    from models.deep_model import DeepModel
    if model_name == "Resnet50":
        model = DeepModel('ResNet50')
    elif model_name == "DenseNet121":
        model = DeepModel('DenseNet121')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def compute_shap(model, input_tensor):
    background = torch.cat([
        input_tensor + torch.randn_like(input_tensor) * 0.1 for _ in range(5)
    ], dim=0).to(device)

    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(input_tensor)

    with torch.no_grad():
        pred_class = model(input_tensor).argmax(dim=1).item()

    # SHAP 버전에 따라 반환 shape이 다름:
    # list 형태: [class0_array, class1_array], 각 shape (batch, C, H, W)
    # ndarray 형태: (n_classes, batch, C, H, W) 또는 (batch, C, H, W)
    if isinstance(shap_values, list):
        idx = pred_class if pred_class < len(shap_values) else 0
        shap_array = np.array(shap_values[idx])          # (batch, C, H, W)
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 5:                          # (n_classes, batch, C, H, W)
            idx = pred_class if pred_class < shap_array.shape[0] else 0
            shap_array = shap_array[idx]

    if shap_array.ndim == 4:                             # (batch, C, H, W) → (C, H, W)
        shap_array = shap_array[0]

    if shap_array.ndim == 3 and shap_array.shape[0] <= 4:  # (C, H, W) → (H, W, C)
        shap_array = np.transpose(shap_array, (1, 2, 0))

    shap_img = np.mean(shap_array, axis=2)

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    abs_max = max(abs(shap_img.min()), abs(shap_img.max()))
    norm = (mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            if abs_max > 0 else mcolors.Normalize(0, 1))

    np_img = input_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    shap_colored = cm.get_cmap('RdBu_r')(norm(shap_img))[:, :, :3]
    return np.clip(0.5 * np_img + 0.5 * shap_colored, 0, 1)


def get_top_confident_samples(model_name, target_class, img_paths, num_samples):
    print(f"  Scoring {len(img_paths)} images with fold0 model...")
    model_path = os.path.join(base_model_path, target_class, "fold0", f"{model_name}.pt")
    if not os.path.exists(model_path):
        return img_paths[:num_samples]

    model = build_model(model_name, model_path)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    scores = []
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(model(tensor), dim=1)
                scores.append((img_path, probs.max().item()))
        except Exception:
            continue

    del model
    torch.cuda.empty_cache()
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"  Selected top {min(num_samples, len(scores))} confident samples")
    return [p for p, _ in scores[:num_samples]]


def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"\nProcessing class: {target_class}")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        selected_paths = []

        for sub in ["abnormal", "normal"]:
            paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                paths.extend(glob.glob(os.path.join(data_path, target_class, sub, "**", ext), recursive=True))
            if paths:
                print(f"  Found {len(paths)} {sub} images")
                top = get_top_confident_samples(model_name, target_class, paths, num_samples_per_subclass)
                selected_paths.extend([(p, sub) for p in top])

        if not selected_paths:
            print(f"  No images found for {target_class}")
            continue

        print(f"  Running SHAP on {len(selected_paths)} images...")
        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  [{idx+1}/{len(selected_paths)}] {os.path.basename(img_path)}")
            try:
                shap_imgs = []
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if not os.path.exists(model_path):
                        print(f"    fold{fold_num} model not found, skipping")
                        continue
                    model = build_model(model_name, model_path)
                    shap_imgs.append(compute_shap(model, input_tensor.clone()))
                    del model
                    torch.cuda.empty_cache()

                if shap_imgs:
                    base_name = os.path.basename(img_path).split('.')[0]
                    img_dir = os.path.join(base_result_path, target_class, model_name, subclass, base_name)
                    if os.path.exists(os.path.join(img_dir, f"{base_name}_average.png")):
                        print(f"    Already done, skipping.")
                        continue
                    os.makedirs(img_dir, exist_ok=True)

                    img.resize((224, 224)).save(os.path.join(img_dir, f"{base_name}_original.png"))

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    plt.suptitle(f"{model_name} SHAP (Fold 0~2)", fontsize=14)
                    for i, si in enumerate(shap_imgs):
                        axes[i].imshow(si)
                        axes[i].set_title(f"Fold {i}")
                        axes[i].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_dir, f"{base_name}_compare.png"), bbox_inches='tight')
                    plt.close()

                    avg_shap = np.mean(np.stack(shap_imgs), axis=0)
                    plt.imsave(os.path.join(img_dir, f"{base_name}_average.png"), avg_shap)
                    print(f"    Saved to {img_dir}")
            except Exception as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    num_samples_per_subclass = 10
    for model_name in TOP2_MODELS:
        print(f"\n{'='*50}")
        print(f"SHAP: {model_name}  |  device: {device}")
        print('='*50)
        run_all_classes(model_name, num_samples_per_subclass)
    print("\nAll SHAP processing complete!")
