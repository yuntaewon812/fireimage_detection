import os
import sys
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

TOP2_MODELS = ["Resnet50", "DenseNet121"]

class_names = ["fireimage"]
base_model_path = "./model_save"
base_result_path = "./results_gradcam"
data_path = "./data"

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


def get_target_layer(model, model_name):
    if model_name == "Resnet50":
        return [model.feature_extractor[-1]]
    elif model_name == "DenseNet121":
        return [model.feature_extractor[-2]]
    else:
        raise ValueError(f"No target layer defined for: {model_name}")


def compute_gradcam(model, input_tensor, model_name):
    try:
        target_layers = get_target_layer(model, model_name)
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        import matplotlib.cm as cm
        heatmap_colored = cm.get_cmap('jet')(grayscale_cam)[:, :, :3]

        visualization = 0.5 * rgb_img + 0.5 * heatmap_colored
        visualization = np.clip(visualization, 0, 1)
        cam.activations_and_grads.release()
        return visualization
    except Exception as e:
        print(f"      GradCAM error: {e}, using original image as fallback")
        return input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()


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
            scores.append((img_path, 0.0))

    del model
    torch.cuda.empty_cache()
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"  Selected top {min(num_samples, len(scores))} confident samples")
    return [p for p, _ in scores[:num_samples]]


def save_fold_visualizations(image_path, gradcam_imgs, save_dir, model_name, subclass):
    base_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.join(save_dir, subclass, base_name)
    os.makedirs(img_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img.save(os.path.join(img_dir, f"{base_name}_original.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.suptitle(f"{model_name} GradCAM (Fold 0~2)", fontsize=14)
    for i in range(3):
        axes[i].imshow(gradcam_imgs[i])
        axes[i].set_title(f"Fold {i}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{base_name}_{model_name}_gradcam_folds_compare.png"), bbox_inches="tight")
    plt.close()

    avg_gradcam = np.mean(np.stack(gradcam_imgs), axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(avg_gradcam)
    ax.set_title("Average GradCAM")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{base_name}_{model_name}_gradcam_average.png"), bbox_inches="tight")
    plt.close()


def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"\nProcessing class: {target_class}")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        selected_paths = []

        for sub in ["abnormal", "normal"]:
            paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                paths.extend(glob.glob(os.path.join(data_path, target_class, sub, "**", ext), recursive=True))
            if paths:
                print(f"  Found {len(paths)} {sub} images")
                top = get_top_confident_samples(model_name, target_class, paths, num_samples_per_subclass)
                selected_paths.extend([(p, sub) for p in top])

        if not selected_paths:
            print(f"  No images found for {target_class}")
            continue

        print(f"  Running GradCAM on {len(selected_paths)} images...")
        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  [{idx+1}/{len(selected_paths)}] {os.path.basename(img_path)}")
            try:
                gradcam_imgs = []
                input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if not os.path.exists(model_path):
                        print(f"    fold{fold_num} model not found, skipping")
                        continue
                    model = build_model(model_name, model_path)
                    gradcam_imgs.append(compute_gradcam(model, input_tensor, model_name))
                    del model
                    torch.cuda.empty_cache()

                if gradcam_imgs:
                    save_fold_visualizations(img_path, gradcam_imgs,
                                             os.path.join(base_result_path, target_class), model_name, subclass)
            except Exception as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    num_samples_per_subclass = 10
    for model_name in TOP2_MODELS:
        print(f"\n{'='*50}")
        print(f"GradCAM: {model_name}  |  device: {device}")
        print('='*50)
        run_all_classes(model_name, num_samples_per_subclass)
    print("\nAll GradCAM processing complete!")
