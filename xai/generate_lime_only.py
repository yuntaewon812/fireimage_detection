import os
import sys
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_image import LimeImageExplainer
import warnings
warnings.filterwarnings('ignore')

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
base_result_path = "./results_lime"
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


def compute_lime(model, original_img):
    explainer = LimeImageExplainer()
    np_img = np.array(original_img.resize((224, 224))).astype(np.float64) / 255.0

    def classifier_fn(x):
        model.eval()
        with torch.no_grad():
            batch = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            probs = torch.nn.functional.softmax(model(batch), dim=1)
            return probs.detach().cpu().numpy()

    explanation = explainer.explain_instance(
        np_img, classifier_fn=classifier_fn,
        top_labels=1, hide_color=0, num_samples=500, batch_size=10
    )

    label = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[label])
    segments = explanation.segments
    heatmap = np.zeros((224, 224))
    for seg_id, weight in dict_heatmap.items():
        heatmap[segments == seg_id] = weight

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    abs_max = max(abs(heatmap.min()), abs(heatmap.max()))
    norm = (mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            if abs_max > 0 else mcolors.Normalize(0, 1))
    heatmap_colored = cm.get_cmap('RdBu_r')(norm(heatmap))[:, :, :3]

    return np.clip(0.4 * np_img + 0.6 * heatmap_colored, 0, 1)


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


def save_fold_visualizations(image_path, lime_imgs, save_dir, model_name, subclass):
    base_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.join(save_dir, subclass, base_name)
    os.makedirs(img_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img.save(os.path.join(img_dir, f"{base_name}_original.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.suptitle(f"{model_name} LIME (Fold 0~2)", fontsize=14)
    for i, lime_img in enumerate(lime_imgs):
        axes[i].imshow(lime_img)
        axes[i].set_title(f"Fold {i}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{base_name}_lime_compare.png"))
    plt.close()

    avg_lime = np.mean(np.stack(lime_imgs), axis=0)
    plt.imsave(os.path.join(img_dir, f"{base_name}_lime_average.png"), avg_lime)


def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"\nProcessing class: {target_class}")
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

        print(f"  Running LIME on {len(selected_paths)} images...")
        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  [{idx+1}/{len(selected_paths)}] {os.path.basename(img_path)}")
            try:
                lime_imgs = []
                img = Image.open(img_path).convert("RGB")

                for fold_num in [0, 1, 2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    if not os.path.exists(model_path):
                        print(f"    fold{fold_num} model not found, skipping")
                        continue
                    model = build_model(model_name, model_path)
                    lime_imgs.append(compute_lime(model, img))
                    del model
                    torch.cuda.empty_cache()

                if lime_imgs:
                    save_fold_visualizations(img_path, lime_imgs,
                                             os.path.join(base_result_path, target_class), model_name, subclass)
            except Exception as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    num_samples_per_subclass = 10
    for model_name in TOP2_MODELS:
        print(f"\n{'='*50}")
        print(f"LIME: {model_name}  |  device: {device}")
        print('='*50)
        run_all_classes(model_name, num_samples_per_subclass)
    print("\nAll LIME processing complete!")
