import torch
from pytorch_grad_cam import GradCAM
from utils.utils import load_data

from PosNeg_auc_util.seed import set_seed
from PosNeg_auc_util.config import AUCConfig
from PosNeg_auc_util.eval import run_xai_pos_neg_auc_cv
from PosNeg_auc_util.load_model import build_model as model_loader

cfg = AUCConfig()
set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP2_MODELS = ["efficientnetv2", "nextvit"]


def get_cam_target_layers(model, model_name):
    name = model_name.lower()
    if "efficient" in name:
        return [model.backbone.blocks[-1]]
    elif "nextvit" in name:
        return [model.backbone.features[-1]]
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return [last_conv]
    mods = list(model.modules())
    return [mods[-1]] if len(mods) > 1 else [model]


def get_gradcam_map(model, input_tensor, model_name=""):
    target_layers = get_cam_target_layers(model, model_name)
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam(input_tensor=input_tensor, targets=None)[0, :]


def get_saliency_map(model, input_tensor, model_name=""):
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)
    output = model(x)
    idx = output.argmax(dim=1).item()
    model.zero_grad(set_to_none=True)
    output[0, idx].backward()
    return (
        torch.max(x.grad.data.abs(), dim=1)[0]
        .squeeze().detach().cpu().numpy()
    )


if __name__ == "__main__":
    for model_name in TOP2_MODELS:
        print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        methods = {
            "GradCAM": lambda model, x, mn=model_name: get_gradcam_map(model, x, mn),
            "Saliency": lambda model, x, mn=model_name: get_saliency_map(model, x, mn),
        }
        run_xai_pos_neg_auc_cv(
            cfg=cfg,
            load_data_fn=load_data,
            model_name=model_name,
            model_loader=model_loader,
            methods=methods,
            device=device,
            print_errors=True,
            out_suffix=f"{model_name}_pos_neg_perturb_auc.csv",
            write_cv_mean=True,
            write_all=True,
        )
