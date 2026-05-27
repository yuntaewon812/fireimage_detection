import torch


def build_model(model_name: str, model_path: str, img_size: int, device):
    name = model_name.lower()

    if name == "resnet50":
        from models.deep_model import DeepModel
        model = DeepModel('ResNet50')
    elif name == "densenet121":
        from models.deep_model import DeepModel
        model = DeepModel('DenseNet121')
    elif "nextvit" in name:
        from models.nextvit import NextViTForImageClassification
        model = NextViTForImageClassification(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="small",
        )
    elif "efficient" in name and "proposal" not in name:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification
        model = EfficientNetV2ForImageClassification(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="s",
        )
    elif "efficient" in name and "proposal" in name:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="s",
        )
    elif "mamba" in name:
        from models.mambavision import MambaVisionForImageClassification_v2
        model = MambaVisionForImageClassification_v2(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="tiny",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model
