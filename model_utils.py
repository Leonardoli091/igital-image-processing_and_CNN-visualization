import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
LOCAL_TORCH_HOME = BASE_DIR / "weights_cache"
LOCAL_TORCH_HOME.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(LOCAL_TORCH_HOME)


MODEL_CONFIGS = {
    "vgg16": {
        "label": "VGG16",
        "display_name": "VGG16 ImageNet pretrained",
        "weight_file": "vgg16-397923af.pth",
        "weight_size": "about 528 MiB",
        "layers": {
            "conv1_1": 0,
            "conv1_2": 2,
            "conv2_1": 5,
            "conv2_2": 7,
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            "conv4_1": 17,
            "conv4_2": 19,
            "conv4_3": 21,
            "conv5_1": 24,
            "conv5_2": 26,
            "conv5_3": 28,
        },
    },
    "resnet18": {
        "label": "ResNet18",
        "display_name": "ResNet18 ImageNet pretrained",
        "weight_file": "resnet18-f37072fd.pth",
        "weight_size": "about 45 MiB",
        "layers": {
            "conv1": "conv1",
            "layer1": "layer1",
            "layer1_block1_conv1": "layer1.0.conv1",
            "layer1_block1_conv2": "layer1.0.conv2",
            "layer1_block1_out": "layer1.0",
            "layer1_block2_conv1": "layer1.1.conv1",
            "layer1_block2_conv2": "layer1.1.conv2",
            "layer1_block2_out": "layer1.1",
            "layer2": "layer2",
            "layer2_block1_conv1": "layer2.0.conv1",
            "layer2_block1_conv2": "layer2.0.conv2",
            "layer2_block1_downsample": "layer2.0.downsample.0",
            "layer2_block1_out": "layer2.0",
            "layer2_block2_conv1": "layer2.1.conv1",
            "layer2_block2_conv2": "layer2.1.conv2",
            "layer2_block2_out": "layer2.1",
            "layer3": "layer3",
            "layer3_block1_conv1": "layer3.0.conv1",
            "layer3_block1_conv2": "layer3.0.conv2",
            "layer3_block1_downsample": "layer3.0.downsample.0",
            "layer3_block1_out": "layer3.0",
            "layer3_block2_conv1": "layer3.1.conv1",
            "layer3_block2_conv2": "layer3.1.conv2",
            "layer3_block2_out": "layer3.1",
            "layer4": "layer4",
            "layer4_block1_conv1": "layer4.0.conv1",
            "layer4_block1_conv2": "layer4.0.conv2",
            "layer4_block1_downsample": "layer4.0.downsample.0",
            "layer4_block1_out": "layer4.0",
            "layer4_block2_conv1": "layer4.1.conv1",
            "layer4_block2_conv2": "layer4.1.conv2",
            "layer4_block2_out": "layer4.1",
        },
    },
}


def list_visual_models():
    return [
        {
            "id": model_id,
            "label": config["label"],
            "name": config["display_name"],
            "weight_size": config["weight_size"],
        }
        for model_id, config in MODEL_CONFIGS.items()
    ]


def list_visual_layers():
    result = {}
    for model_id, config in MODEL_CONFIGS.items():
        result[model_id] = [
            {"id": name, "label": name, "index": index}
            for name, index in config["layers"].items()
        ]
    return result


def model_status():
    try:
        import torchvision  # noqa: F401

        return {
            "ready": True,
            "model": "VGG16 / ResNet18 ImageNet pretrained",
            "message": "CNN dependencies are ready. Weights are downloaded only when you generate feature maps.",
            "cache_dir": str(LOCAL_TORCH_HOME),
            "weights": get_local_weight_paths(),
        }
    except Exception as exc:
        return {
            "ready": False,
            "model": "VGG16 / ResNet18 ImageNet pretrained",
            "message": "torch or torchvision is not available. Install dependencies from requirements.txt first.",
            "detail": str(exc),
        }


def get_local_weight_paths():
    paths = {}
    for model_id, config in MODEL_CONFIGS.items():
        paths[model_id] = str(LOCAL_TORCH_HOME / "hub" / "checkpoints" / config["weight_file"])
    return paths


@lru_cache(maxsize=1)
def _load_vgg16():
    from torchvision import models, transforms

    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model.eval()
    preprocess = _imagenet_preprocess(transforms)
    return model, preprocess, weights


@lru_cache(maxsize=1)
def _load_resnet18():
    from torchvision import models, transforms

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    preprocess = _imagenet_preprocess(transforms)
    return model, preprocess, weights


def _imagenet_preprocess(transforms):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_layer_features(image: Image.Image, layer_name: str, model_id: str = "vgg16"):
    if model_id == "resnet18":
        return _extract_resnet18_features(image, layer_name)
    if model_id == "vgg16":
        return _extract_vgg16_features(image, layer_name)
    raise ValueError("Unknown CNN model.")


def grad_cam_heatmap(image: Image.Image, layer_name: str, model_id: str = "resnet18"):
    model, preprocess, weights_meta, target_module = _grad_cam_target(model_id, layer_name)
    activation = {}
    gradient = {}

    def forward_hook(_module, _inputs, output):
        activation["value"] = output
        output.register_hook(lambda grad: gradient.__setitem__("value", grad))

    handle = target_module.register_forward_hook(forward_hook)
    try:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        score.backward()
    finally:
        handle.remove()

    if "value" not in activation or "value" not in gradient:
        raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

    features = activation["value"][0].detach()
    grads = gradient["value"][0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * features).sum(dim=0))
    cam_np = cam.cpu().numpy().astype(np.float32)
    cam_np = _normalize_cam(cam_np)

    base = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    heat = Image.fromarray((cam_np * 255).astype(np.uint8), "L").resize((224, 224), Image.Resampling.BICUBIC)
    heat_rgb = Image.fromarray(_jet_color(np.asarray(heat, dtype=np.float32) / 255.0), "RGB")
    overlay = Image.blend(base, heat_rgb, 0.45)

    info = {
        "model": MODEL_CONFIGS[model_id]["display_name"],
        "model_id": model_id,
        "layer": layer_name,
        "class_index": class_index,
        "class_label": _imagenet_label(weights_meta, class_index),
        "score": float(score.detach().cpu().item()),
        "shape": list(features.shape),
    }
    return overlay, info


def _grad_cam_target(model_id: str, layer_name: str):
    if model_id == "resnet18":
        config = MODEL_CONFIGS["resnet18"]
        if layer_name not in config["layers"]:
            layer_name = "layer4"
        model, preprocess, weights_meta = _load_resnet18()
        return model, preprocess, weights_meta, _resolve_module(model, config["layers"][layer_name])

    if model_id == "vgg16":
        config = MODEL_CONFIGS["vgg16"]
        if layer_name not in config["layers"]:
            layer_name = "conv5_3"
        model, preprocess, weights_meta = _load_vgg16()
        return model, preprocess, weights_meta, model.features[config["layers"][layer_name]]

    raise ValueError("Unknown CNN model.")


def _imagenet_label(weights_meta, class_index: int) -> str:
    categories = getattr(weights_meta, "meta", {}).get("categories", [])
    if 0 <= class_index < len(categories):
        return categories[class_index]
    try:
        from torchvision.models._meta import _IMAGENET_CATEGORIES

        if 0 <= class_index < len(_IMAGENET_CATEGORIES):
            return _IMAGENET_CATEGORIES[class_index]
    except Exception:
        pass
    return ""


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - float(cam.min())
    high = float(cam.max())
    if high < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return cam / high


def _jet_color(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1).clip(0, 1) * 255).astype(np.uint8)


def _extract_vgg16_features(image: Image.Image, layer_name: str):
    config = MODEL_CONFIGS["vgg16"]
    if layer_name not in config["layers"]:
        raise ValueError("Unknown VGG16 convolution layer.")

    model, preprocess, _weights_meta = _load_vgg16()
    target_index = config["layers"][layer_name]
    activation = {}

    def hook(_module, _inputs, output):
        activation["value"] = output.detach().cpu()

    handle = model.features[target_index].register_forward_hook(hook)
    try:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            model.features(tensor)
    finally:
        handle.remove()

    return _format_feature_result(activation, "vgg16", layer_name)


def _extract_resnet18_features(image: Image.Image, layer_name: str):
    config = MODEL_CONFIGS["resnet18"]
    if layer_name not in config["layers"]:
        raise ValueError("Unknown ResNet18 feature layer.")

    model, preprocess, _weights_meta = _load_resnet18()
    target_module = _resolve_module(model, config["layers"][layer_name])
    activation = {}

    def hook(_module, _inputs, output):
        activation["value"] = output.detach().cpu()

    handle = target_module.register_forward_hook(hook)
    try:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            model(tensor)
    finally:
        handle.remove()

    return _format_feature_result(activation, "resnet18", layer_name)


def _resolve_module(model, module_path: str):
    module = model
    for part in str(module_path).split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _format_feature_result(activation, model_id: str, layer_name: str):
    if "value" not in activation:
        raise RuntimeError("Failed to capture the selected feature layer output.")

    features = activation["value"][0]
    info = {
        "model": MODEL_CONFIGS[model_id]["display_name"],
        "model_id": model_id,
        "layer": layer_name,
        "shape": list(features.shape),
        "message": "The grid shows selected channels from this intermediate feature tensor.",
    }
    return features, info
