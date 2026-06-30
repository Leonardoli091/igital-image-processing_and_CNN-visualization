import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.dont_write_bytecode = True

import io

from flask import Flask, jsonify, render_template, request

from classical_filters import apply_classical_operation, list_classical_operations
from image_utils import (
    ALLOWED_EXTENSIONS,
    ensure_example_images,
    image_to_data_url,
    load_request_image,
    open_image,
)
from model_utils import grad_cam_heatmap, extract_layer_features, list_visual_layers, list_visual_models, model_status
from resize_utils import apply_resize_compression
from visualize_utils import feature_maps_to_grid, feature_maps_to_summary_images, top_activated_channels


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "static" / "examples"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True


def _asset_version() -> int:
    asset_paths = [
        BASE_DIR / "static" / "main.js",
        BASE_DIR / "static" / "style.css",
    ]
    try:
        return int(max(path.stat().st_mtime for path in asset_paths))
    except OSError:
        return 1


@app.route("/")
def index():
    ensure_example_images(EXAMPLE_DIR)
    return render_template("index.html", asset_version=_asset_version())


@app.get("/api/examples")
def examples():
    ensure_example_images(EXAMPLE_DIR)
    files = []
    for path in sorted(EXAMPLE_DIR.glob("*")):
        if path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(
                {
                    "name": path.name,
                    "label": path.stem.replace("_", " "),
                    "url": f"/static/examples/{path.name}",
                }
            )
    return jsonify({"examples": files})


@app.post("/api/upload-example")
def upload_example():
    try:
        if "image" not in request.files or not request.files["image"].filename:
            raise ValueError("请选择要保存到图片仓库的图片。")

        upload = request.files["image"]
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("只支持 jpg、jpeg、png、webp 图片。")

        image = open_image(upload.stream)
        EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"user_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = EXAMPLE_DIR / filename
        counter = 1
        while save_path.exists():
            filename = f"user_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}.png"
            save_path = EXAMPLE_DIR / filename
            counter += 1

        image.save(save_path, format="PNG")
        return jsonify(
            {
                "name": filename,
                "label": save_path.stem.replace("_", " "),
                "url": f"/static/examples/{filename}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/options")
def options():
    return jsonify(
        {
            "classical_operations": list_classical_operations(),
            "cnn_models": list_visual_models(),
            "cnn_layers": list_visual_layers(),
            "model_status": model_status(),
        }
    )


@app.post("/api/classical")
def classical():
    try:
        image = load_request_image(request, EXAMPLE_DIR)
        operation = request.form.get("operation", "grayscale")
        result = apply_classical_operation(image, operation)
        result_payload = _image_bytes(result.image, "PNG")
        return jsonify(
            {
                "result": image_to_data_url(result.image),
                "operation": operation,
                "title": result.title,
                "description": result.description,
                "kernel": result.kernel,
                "original_width": image.size[0],
                "original_height": image.size[1],
                "result_width": result.image.size[0],
                "result_height": result.image.size[1],
                "original_kb": len(_image_bytes(image, "PNG")) / 1024,
                "result_kb": len(result_payload) / 1024,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/resize-compress")
def resize_compress():
    try:
        image = load_request_image(request, EXAMPLE_DIR)
        method = request.form.get("method", "linear")
        scale = float(request.form.get("scale", "1"))
        target_raw = request.form.get("target_kb", "")
        target_kb = float(target_raw) if target_raw else None
        target_width = _optional_int(request.form.get("target_width", ""))
        target_height = _optional_int(request.form.get("target_height", ""))
        result = apply_resize_compression(
            image,
            method=method,
            scale=scale,
            target_kb=target_kb,
            target_size=(target_width, target_height),
        )
        return jsonify(
            {
                "result": result.data_url,
                "title": result.title,
                "description": result.description,
                "original_width": result.original_size[0],
                "original_height": result.original_size[1],
                "result_width": result.result_size[0],
                "result_height": result.result_size[1],
                "original_kb": result.original_bytes / 1024,
                "result_kb": result.result_bytes / 1024,
                "format": result.fmt,
                "quality": result.quality,
                "target_kb": result.target_kb,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def _image_bytes(image, fmt="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _optional_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(number, 2048))


@app.post("/api/cnn")
def cnn_visualize():
    try:
        image = load_request_image(request, EXAMPLE_DIR)
        model_id = request.form.get("model", "vgg16")
        layer = request.form.get("layer", "conv1_1")
        max_channels = int(request.form.get("channels", "16"))
        max_channels = max(1, min(max_channels, 512))
        tile_size = int(request.form.get("tile_size", "128"))
        tile_size = max(72, min(tile_size, 224))
        top_k = int(request.form.get("top_k", "6"))
        top_k = max(1, min(top_k, 16))

        feature_maps, info = extract_layer_features(image, layer, model_id=model_id)
        grid = feature_maps_to_grid(feature_maps, max_channels=max_channels, cell_size=tile_size)
        mean_map, max_map, summary_info = feature_maps_to_summary_images(feature_maps, cell_size=224)
        top_channels = top_activated_channels(feature_maps, top_k=top_k, cell_size=160)
        total_channels = int(feature_maps.shape[0])
        displayed_channels = min(max_channels, total_channels)

        return jsonify(
            {
                "result": image_to_data_url(grid),
                "mean_result": image_to_data_url(mean_map),
                "max_result": image_to_data_url(max_map),
                "summary": summary_info,
                "top_channels": [
                    {
                        "rank": item["rank"],
                        "channel_index": item["channel_index"],
                        "mean_activation": item["mean_activation"],
                        "max_activation": item["max_activation"],
                        "active_ratio": item["active_ratio"],
                        "image": image_to_data_url(item["image"]),
                    }
                    for item in top_channels
                ],
                "model_id": info["model_id"],
                "layer": layer,
                "channels": displayed_channels,
                "total_channels": total_channels,
                "shape": info["shape"],
                "model": info["model"],
                "message": info.get("message", ""),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "model_status": model_status()}), 400


@app.post("/api/gradcam")
def gradcam_visualize():
    try:
        image = load_request_image(request, EXAMPLE_DIR)
        model_id = request.form.get("model", "resnet18")
        layer = request.form.get("layer", "layer4")
        heatmap, info = grad_cam_heatmap(image, layer, model_id=model_id)
        return jsonify(
            {
                "result": image_to_data_url(heatmap),
                "model": info["model"],
                "model_id": info["model_id"],
                "layer": info["layer"],
                "class_index": info["class_index"],
                "class_label": info["class_label"],
                "score": info["score"],
                "shape": info["shape"],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "model_status": model_status()}), 400


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "source": str(Path(__file__).resolve()),
            "api_version": "feature-summary-topk-v3",
            "has_summary_maps": True,
            "has_class_label": True,
            "has_top_channels": True,
        }
    )


if __name__ == "__main__":
    ensure_example_images(EXAMPLE_DIR)
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Loading server from {Path(__file__).resolve()}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=debug)
