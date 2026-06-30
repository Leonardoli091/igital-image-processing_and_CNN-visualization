import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def open_image(file_obj) -> Image.Image:
    image = Image.open(file_obj)
    image = image.convert("RGB")
    return limit_image_size(image)


def limit_image_size(image: Image.Image, max_side: int = 768) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_request_image(request, example_dir: Path) -> Image.Image:
    if "image" in request.files and request.files["image"].filename:
        return open_image(request.files["image"].stream)

    example_name = request.form.get("example", "")
    if not example_name:
        raise ValueError("请选择示例图片或上传图片。")

    safe_name = Path(example_name).name
    path = example_dir / safe_name
    if path.suffix.lower() not in ALLOWED_EXTENSIONS or not path.exists():
        raise ValueError("示例图片不存在。")

    return open_image(path)


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def ensure_example_images(example_dir: Path) -> None:
    example_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "gradient_shapes.png": _make_gradient_shapes,
        "noisy_blocks.png": _make_noisy_blocks,
        "edge_scene.png": _make_edge_scene,
    }

    for filename, factory in examples.items():
        path = example_dir / filename
        if not path.exists():
            factory().save(path)


def _make_gradient_shapes(size: int = 420) -> Image.Image:
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(70 + 130 * xx, 0, 255)
    rgb[:, :, 1] = np.clip(90 + 120 * yy, 0, 255)
    rgb[:, :, 2] = np.clip(120 + 70 * (1 - xx), 0, 255)

    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((55, 70, 180, 250), fill=(230, 85, 80, 210))
    draw.ellipse((210, 65, 355, 210), fill=(70, 150, 210, 220))
    draw.polygon([(110, 330), (215, 205), (330, 335)], fill=(250, 185, 70, 220))
    draw.line((40, 45, 380, 375), fill=(30, 35, 45, 180), width=8)
    return image


def _make_noisy_blocks(size: int = 420) -> Image.Image:
    rng = np.random.default_rng(7)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:] = (180, 184, 174)
    base[55:190, 60:210] = (80, 135, 190)
    base[120:310, 240:365] = (210, 115, 90)
    base[255:370, 55:190] = (235, 195, 90)
    noise = rng.normal(0, 24, base.shape)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(noisy, "RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=0.2))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((60, 55, 210, 190), outline=(20, 35, 55, 220), width=4)
    draw.rectangle((240, 120, 365, 310), outline=(60, 35, 30, 220), width=4)
    draw.rectangle((55, 255, 190, 370), outline=(80, 65, 25, 220), width=4)
    return image


def _make_edge_scene(size: int = 420) -> Image.Image:
    image = Image.new("RGB", (size, size), (226, 230, 222))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 285, size, size), fill=(150, 170, 150))
    draw.rectangle((65, 150, 205, 300), fill=(184, 126, 92))
    draw.polygon([(45, 150), (135, 70), (225, 150)], fill=(108, 91, 78))
    draw.rectangle((95, 215, 135, 300), fill=(70, 73, 83))
    draw.rectangle((155, 180, 188, 220), fill=(230, 217, 150))
    draw.rectangle((260, 110, 355, 300), fill=(93, 135, 174))
    draw.polygon([(245, 110), (308, 55), (370, 110)], fill=(74, 83, 96))
    draw.ellipse((295, 170, 330, 205), fill=(235, 205, 90))
    draw.line((10, 300, 410, 300), fill=(55, 70, 60), width=3)
    return image
