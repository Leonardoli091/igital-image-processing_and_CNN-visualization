import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ResizeCompressResult:
    image: Image.Image
    data_url: str
    title: str
    description: str
    original_size: Tuple[int, int]
    result_size: Tuple[int, int]
    original_bytes: int
    result_bytes: int
    fmt: str
    quality: Optional[int]
    target_kb: Optional[float]


METHOD_LABELS = {
    "linear": "线性插值放缩",
    "pca": "主成分分析低秩压缩",
    "factor": "因子分析低秩近似",
}


def apply_resize_compression(
    image: Image.Image,
    method: str = "linear",
    scale: float = 1.0,
    target_kb: Optional[float] = None,
    target_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> ResizeCompressResult:
    method = method if method in METHOD_LABELS else "linear"
    scale = max(0.1, min(float(scale), 4.0))
    target_kb = _clean_target_kb(target_kb)

    source = image.convert("RGB")
    original_size = source.size
    original_bytes = len(_encode_image(source, "PNG"))

    resized = _resize_linear(source, scale, target_size=target_size)
    if method == "pca":
        processed = _low_rank_image(resized, mode="pca", target_kb=target_kb)
    elif method == "factor":
        processed = _low_rank_image(resized, mode="factor", target_kb=target_kb)
    else:
        processed = resized

    payload, fmt, quality = _encode_for_target(processed, target_kb)
    data_url = _bytes_to_data_url(payload, fmt)
    title = METHOD_LABELS[method]
    description = _build_description(method, scale, target_kb, fmt, quality)
    return ResizeCompressResult(
        image=processed,
        data_url=data_url,
        title=title,
        description=description,
        original_size=original_size,
        result_size=processed.size,
        original_bytes=original_bytes,
        result_bytes=len(payload),
        fmt=fmt,
        quality=quality,
        target_kb=target_kb,
    )


def _resize_linear(
    image: Image.Image,
    scale: float,
    target_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Image.Image:
    width, height = image.size
    new_width, new_height = _resolve_target_size(width, height, scale, target_size)
    max_side = 2048
    longest = max(new_width, new_height)
    if longest > max_side:
        ratio = max_side / longest
        new_width = max(1, int(round(new_width * ratio)))
        new_height = max(1, int(round(new_height * ratio)))
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def _resolve_target_size(
    width: int,
    height: int,
    scale: float,
    target_size: Optional[Tuple[Optional[int], Optional[int]]],
) -> Tuple[int, int]:
    target_width = None
    target_height = None
    if target_size:
        target_width, target_height = target_size

    if target_width and target_height:
        return max(1, int(target_width)), max(1, int(target_height))
    if target_width:
        ratio = int(target_width) / width
        return max(1, int(target_width)), max(1, int(round(height * ratio)))
    if target_height:
        ratio = int(target_height) / height
        return max(1, int(round(width * ratio))), max(1, int(target_height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _low_rank_image(image: Image.Image, mode: str, target_kb: Optional[float]) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    components = _component_count(width, height, target_kb)
    channels = []
    for channel_index in range(3):
        channel = arr[:, :, channel_index]
        if mode == "pca":
            mean = channel.mean(axis=0, keepdims=True)
            base = channel - mean
            reconstructed = _truncated_svd(base, components) + mean
        else:
            reconstructed = _truncated_svd(channel, components)
        channels.append(np.clip(reconstructed, 0, 255))
    out = np.stack(channels, axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def _truncated_svd(matrix: np.ndarray, components: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = max(1, min(components, s.shape[0]))
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def _component_count(width: int, height: int, target_kb: Optional[float]) -> int:
    base = min(width, height)
    if target_kb:
        estimate = int((target_kb * 1024) / max(width + height, 1))
        return max(4, min(96, estimate))
    return max(8, min(64, base // 10))


def _encode_for_target(image: Image.Image, target_kb: Optional[float]) -> Tuple[bytes, str, Optional[int]]:
    if not target_kb:
        return _encode_image(image, "PNG"), "PNG", None

    target_bytes = int(target_kb * 1024)
    best_payload = None
    best_quality = None
    low, high = 5, 95
    while low <= high:
        quality = (low + high) // 2
        payload = _encode_jpeg(image, quality)
        if len(payload) <= target_bytes:
            best_payload = payload
            best_quality = quality
            low = quality + 1
        else:
            high = quality - 1

    if best_payload is None:
        best_quality = 5
        best_payload = _encode_jpeg(image, best_quality)
    return best_payload, "JPEG", best_quality


def _encode_image(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _bytes_to_data_url(payload: bytes, fmt: str) -> str:
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _clean_target_kb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return max(5.0, min(number, 4096.0))


def _build_description(method: str, scale: float, target_kb: Optional[float], fmt: str, quality: Optional[int]) -> str:
    parts = [f"尺寸比例 {scale:g}x"]
    if method == "linear":
        parts.append("使用双线性插值计算新像素")
    elif method == "pca":
        parts.append("对 RGB 各通道做中心化 SVD，并保留主要主成分")
    else:
        parts.append("对 RGB 各通道做低秩因子近似，保留主要结构")
    if target_kb:
        parts.append(f"按目标 {target_kb:g} KB 输出 JPEG，质量约 {quality}")
    else:
        parts.append(f"输出 {fmt}")
    return "；".join(parts) + "。"
