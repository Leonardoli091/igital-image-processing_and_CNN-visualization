import math

import numpy as np
from PIL import Image, ImageDraw


def feature_maps_to_grid(feature_maps, max_channels: int = 16, cell_size: int = 112) -> Image.Image:
    total_channels = int(feature_maps.shape[0])
    count = max(1, min(max_channels, total_channels))
    columns = min(8, count)
    rows = math.ceil(count / columns)
    label_h = 20
    gap = 8
    width = columns * cell_size + (columns + 1) * gap
    height = rows * (cell_size + label_h) + (rows + 1) * gap

    grid = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(grid)

    for index in range(count):
        row = index // columns
        col = index % columns
        x = gap + col * (cell_size + gap)
        y = gap + row * (cell_size + label_h + gap)

        fmap = feature_maps[index].numpy()
        fmap = normalize_feature(fmap)
        tile = Image.fromarray(fmap, "L").resize((cell_size, cell_size), Image.Resampling.BICUBIC).convert("RGB")
        grid.paste(tile, (x, y + label_h))
        draw.text((x + 4, y + 2), f"ch {index + 1}", fill=(55, 65, 81))

    return grid


def feature_maps_to_summary_images(feature_maps, cell_size: int = 224):
    maps = feature_maps.numpy().astype(np.float32)
    mean_map = maps.mean(axis=0)
    positive_maps = np.maximum(maps, 0.0)
    channel_scores = positive_maps.mean(axis=(1, 2))
    max_channel_index = int(channel_scores.argmax())
    max_channel_map = maps[max_channel_index]
    return (
        _feature_to_image(mean_map, cell_size),
        _feature_to_image(max_channel_map, cell_size),
        {
            "max_channel_index": max_channel_index,
            "max_channel_score": float(channel_scores[max_channel_index]),
        },
    )


def top_activated_channels(feature_maps, top_k: int = 6, cell_size: int = 160):
    maps = feature_maps.numpy().astype(np.float32)
    positive_maps = np.maximum(maps, 0.0)
    total_channels = int(maps.shape[0])
    count = max(1, min(int(top_k), total_channels))
    mean_scores = positive_maps.mean(axis=(1, 2))
    max_scores = positive_maps.max(axis=(1, 2))
    active_ratios = (positive_maps > 1e-6).mean(axis=(1, 2))
    top_indices = np.argsort(-mean_scores)[:count]

    results = []
    for rank, channel_index in enumerate(top_indices, start=1):
        index = int(channel_index)
        results.append(
            {
                "rank": rank,
                "channel_index": index,
                "mean_activation": float(mean_scores[index]),
                "max_activation": float(max_scores[index]),
                "active_ratio": float(active_ratios[index]),
                "image": _feature_to_image(maps[index], cell_size),
            }
        )
    return results


def _feature_to_image(feature: np.ndarray, cell_size: int) -> Image.Image:
    normalized = normalize_feature(feature)
    return Image.fromarray(normalized, "L").resize((cell_size, cell_size), Image.Resampling.BICUBIC).convert("RGB")


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32)
    low = float(feature.min())
    high = float(feature.max())
    if high - low < 1e-6:
        return np.zeros(feature.shape, dtype=np.uint8)
    return ((feature - low) / (high - low) * 255).astype(np.uint8)
