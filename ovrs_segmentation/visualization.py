from __future__ import annotations

from typing import Mapping, Sequence
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PALETTE = {
    0: (0, 0, 0),
    1: (230, 25, 75),
    2: (0, 130, 200),
    3: (255, 225, 25),
    4: (245, 130, 48),
    5: (60, 180, 75),
    6: (145, 30, 180),
    255: (128, 128, 128),
}

OPENEARTHMAP_CLASSES = {
    1: "bareland",
    2: "rangeland",
    3: "developed space",
    4: "road",
    5: "tree",
    6: "water",
    7: "agriculture land",
    8: "building",
}

OPENEARTHMAP_PALETTE = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 255, 36),
    3: (148, 148, 148),
    4: (255, 255, 255),
    5: (34, 97, 38),
    6: (0, 69, 255),
    7: (75, 181, 73),
    8: (222, 31, 7),
    255: (0, 0, 0),
}


def palette_for_classes(class_text_by_id: Mapping[int, str]) -> Mapping[int, Sequence[int]]:
    """Return an official palette when the supplied ontology identifies one."""
    normalized = {
        int(class_id): " ".join(str(class_text).strip().lower().split())
        for class_id, class_text in class_text_by_id.items()
    }
    if all(normalized.get(class_id) == name for class_id, name in OPENEARTHMAP_CLASSES.items()):
        return OPENEARTHMAP_PALETTE
    return DEFAULT_PALETTE


def colorize_mask(mask: np.ndarray, palette: Mapping[int, Sequence[int]] = DEFAULT_PALETTE) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        numeric_id = int(class_id)
        color = palette.get(numeric_id)
        if color is None:
            hue = (numeric_id * 0.618033988749895) % 1.0
            color = tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.72, 0.95))
        output[mask == class_id] = color
    return output


def overlay(image: np.ndarray, color_mask: np.ndarray, active: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    output = image.copy()
    output[active] = np.clip(
        (1 - alpha) * output[active].astype(float) + alpha * color_mask[active].astype(float), 0, 255
    ).astype(np.uint8)
    return output


def render_attention_overlay(
    image_rgb: np.ndarray,
    attention: np.ndarray,
    label: str,
    *,
    class_id: int | None = None,
    score: float | None = None,
    alpha: float = 0.62,
    image_brightness: float = 0.92,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a query-conditioned spatial map, optionally percentile-normalized."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    attention = np.asarray(attention, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H, W, 3], got {image_rgb.shape}")
    if attention.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Image/attention shape mismatch: {image_rgb.shape[:2]} versus "
            f"{attention.shape}"
        )
    if not np.isfinite(attention).all():
        raise ValueError("attention contains NaN or Inf")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 <= image_brightness <= 1.0:
        raise ValueError("image_brightness must be between 0 and 1")

    if normalize:
        lower, upper = np.percentile(attention, (2.0, 98.0))
        if upper <= lower + 1e-8:
            normalized = np.zeros_like(attention, dtype=np.float32)
        else:
            normalized = np.clip((attention - lower) / (upper - lower), 0.0, 1.0)
    else:
        # Preserve calibrated sigmoid magnitude for adapted Scene inference.
        normalized = np.clip(attention, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8),
        getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    )
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    darkened = np.clip(
        image_rgb.astype(np.float32) * image_brightness, 0, 255
    )
    # Low-attention areas retain the source image; high-attention regions receive
    # the strongest color overlay.
    opacity = (alpha * normalized)[..., None]
    fused = np.clip(
        (1.0 - opacity) * darkened + opacity * heat_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    canvas = Image.fromarray(fused, mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    height, width = attention.shape
    padding = max(4, min(height, width) // 100)
    font_size = max(10, min(24, width // 40))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    title = "Scene class attention"
    class_label = f"{class_id}: {label}" if class_id is not None else str(label)
    if score is not None:
        class_label += f"  p={float(score):.3f}"
    subtitle = (
        "blue = low, red = high (relative spatial activation)"
        if normalize
        else "blue = 0, red = 1 (raw sigmoid activation)"
    )
    title_box = draw.textbbox((0, 0), title, font=font)
    class_box = draw.textbbox((0, 0), class_label, font=font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=font)
    line_height = max(
        font_size + 5,
        title_box[3] - title_box[1] + 5,
        class_box[3] - class_box[1] + 5,
        subtitle_box[3] - subtitle_box[1] + 5,
    )
    swatch = max(6, min(font_size, line_height - 4))
    box_width = min(
        width,
        max(
            title_box[2] - title_box[0],
            class_box[2] - class_box[0] + swatch + padding,
            subtitle_box[2] - subtitle_box[0],
        )
        + padding * 2,
    )
    box_height = min(height, line_height * 3 + padding * 2)
    top = height - box_height
    draw.rectangle((0, top, box_width, height), fill=(0, 0, 0, 185))
    draw.text((padding, top + padding), title, fill=(255, 255, 255, 255), font=font)
    class_y = top + padding + line_height
    swatch_color = (
        tuple(int(value) for value in colorize_mask(np.asarray([[class_id]]))[0, 0])
        if class_id is not None
        else (255, 255, 255)
    )
    swatch_y = class_y + max(0, (line_height - swatch) // 2)
    draw.rectangle(
        (padding, swatch_y, padding + swatch, swatch_y + swatch),
        fill=(*swatch_color, 255),
    )
    draw.text(
        (padding * 2 + swatch, class_y),
        class_label,
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (padding, top + padding + line_height * 2),
        subtitle,
        fill=(230, 230, 230, 255),
        font=font,
    )
    return np.asarray(canvas)


def render_semantic_overlay(
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    class_text_by_id: Mapping[int, str],
    *,
    alpha: float = 0.62,
    image_brightness: float = 0.82,
    background_id: int | None = 0,
    ignore_id: int = 255,
) -> np.ndarray:
    """Render a strong semantic overlay with a bottom-left class legend."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    semantic_mask = np.asarray(semantic_mask)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H, W, 3], got {image_rgb.shape}")
    if semantic_mask.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Image/mask shape mismatch: {image_rgb.shape[:2]} versus {semantic_mask.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 <= image_brightness <= 1.0:
        raise ValueError("image_brightness must be between 0 and 1")

    palette = palette_for_classes(class_text_by_id)
    darkened = np.clip(image_rgb.astype(np.float32) * image_brightness, 0, 255).astype(np.uint8)
    colored = colorize_mask(semantic_mask, palette)
    active = np.ones(semantic_mask.shape, dtype=bool)
    if background_id is not None:
        active &= semantic_mask != background_id
    active &= semantic_mask != ignore_id
    fused = overlay(darkened, colored, active, alpha=alpha)
    excluded_ids = {ignore_id}
    if background_id is not None:
        excluded_ids.add(background_id)
    present_ids = [
        int(class_id)
        for class_id in np.unique(semantic_mask)
        if int(class_id) not in excluded_ids
        and int(class_id) in class_text_by_id
    ]
    if not present_ids:
        return fused

    height, width = semantic_mask.shape
    padding = max(3, min(height, width) // 100)
    font_size = max(8, min(22, width // 45, max(8, height // (len(present_ids) + 3))))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.fromarray(fused, mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    title = "Classes in this image"
    labels = [f"{class_id}: {class_text_by_id[class_id]}" for class_id in present_ids]
    title_box = draw.textbbox((0, 0), title, font=font)
    text_boxes = [draw.textbbox((0, 0), label, font=font) for label in labels]
    text_height = max(
        title_box[3] - title_box[1],
        *(box[3] - box[1] for box in text_boxes),
    )
    line_height = max(font_size + 5, text_height + 5)
    swatch = max(5, min(font_size, line_height - 4))
    legend_width = min(
        width,
        max(
            title_box[2] - title_box[0] + padding * 2,
            max(box[2] - box[0] for box in text_boxes) + swatch + padding * 3,
        ),
    )
    legend_height = min(height, line_height * (len(labels) + 1) + padding * 2)
    top = height - legend_height
    draw.rectangle((0, top, legend_width, height), fill=(0, 0, 0, 185))
    draw.rectangle((0, top, max(0, legend_width - 1), height - 1), outline=(255, 255, 255, 150), width=1)
    draw.text((padding, top + padding), title, fill=(255, 255, 255, 255), font=font)
    for index, (class_id, label) in enumerate(zip(present_ids, labels)):
        y = top + padding + (index + 1) * line_height
        if y + line_height > height:
            break
        color = tuple(
            int(value)
            for value in colorize_mask(np.asarray([[class_id]]), palette)[0, 0]
        )
        swatch_y = y + max(0, (line_height - swatch) // 2)
        draw.rectangle((padding, swatch_y, padding + swatch, swatch_y + swatch), fill=(*color, 255))
        draw.text((padding * 2 + swatch, y), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(canvas)
