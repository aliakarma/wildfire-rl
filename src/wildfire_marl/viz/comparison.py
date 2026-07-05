"""Visualization tools for side-by-side policy comparisons."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def generate_side_by_side_comparison(
    img_left: Image.Image,
    img_right: Image.Image,
    label_left: str,
    label_right: str,
    save_path: str | Path,
):
    """Combines two rollout frames side-by-side with clear text headers."""
    w1, h1 = img_left.size
    w2, h2 = img_right.size

    # Target dimensions
    max_h = max(h1, h2)
    total_w = w1 + w2

    # Create combined image with some extra space at the top for title
    title_height = 40
    combined = Image.new("RGBA", (total_w, max_h + title_height), "#FFFFFF")

    # Paste images
    combined.paste(img_left, (0, title_height))
    combined.paste(img_right, (w1, title_height))

    # Draw headers
    draw = ImageDraw.Draw(combined)

    # Use standard fonts or draw rectangle labels
    draw.rectangle([0, 0, total_w, title_height], fill="#1E1E1E")

    # Simple drawing logic (no external font dependency to avoid OS crashes)
    draw.text((w1 // 2, title_height // 2), label_left, fill="#FFFFFF", anchor="mm")
    draw.text((w1 + w2 // 2, title_height // 2), label_right, fill="#FFFFFF", anchor="mm")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(save_path)
