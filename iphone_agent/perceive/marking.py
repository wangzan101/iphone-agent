from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from iphone_agent.perceive.elements import Element


def draw_marks(image: Image.Image, elements: list["Element"]) -> Image.Image:
    """在原图上画框与编号角标：小字、半透明底、放框左上角。"""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font_size = max(12, base.height // 90)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()
    for e in elements:
        x1, y1, x2, y2 = e.box
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=2)
        label = str(e.id)
        tw = d.textlength(label, font=font)
        th = font_size
        d.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=(255, 0, 0, 170))
        d.text((x1 + 2, y1 - th - 1), label, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")
