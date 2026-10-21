"""
One-time generator for the app icon / favicon (PIL-drawn leaf on green).
Run once:  python tools/make_icons.py
Outputs frontend/app/icon.png + icon-192/512 (PWA). Files are committed;
this script documents how they were made and can regenerate them.
"""

from PIL import Image, ImageDraw

SIZE = 512
BASE, DARK, LIGHT = (63, 140, 55), (30, 80, 30), (235, 250, 235)

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded green square background
d.rounded_rectangle([16, 16, SIZE - 16, SIZE - 16], radius=96, fill=BASE)

# Simple leaf: ellipse rotated visually via polygon-ish drawing
leaf = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
ld = ImageDraw.Draw(leaf)
ld.ellipse([100, 140, 412, 372], fill=LIGHT)
img.alpha_composite(leaf)

# Central vein
d.line([150, 256, 380, 256], fill=DARK, width=14)
for i in range(4):
    x = 190 + i * 48
    d.line([x, 256, x + 30, 200], fill=DARK, width=9)
    d.line([x, 256, x + 30, 312], fill=DARK, width=9)

img.save("frontend/app/icon.png", "PNG")
img.resize((192, 192), Image.LANCZOS).save("frontend/public/icon-192.png", "PNG")
img.resize((512, 512), Image.LANCZOS).save("frontend/public/icon-512.png", "PNG")
print("icons written")
