"""Tiles stills 2x2 into sheets for review: python sheet.py out/sheets a.png b.png ..."""
import os
import sys

from PIL import Image

out, files = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
for k in range(0, len(files), 4):
    sheet = Image.new("RGB", (1920, 1080), (40, 40, 40))
    for j, f in enumerate(files[k:k + 4]):
        im = Image.open(f).convert("RGB").resize((958, 538))
        sheet.paste(im, ((j % 2) * 962, (j // 2) * 542))
    p = os.path.join(out, f"sheet{k // 4:02d}.png")
    sheet.save(p)
    print(p)
