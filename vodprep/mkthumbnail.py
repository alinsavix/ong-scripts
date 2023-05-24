#!/usr/bin/env python3
import sys

from PIL import Image, ImageColor, ImageDraw, ImageFont

TEXT_CENTER=606
TEXT_BASELINE=650
TEXT_SIZE=220  # points
# our text centerline will be 606 pixels

text = sys.argv[1]

img = Image.open("thumbnail_template.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("mtcorsva_0.ttf", TEXT_SIZE)

size_x = draw.textlength(text, font=font)
centering_x = TEXT_CENTER - (size_x / 2)

draw.text((centering_x, TEXT_BASELINE), text, font=font, fill="#fe9a0d")
# img.show()
img.save(f"{text}.jpg", quality=95)
