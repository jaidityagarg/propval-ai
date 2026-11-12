"""Draws the PropVal AI logo (assets/logo.png, assets/icon.png). Run once: python make_logo.py"""
from PIL import Image, ImageDraw, ImageFont

TEAL, DARK, GREEN, AMBER, RED = (15, 124, 128), (11, 60, 73), (46, 155, 90), (227, 154, 45), (200, 66, 59)
FONT = "assets/fonts/DejaVuSans-Bold.ttf"
S = 4  # draw large, then shrink for smooth edges


def icon(size):
    im = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = size * S / 100
    w = int(7 * u)
    # roof and walls of a house, open line drawing
    d.line([(12 * u, 46 * u), (50 * u, 12 * u), (88 * u, 46 * u)], fill=TEAL, width=w, joint="curve")
    d.line([(22 * u, 40 * u), (22 * u, 88 * u), (78 * u, 88 * u), (78 * u, 40 * u)], fill=TEAL, width=w, joint="curve")
    # three confidence lights as windows
    for i, c in enumerate([GREEN, AMBER, RED]):
        cx, r = (34 + 16 * i) * u, 6.5 * u
        d.ellipse([cx - r, 62 * u - r, cx + r, 62 * u + r], fill=c)
    return im.resize((size, size), Image.LANCZOS)


def logo(height=160):
    ic = icon(height)
    f1 = ImageFont.truetype(FONT, int(height * 0.46))
    f2 = ImageFont.truetype("assets/fonts/DejaVuSans.ttf", int(height * 0.15))
    w1 = int(f1.getlength("PropVal ")); w2 = int(f1.getlength("AI"))
    tag = "Instant, honest property valuation"
    width = height + 24 + max(w1 + w2, int(f2.getlength(tag))) + 40
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    im.paste(ic, (0, 0), ic)
    d = ImageDraw.Draw(im)
    x = height + 24
    d.text((x, int(height * 0.12)), "PropVal ", font=f1, fill=DARK)
    d.text((x + w1, int(height * 0.12)), "AI", font=f1, fill=TEAL)
    d.text((x + 2, int(height * 0.70)), tag, font=f2, fill=(94, 123, 133))
    return im


if __name__ == "__main__":
    icon(256).save("assets/icon.png")
    logo(160).save("assets/logo.png")
    print("saved assets/icon.png, assets/logo.png")
