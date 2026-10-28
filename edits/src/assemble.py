"""Cut graded shots together with dissolves, set type, add end card. Silent master.
Usage: python3 -I assemble.py <out.mp4> [logo.png] [--stills]"""
import sys, os, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
D = 8                      # dissolve length (frames)
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")
IVORY = (238, 229, 211)

SHOTS = [f"g{i}.mkv" for i in range(1, 8)]

LINES = [
    # (text, first shot idx, last shot idx, italic, y) -- y sits on clean velvet, clear of the latch
    ("Real rubies. Real emeralds.", 2, 3, False, 0.195),
    ("18 karat gold. Certified stones.", 5, 5, False, 0.70),
]
TAGLINE = "Inspired by Parisian high jewellery."
END_LEN = 105
BLACK_GAP = 8
TAIL_FADE = 18


def read(path):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = W * H * 3
    out = []
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
    p.wait()
    return out


def font(size, italic=False, weight=400):
    name = f"CG{'Italic' if italic else ''}-{weight}.ttf"
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def text_layer(text, f, tracking_em, cy):
    """RGBA layer with tracked text centred at cy, plus soft shadow for legibility."""
    size = f.size
    track = tracking_em * size
    widths = [f.getlength(ch, features=["lnum"]) for ch in text]
    total = sum(widths) + track * (len(text) - 1)
    x = (W - total) / 2
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    for ch, w in zip(text, widths):
        d.text((x, cy), ch, font=f, fill=255, anchor="lm", features=["lnum"])
        x += w + track
    shadow = layer.filter(ImageFilter.GaussianBlur(10))
    return np.asarray(layer, np.float32) / 255, np.asarray(shadow, np.float32) / 255


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def comp_text(frame, a, layer, shadow):
    x = frame.astype(np.float32)
    x *= 1 - (shadow * 0.55 * a)[..., None]
    x = x * (1 - (layer * a)[..., None]) + np.array(IVORY, np.float32) * (layer * a)[..., None]
    return x


def end_card(logo, k, rng):
    """k: frame index within end card."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H * 0.48) / (H * 0.30)) ** 2))
    base = np.array([10, 8, 7], np.float32) + glow[..., None] * np.array([26, 19, 13], np.float32)
    return base


def main():
    outp = sys.argv[1]
    logo_path = next((a for a in sys.argv[2:] if a.endswith(".png")), None)
    stills = "--stills" in sys.argv
    shots = [read(s) for s in SHOTS]
    starts, t = [], 0
    for s in shots:
        starts.append(t)
        t += len(s) - D
    body_len = t + D
    total = body_len + BLACK_GAP + END_LEN

    # timeline lookup
    def body_frame(i):
        acc = None
        for k, s in enumerate(shots):
            j = i - starts[k]
            if 0 <= j < len(s):
                f = s[j].astype(np.float32)
                w = 1.0
                if k > 0 and j < D:
                    w = ease((j + 1) / (D + 1))
                acc = f if acc is None else acc * (1 - w) + f * w
        return acc

    # text events
    body_font = font(54, False, 400)
    events = []
    for text, a, b, it, y in LINES:
        s0 = starts[a] + 14
        s1 = starts[b] + len(shots[b]) - D - 6
        events.append((text, s0, s1, font(56, it, 400), y))

    tag_font = font(62, True, 400)
    logo = None
    if logo_path:
        lg = Image.open(logo_path).convert("RGBA")
        sc = min(560 / lg.width, 360 / lg.height)
        lg = lg.resize((int(lg.width * sc), int(lg.height * sc)), Image.LANCZOS)
        logo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        logo.paste(lg, ((W - lg.width) // 2, int(H * 0.43) - lg.height // 2), lg)
        logo = np.asarray(logo, np.float32) / 255
    tag_y = H * 0.56 if logo is not None else H * 0.50
    card_base = end_card(None, 0, None)
    rng = np.random.default_rng(42)

    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "16", "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                            "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", "-an", outp],
                           stdin=subprocess.PIPE)
    still_at = {starts[k] + len(shots[k]) // 2: k for k in range(len(shots))}
    for i in range(total):
        if i < body_len:
            x = body_frame(i)
            if i < 10:
                x *= ease((i + 1) / 10)
            if i >= body_len - TAIL_FADE:
                x *= 1 - ease((i - (body_len - TAIL_FADE) + 1) / TAIL_FADE)
            for text, s0, s1, f, y in events:
                if s0 <= i <= s1:
                    life = (i - s0) / (s1 - s0)
                    a = ease((i - s0) / 18) * (1 - ease((i - (s1 - 15)) / 15))
                    L, S = text_layer(text, f, 0.035 + 0.025 * life, H * y)
                    x = comp_text(x, a, L, S)
        elif i < body_len + BLACK_GAP:
            x = np.zeros((H, W, 3), np.float32)
        else:
            k = i - body_len - BLACK_GAP
            x = card_base * ease((k + 1) / 24)
            if logo is not None:
                a = ease((k - 6) / 24) * logo[..., 3]
                x = x * (1 - a[..., None]) + logo[..., :3] * 255 * a[..., None]
            a = ease((k - 20) / 22)
            L, S = text_layer(TAGLINE, tag_font, 0.02 + 0.02 * k / END_LEN, tag_y)
            x = comp_text(x, a, L, S)
            g = rng.normal(0, 2.2, (H // 2, W // 2)).astype(np.float32)
            x = x + np.asarray(Image.fromarray(g, mode="F").resize((W, H), Image.BILINEAR))[..., None]
        fr = np.clip(x + 0.5, 0, 255).astype(np.uint8)
        enc.stdin.write(fr.tobytes())
        if stills and (i in still_at or i in [e[1] + 30 for e in events] + [e[2] - 20 for e in events] or i == total - 10):
            Image.fromarray(fr).save(f"still_{i:04d}.jpg", quality=90)
    enc.stdin.close()
    enc.wait()
    print("frames", total, "seconds", round(total / FPS, 2))


if __name__ == "__main__":
    main()
