"""Per-shot look: reframe + slow push-in, sculpted spotlight, filmic grade, bloom, grain.
Usage: python3 -I grade.py <shot_id>   (reads sN.mp4, writes gN.mkv lossless)"""
import sys, subprocess, numpy as np
from PIL import Image, ImageFilter

W, H = 1080, 1920
SW, SH = 720, 1280

# cx, cy: framing centre (normalised source coords), keyframed start->end
# z: zoom start->end, spot: light centre start->end, r: light radii (x, y) normalised
# gobo: soft window-shadow stripes, sweep: travelling light pass
SHOTS = {
    "s1": dict(c=((0.50, 0.50), (0.52, 0.48)), z=(1.10, 1.20), spot=((0.42, 0.40), (0.55, 0.46)),
               r=(0.62, 0.36), floor=0.10, gobo=True, sweep=None, seed=1, exp=0.72, sat=0.82),
    "s2": dict(c=((0.50, 0.55), (0.50, 0.58)), z=(1.08, 1.16), spot=((0.50, 0.55), (0.50, 0.62)),
               r=(0.55, 0.30), floor=0.08, gobo=False, sweep=(0.35, 0.95), seed=2, exp=0.9, sat=0.92),
    "s3": dict(c=((0.50, 0.56), (0.49, 0.56)), z=(1.20, 1.30), spot=((0.50, 0.57), (0.49, 0.56)),
               r=(0.34, 0.20), floor=0.06, gobo=False, sweep=None, seed=3),
    "s4": dict(c=((0.50, 0.60), (0.52, 0.60)), z=(1.20, 1.30), spot=((0.49, 0.61), (0.52, 0.60)),
               r=(0.34, 0.20), floor=0.06, gobo=False, sweep=None, seed=4),
    "s5": dict(c=((0.50, 0.57), (0.50, 0.58)), z=(1.12, 1.22), spot=((0.50, 0.57), (0.50, 0.58)),
               r=(0.46, 0.24), floor=0.07, gobo=False, sweep=None, seed=5),
    "s6": dict(c=((0.50, 0.57), (0.50, 0.57)), z=(1.04, 1.12), spot=((0.50, 0.57), (0.50, 0.58)),
               r=(0.85, 0.22), floor=0.07, gobo=False, sweep=None, seed=6),
    "s7": dict(c=((0.50, 0.62), (0.50, 0.60)), z=(1.08, 1.18), spot=((0.48, 0.66), (0.50, 0.60)),
               r=(0.52, 0.26), floor=0.07, gobo=False, sweep=(0.15, 0.75), seed=7, sat=0.95, exp=0.88),
}


def ease(t):
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def read_frames(path):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = SW * SH * 3
    frames = []
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        frames.append(np.frombuffer(b, np.uint8).reshape(SH, SW, 3))
    p.wait()
    return frames


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
xn, yn = xx / W, yy / H


def light_mask(p, t, fi):
    sx = lerp(p["spot"][0][0], p["spot"][1][0], t)
    sy = lerp(p["spot"][0][1], p["spot"][1][1], t)
    rx, ry = p["r"]
    d2 = ((xn - sx) / rx) ** 2 + ((yn - sy) / ry) ** 2
    spot = np.exp(-d2 * 1.1)
    m = p["floor"] + (1 - p["floor"]) * spot
    if p["gobo"]:
        # soft diagonal shadows from an unseen window, drifting slowly
        u = (xn * 0.8 + yn * 0.55) * 2.1 + t * 0.35
        stripes = 0.5 + 0.5 * np.sin(u * 2 * np.pi)
        m *= 1 - 0.30 * stripes ** 3
    if p["sweep"]:
        a, b = p["sweep"]
        if a <= t <= b:
            k = (t - a) / (b - a)
            pos = lerp(-0.4, 1.4, ease(k))
            band = np.exp(-(((xn * 0.7 - yn * 0.45) - pos + 0.2) / 0.11) ** 2)
            m *= 1 + 0.45 * band * np.sin(np.pi * k)
    return m


def grade(img, mask, rng, exp=1.0, sat=1.0):
    x = img.astype(np.float32) / 255.0
    x = x ** 1.08
    x *= mask[..., None] * exp
    l0 = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = l0 + (x - l0) * sat
    # tame the box red into oxblood, keep gem reds pure: pull saturated mid reds down a touch
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    redness = np.clip(x[..., 0] - np.maximum(x[..., 1], x[..., 2]) - 0.25, 0, 1)
    mid = np.clip(1 - np.abs(lum - 0.28) / 0.28, 0, 1)
    x[..., 0] -= 0.10 * redness * mid
    # filmic S-curve with crushed blacks and rolled highlights
    x = np.clip((x - 0.018) / 0.982, 0, None)
    x = x / (x + 0.55) * 1.55
    x = np.clip(x, 0, 1)
    x = x * x * (3 - 2 * x) * 0.55 + x * 0.45
    # warm highlights, neutral-deep shadows
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    hi = np.clip((lum - 0.45) / 0.55, 0, 1)[..., None]
    x = x * (1 + hi * np.array([0.035, 0.012, -0.03], np.float32))
    # bloom on diamonds and gold
    bright = np.clip(x - 0.72, 0, 1)
    bi = Image.fromarray((bright * 255 * 3).clip(0, 255).astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR)
    b1 = np.asarray(bi.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR), np.float32) / 255 / 3
    b2 = np.asarray(bi.filter(ImageFilter.GaussianBlur(22)).resize((W, H), Image.BILINEAR), np.float32) / 255 / 3
    x = x + (b1 * 0.45 + b2 * 0.35) * np.array([1.0, 0.94, 0.84], np.float32)
    # fine grain
    g = rng.normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
    g = np.asarray(Image.fromarray(g, mode="F").resize((W, H), Image.BILINEAR))
    x = x + g[..., None] * (0.4 + 0.6 * (1 - np.clip(lum, 0, 1)))[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def reframe(src, cx, cy, z):
    s = 1.5 * z  # source -> output scale
    # keep the crop inside the frame
    hw, hh = W / s / 2, H / s / 2
    px = np.clip(cx * SW, hw, SW - hw)
    py = np.clip(cy * SH, hh, SH - hh)
    a, c = 1 / s, px - hw
    e, f = 1 / s, py - hh
    im = Image.fromarray(src).transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), Image.BICUBIC)
    return np.asarray(im)


def main(sid):
    p = SHOTS[sid]
    frames = read_frames(f"{sid}.mp4")
    n = len(frames)
    rng = np.random.default_rng(p["seed"])
    out = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "ffv1", f"g{sid[1:]}.mkv"],
                           stdin=subprocess.PIPE)
    for i, fr in enumerate(frames):
        t = i / max(n - 1, 1)
        te = ease(t)
        cx = lerp(p["c"][0][0], p["c"][1][0], te)
        cy = lerp(p["c"][0][1], p["c"][1][1], te)
        z = lerp(p["z"][0], p["z"][1], t)  # linear push reads as a camera move, not an effect
        img = reframe(fr, cx, cy, z)
        out.stdin.write(grade(img, light_mask(p, t, i), rng, p.get("exp", 1.0), p.get("sat", 1.0)).tobytes())
    out.stdin.close()
    out.wait()
    print(sid, n)


if __name__ == "__main__":
    main(sys.argv[1])
