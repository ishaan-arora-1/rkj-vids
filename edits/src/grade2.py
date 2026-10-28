"""v2 look: full frame, real-time, straight from the camera files. Clean Lanczos upscale,
colourist-style grade (deeper contrast, carpet pulled down), optional light pass. No vignette.
Usage: python3 -I grade2.py <shot_id>  -> h<id>.mkv (lossless)"""
import sys, subprocess, numpy as np
from PIL import Image, ImageFilter

W, H = 1080, 1920
SW, SH = 720, 1280
SRC = "/home/user/rkj-vids/IMG_{}.MP4"

# clip, in-point, duration, light pass window (fraction of shot) or None
SHOTS = {
    "1": ("0609", 0.00, 3.40, None),
    "2": ("0613", 4.00, 3.00, None),
    "3": ("0625", 5.50, 3.40, None),
    "4": ("0627", 1.40, 3.00, None),
    "5": ("0614", 0.00, 3.00, None),
    "6": ("0621", 3.00, 3.90, (0.20, 0.85)),
}


def read(clip, ss, dur):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", SRC.format(clip), "-ss", str(ss), "-t", str(dur),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = SW * SH * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(SH, SW, 3)
    p.wait()


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
xn, yn = xx / W, yy / H


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def grade(img, t, sweep):
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # qualifier: brown carpet / table (warm, low saturation, mid-dark) -> deeper and quieter
    warm = np.clip((r - b) * 6, 0, 1) * np.clip((r - g + 0.08) * 6, 0, 1)
    q = warm * np.clip((0.42 - sat) / 0.15, 0, 1) * np.clip((0.55 - lum) / 0.15, 0, 1) * np.clip((lum - 0.04) / 0.06, 0, 1)
    q = q[..., None]
    x = x * (1 - 0.45 * q)                                            # darken
    l2 = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = x * (1 - 0.5 * q) + l2 * 0.5 * q                              # and desaturate
    # box red: a touch deeper, less pink
    redness = np.clip(x[..., 0] - np.maximum(x[..., 1], x[..., 2]) - 0.25, 0, 1)
    x[..., 0] -= 0.06 * redness
    # contrast: gentle S-curve, true blacks, protected highlights
    x = np.clip((x - 0.025) / 0.975, 0, 1)
    x = x * 0.93
    s = x * x * (3 - 2 * x)
    x = x * 0.6 + s * 0.4
    # warm highlights slightly
    l3 = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    hi = np.clip((l3 - 0.5) / 0.5, 0, 1)[..., None]
    x = x * (1 + hi * np.array([0.03, 0.01, -0.025], np.float32))
    if sweep and sweep[0] <= t <= sweep[1]:
        k = (t - sweep[0]) / (sweep[1] - sweep[0])
        pos = -0.4 + 1.8 * ease(k)
        band = np.exp(-(((xn * 0.7 - yn * 0.45) - pos + 0.2) / 0.12) ** 2)
        x = x * (1 + (0.28 * np.sin(np.pi * k)) * band)[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main(sid):
    clip, ss, dur, sweep = SHOTS[sid]
    frames = list(read(clip, ss, dur))
    n = len(frames)
    out = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "ffv1", f"h{sid}.mkv"],
                           stdin=subprocess.PIPE)
    for i, fr in enumerate(frames):
        im = Image.fromarray(fr).resize((W, H), Image.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=55, threshold=2))
        out.stdin.write(grade(np.asarray(im), i / max(n - 1, 1), sweep).tobytes())
    out.stdin.close()
    out.wait()
    print(sid, n)


if __name__ == "__main__":
    main(sys.argv[1])
