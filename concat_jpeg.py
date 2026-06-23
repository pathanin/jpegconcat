#!/usr/bin/env python3
"""
concat_jpeg.py — Concatenate JPEG images while preserving original encoding parameters.

Usage:
    python3 concat_jpeg.py img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg]
                           [--direction horizontal|vertical|auto]
                           [--order auto|as-given]

Auto-detection (both on by default):
  order:     for 2 images, uses edge-color seam matching; otherwise sorts by
             embedded numeric sequence in filename, then EXIF timestamp, then mtime
  direction: for 2 images, determined by edge-color seam matching; otherwise
             portrait → horizontal, landscape → vertical

Note: The lossless jpegtran fast-path uses -copy none and strips all metadata
(EXIF, ICC profiles, XMP). The Pillow fallback preserves EXIF from the first
source image.
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow --break-system-packages")
    sys.exit(1)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

EDGE_DEPTH = 8    # pixels deep to average from each edge
EDGE_LEN   = 256  # normalize all strips to this length for comparison

_STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

# MCU block dimensions indexed by subsampling value (0=4:4:4, 1=4:2:2, 2=4:2:0)
_MCU_DIMS = {0: (8, 8), 1: (16, 8), 2: (16, 16)}


# ── JPEG parameter detection ──────────────────────────────────────────────────

def _jpeg_params(path):
    """
    Return (qtables, quality, subsampling, is_grayscale) for a JPEG file.

    qtables — Pillow's img.quantization dict (exact tables from libjpeg, suitable
              for canvas.save(qtables=...)), or None if unavailable.
    quality — integer estimate derived from the luma table, for display only.
    subsampling — 0/1/2 (4:4:4 / 4:2:2 / 4:2:0) from the SOF marker.
    is_grayscale — True when the JPEG SOF marker has exactly 1 component.
    """
    qtables = None
    quality = 85
    subsampling = 2
    is_grayscale = False

    # Quality + tables: use Pillow's libjpeg-parsed quantization dict (header-only open)
    try:
        _img = Image.open(path)
        is_grayscale = (_img.mode == "L")
        if _img.quantization:
            qtables = _img.quantization
            tbl = qtables.get(0, [])
            if len(tbl) == 64:
                s = sum(tbl[k] * 100 / _STD_LUMA[k] for k in range(64)) / 64
                q = (200 - s) / 2 if s <= 100 else 5000 / s
                quality = max(1, min(95, round(q)))
    except Exception:
        pass

    # Subsampling: stream the SOF marker (no clean public Pillow API for this)
    try:
        with open(path, "rb") as f:
            if f.read(2) == b'\xff\xd8':
                while True:
                    b = f.read(2)
                    if len(b) < 2 or b[0] != 0xFF:
                        break
                    marker = b[1]
                    if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                        continue
                    if marker == 0xDA:
                        break
                    lb = f.read(2)
                    if len(lb) < 2:
                        break
                    seg = f.read(struct.unpack(">H", lb)[0] - 2)
                    if marker in (0xC0, 0xC1, 0xC2):
                        if len(seg) >= 15 and seg[5] >= 3:
                            y_h,  y_v  = seg[7] >> 4, seg[7] & 0xF
                            cb_h, cb_v = seg[10] >> 4, seg[10] & 0xF
                            if y_h == cb_h and y_v == cb_v:
                                subsampling = 0
                            elif y_v == cb_v:
                                subsampling = 1
                            else:
                                subsampling = 2
                        break
    except Exception:
        pass

    return qtables, quality, subsampling, is_grayscale


# ── jpegtran lossless fast-path ───────────────────────────────────────────────

def _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling):
    """
    Attempt lossless DCT-level concatenation via jpegtran -drop.

    Returns True on success, False if any precondition is unmet.

    NOTE: jpegtran is called with -copy none, so all metadata (EXIF, ICC
    profiles, XMP) is stripped from the output. This is unavoidable with
    the DCT-level compositing approach — there is no source image whose
    metadata accurately describes the composite. Use the Pillow fallback
    path if metadata preservation is required.

    Preconditions:
      - jpegtran (libjpeg-turbo >= 1.4) is in PATH
      - All inputs are JPEG with identical chroma subsampling
      - The perpendicular dimension of each image and every join offset
        are multiples of the MCU block size for that subsampling
    """
    jpegtran = shutil.which("jpegtran")
    if not jpegtran:
        return False

    if not all(fmt == "JPEG" for fmt in input_formats):
        return False

    # All inputs must share the same subsampling as the first image
    for path in image_paths[1:]:
        _, _, s, _ = _jpeg_params(path)
        if s != subsampling:
            return False

    mcu_w, mcu_h = _MCU_DIMS.get(subsampling, (8, 8))

    if direction == "horizontal":
        # Every image's height and every cumulative x-offset must be MCU-aligned
        if not all(img.height % mcu_h == 0 for img in images):
            return False
        x = 0
        for img in images[:-1]:
            x += img.width
            if x % mcu_w != 0:
                return False
    else:  # vertical
        if not all(img.width % mcu_w == 0 for img in images):
            return False
        y = 0
        for img in images[:-1]:
            y += img.height
            if y % mcu_h != 0:
                return False

    with tempfile.TemporaryDirectory() as td:
        # Build a blank black canvas JPEG with the source's quantization tables
        if direction == "horizontal":
            canvas_size = (sum(img.width for img in images), max(img.height for img in images))
        else:
            canvas_size = (max(img.width for img in images), sum(img.height for img in images))

        canvas_path = os.path.join(td, "canvas.jpg")
        blank = Image.new("RGB", canvas_size, (0, 0, 0))
        if qtables:
            blank.save(canvas_path, "JPEG", qtables=qtables, subsampling=subsampling)
        else:
            blank.save(canvas_path, "JPEG", quality=85, subsampling=subsampling)

        # Drop each source image into the canvas at its offset (DCT-level, no re-encode)
        # -copy none: no metadata propagates (see docstring above)
        current = canvas_path
        x = y = 0
        for idx, (path, img) in enumerate(zip(image_paths, images)):
            out_path = os.path.join(td, f"step_{idx}.jpg")
            result = subprocess.run(
                [jpegtran, "-copy", "none", "-drop", f"+{x}+{y}", path, current],
                capture_output=True,
            )
            if result.returncode != 0 or not result.stdout:
                return False
            with open(out_path, "wb") as fout:
                fout.write(result.stdout)
            current = out_path
            if direction == "horizontal":
                x += img.width
            else:
                y += img.height

        shutil.copy(current, output_path)

    return True


# ── Filename / EXIF ordering (fallback) ───────────────────────────────────────

def _sort_key(path, img):
    nums = tuple(int(n) for n in re.findall(r"\d+", os.path.basename(path)))
    try: dt = img.getexif().get(0x9003) or ""
    except Exception: dt = ""
    try: mtime = os.path.getmtime(path)
    except OSError: mtime = 0
    return (nums, dt, mtime)


# ── Edge-color seam matching ──────────────────────────────────────────────────

def _edge_strip(img, side):
    """
    Extract a (EDGE_LEN, 3) float32 mean-color strip from one edge of a PIL Image.
    Crops the EDGE_DEPTH-pixel border before converting to float32, so only a thin
    slice enters memory instead of the full image array.
    """
    w, h = img.size
    if side == "right":
        arr   = np.asarray(img.crop((w - EDGE_DEPTH, 0, w, h)), dtype=np.float32)
        strip = arr.mean(axis=1)   # (H, 3)
    elif side == "left":
        arr   = np.asarray(img.crop((0, 0, EDGE_DEPTH, h)), dtype=np.float32)
        strip = arr.mean(axis=1)   # (H, 3)
    elif side == "bottom":
        arr   = np.asarray(img.crop((0, h - EDGE_DEPTH, w, h)), dtype=np.float32)
        strip = arr.mean(axis=0)   # (W, 3)
    else:  # top
        arr   = np.asarray(img.crop((0, 0, w, EDGE_DEPTH)), dtype=np.float32)
        strip = arr.mean(axis=0)   # (W, 3)

    cur = strip.shape[0]
    if cur != EDGE_LEN:
        idx   = np.linspace(0, cur - 1, EDGE_LEN)
        lo    = np.floor(idx).astype(int)
        hi    = np.minimum(lo + 1, cur - 1)
        t     = (idx - lo)[:, None]
        strip = strip[lo] * (1 - t) + strip[hi] * t
    return strip


def _seam_mad(img_a, side_a, img_b, side_b):
    """Mean absolute difference between two touching edges (lower = better seam)."""
    return float(np.mean(np.abs(_edge_strip(img_a, side_a) - _edge_strip(img_b, side_b))))


def _visual_balance(img, direction):
    """
    Grayscale brightness asymmetry across the image midpoint.

    Horizontal: right_mean - left_mean.  Positive → content heavier on right.
    Vertical:   bottom_mean - top_mean.  Positive → content heavier on bottom.

    Used to determine layout order: the image with higher score (more content on
    the "inner" side) goes first — left for horizontal, top for vertical — so
    subjects face inward toward the join seam rather than out toward the edge.
    """
    arr = np.asarray(img.convert('L'), dtype=np.float32)
    if direction == "horizontal":
        mid = arr.shape[1] // 2
        return float(arr[:, mid:].mean() - arr[:, :mid].mean())
    else:
        mid = arr.shape[0] // 2
        return float(arr[mid:, :].mean() - arr[:mid, :].mean())


def find_best_arrangement_2(paths, images, preserve_order, fix_direction):
    """
    For 2 images, determine direction and order.

    Direction — orientation heuristic (portrait-majority → horizontal). Seam
    matching is not used for direction because neutral/dark borders produce
    spuriously low cross-orientation scores.

    Order — visual balance: compare how much brightness each image has on its
    "inner" side (right half for horizontal, bottom half for vertical). The
    image with more content on that side goes first so subjects face inward.
    Seam matching overrides this only when one arrangement scores ≥5× better,
    which indicates a genuine panoramic seam rather than coincidental edge tone.

    preserve_order=True  → skip order, only determine direction
    fix_direction=str    → use that direction instead of orientation heuristic
    """
    # Direction from orientation heuristic (same logic as the 3+-image fallback)
    if fix_direction:
        direction = fix_direction
    else:
        portrait_count = sum(1 for img in images if img.height >= img.width)
        direction = "horizontal" if portrait_count * 2 >= len(images) else "vertical"
    print(f"Direction: orientation heuristic → {direction}")

    if preserve_order:
        return paths, images, direction

    sep = " | " if direction == "horizontal" else "\n─\n"
    a, b = os.path.basename(paths[0]), os.path.basename(paths[1])

    # Seam scores — used only for clear panoramic matches (≥5× ratio)
    if direction == "horizontal":
        score_01 = _seam_mad(images[0], "right", images[1], "left")
        score_10 = _seam_mad(images[1], "right", images[0], "left")
    else:
        score_01 = _seam_mad(images[0], "bottom", images[1], "top")
        score_10 = _seam_mad(images[1], "bottom", images[0], "top")

    lo, hi = min(score_01, score_10), max(score_01, score_10)
    if lo > 0 and hi / lo >= 5.0:
        # Panoramic seam — clear winner
        best_order = [0, 1] if score_01 <= score_10 else [1, 0]
        order_note = "seam matching (panoramic)"
    else:
        # Visual balance — image with more content on the inner side goes first
        b0 = _visual_balance(images[0], direction)
        b1 = _visual_balance(images[1], direction)
        best_order = [0, 1] if b0 >= b1 else [1, 0]
        order_note = f"visual balance ({b0:+.1f} vs {b1:+.1f})"

    first, second = best_order
    print(f"Order: {order_note} → [{os.path.basename(paths[first])}]{sep}[{os.path.basename(paths[second])}]")

    return [paths[i] for i in best_order], [images[i] for i in best_order], direction


_FORMAT_TO_EXT = {
    "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff",
    "BMP": ".bmp", "GIF": ".gif", "ICO": ".ico", "PDF": ".pdf",
}


def _make_output_path(image_paths, opened_images):
    """Auto-generate output path next to the first input, with dedup guard."""
    out_dir = os.path.dirname(os.path.abspath(image_paths[0]))
    fmts = {img.format for img in opened_images if img.format}
    ext = _FORMAT_TO_EXT.get(fmts.pop(), ".jpg") if len(fmts) == 1 else ".jpg"
    stem = "concat"
    candidate = os.path.join(out_dir, f"{stem}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(out_dir, f"{stem}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


# ── Main concat logic ─────────────────────────────────────────────────────────

def concat_images(image_paths, output_path=None, direction="auto", order="auto"):
    """Concatenate images preserving source encoding parameters.

    output_path may be None — the path is auto-generated next to the first input.
    """
    opened = [Image.open(p) for p in image_paths]
    input_formats = [img.format for img in opened]

    if output_path is None:
        output_path = _make_output_path(image_paths, opened)

    images = [img.convert("RGB") for img in opened]

    preserve_order  = (order != "auto")
    fix_direction = None if direction == "auto" else direction

    use_edge_match = (
        _HAS_NUMPY
        and len(image_paths) == 2
        and (not preserve_order or fix_direction is None)  # at least one thing to decide
    )

    if use_edge_match:
        image_paths, images, direction = find_best_arrangement_2(
            image_paths, images, preserve_order=preserve_order, fix_direction=fix_direction
        )
    else:
        if not preserve_order:
            original = list(image_paths)
            paired = sorted(zip(image_paths, images), key=lambda t: _sort_key(*t))
            new_paths = [p for p, _ in paired]
            if new_paths != original:
                image_paths = new_paths
                images = [img for _, img in paired]
                print("Order: auto-sorted by filename sequence")
                for p in image_paths:
                    print(f"  {os.path.basename(p)}")

        if fix_direction is None:
            portrait_count = sum(1 for img in images if img.height >= img.width)
            direction = "horizontal" if portrait_count * 2 >= len(images) else "vertical"
            print(f"Direction: auto (orientation heuristic) → {direction}")
        else:
            direction = fix_direction

        if not _HAS_NUMPY and len(image_paths) == 2:
            print("(install numpy for edge-color seam matching)")

    print(f"\nLayout: {direction}")

    # Detect encoding params from the first input image
    first = image_paths[0]
    qtables = quality = subsampling = None
    is_grayscale = False
    if input_formats[0] == "JPEG":
        qtables, quality, subsampling, is_grayscale = _jpeg_params(first)
    else:
        qtables, quality, subsampling = None, 85, 2

    # ── Lossless fast-path: jpegtran -drop (DCT-level, no pixel decode/encode) ─
    lossless = _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling)

    # ── Pillow re-encode fallback ─────────────────────────────────────────────
    if not lossless:
        if direction == "horizontal":
            total_w = sum(img.width  for img in images)
            total_h = max(img.height for img in images)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            x = 0
            for img in images:
                canvas.paste(img, (x, 0))
                x += img.width
        else:
            total_w = max(img.width  for img in images)
            total_h = sum(img.height for img in images)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            y = 0
            for img in images:
                canvas.paste(img, (0, y))
                y += img.height

        out_ext = os.path.splitext(output_path)[1].lower()
        if out_ext in (".jpg", ".jpeg"):
            # Preserve EXIF from the first source image (in the final arrangement)
            exif_data = opened[0].info.get('exif') if input_formats[0] == "JPEG" else None

            save_kwargs = {"exif": exif_data} if exif_data else {}
            if is_grayscale:
                canvas = canvas.convert("L")
                canvas.save(output_path, "JPEG", qtables=qtables, **save_kwargs)
            elif qtables:
                canvas.save(output_path, "JPEG", qtables=qtables, subsampling=subsampling, **save_kwargs)
            else:
                canvas.save(output_path, "JPEG", quality=quality, subsampling=subsampling, **save_kwargs)
        else:
            canvas.save(output_path)

    # ── Size report ───────────────────────────────────────────────────────────
    input_sizes = [os.path.getsize(p) for p in image_paths]
    output_size = os.path.getsize(output_path)
    combined    = sum(input_sizes)

    print(f"\nSize report:")
    for p, s in zip(image_paths, input_sizes):
        print(f"  {os.path.basename(p)}: {s // 1024} KB")
    print(f"  ─────────────────────────")
    print(f"  Combined input:  {combined    // 1024} KB")
    print(f"  Output:          {output_size // 1024} KB  ({output_size / combined:.2f}x)")

    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext in (".jpg", ".jpeg"):
        if lossless:
            enc = "lossless (jpegtran DCT)"
        elif qtables:
            enc = f"quality≈{quality} (exact source tables)"
        else:
            enc = f"quality={quality}"
        if is_grayscale:
            ss_label = "grayscale"
        else:
            ss_label = ['4:4:4', '4:2:2', '4:2:0'][subsampling]
        print(f"\nEncoding: {enc}, subsampling={ss_label}")
    else:
        print(f"\nEncoding: {out_ext.lstrip('.')} (lossless)")
    print(f"Saved: {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Concatenate JPEG images preserving encoding params.",
        usage="%(prog)s img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg] [--direction h|v|auto] [--order auto|as-given]",
    )
    _DIR_ALIASES = {"h": "horizontal", "v": "vertical", "a": "auto"}

    parser.add_argument("images",      nargs="+", help="Input image paths")
    parser.add_argument("--output", "-o", default=None, help="Output path (auto-generated if omitted)")
    parser.add_argument("--direction", "-d",
                        choices=["horizontal", "vertical", "auto", "h", "v", "a"],
                        default="auto",
                        metavar="{horizontal|h, vertical|v, auto|a}",
                        help="Layout direction (default: auto)")
    parser.add_argument("--order",     choices=["auto", "as-given"],               default="auto")
    args = parser.parse_args()

    direction = _DIR_ALIASES.get(args.direction, args.direction)

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        print(f"Error: files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    # Defer output-path computation to concat_images to avoid opening
    # input files twice (once here for format detection, once for processing).
    concat_images(args.images, args.output, direction, args.order)


if __name__ == "__main__":
    main()
