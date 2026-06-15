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
    from PIL import Image, ExifTags
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
    Return (qtables, quality, subsampling) for a JPEG file.

    qtables — Pillow's img.quantization dict (exact tables from libjpeg, suitable
              for canvas.save(qtables=...)), or None if unavailable.
    quality — integer estimate derived from the luma table, for display only.
    subsampling — 0/1/2 (4:4:4 / 4:2:2 / 4:2:0) from the SOF marker.
    """
    qtables = None
    quality = 85
    subsampling = 2

    # Quality + tables: use Pillow's libjpeg-parsed quantization dict (header-only open)
    try:
        _img = Image.open(path)
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
                        if len(seg) >= 11 and seg[5] >= 3:
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

    return qtables, quality, subsampling


# ── jpegtran lossless fast-path ───────────────────────────────────────────────

def _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling):
    """
    Attempt lossless DCT-level concatenation via jpegtran -drop.
    Returns True on success, False if any precondition is unmet.

    Preconditions:
      - jpegtran (libjpeg-turbo ≥ 1.4) is in PATH
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
        _, _, s = _jpeg_params(path)
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

def _exif_dt(img):
    """Extract DateTimeOriginal from an already-open PIL Image (no extra file I/O)."""
    try:
        exif = img.getexif()
        for tag, val in exif.items():
            if ExifTags.TAGS.get(tag) == "DateTimeOriginal":
                return val
    except Exception:
        pass
    return None


def _sort_key(path, img):
    name = os.path.basename(path)
    nums = tuple(int(n) for n in re.findall(r"\d+", name))
    return (nums, _exif_dt(img) or "", os.path.getmtime(path))


def auto_direction_fallback(images):
    """Portrait majority → horizontal; landscape majority → vertical."""
    portrait = sum(1 for img in images if img.height >= img.width)
    return "horizontal" if portrait >= len(images) - portrait else "vertical"


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


def find_best_arrangement_2(paths, images, fix_order, fix_direction):
    """
    Try every allowed seam connection for 2 images and return
    (ordered_paths, ordered_images, direction) for the best match.

    fix_order=True     → keep as-given order, only test directions
    fix_direction=str  → keep that direction, only test orderings
    """
    orders     = [[0, 1]] if fix_order else [[0, 1], [1, 0]]
    directions = [fix_direction] if fix_direction else ["horizontal", "vertical"]

    candidates = []
    for order in orders:
        for direction in directions:
            if direction == "horizontal":
                score = _seam_mad(images[order[0]], "right", images[order[1]], "left")
            else:
                score = _seam_mad(images[order[0]], "bottom", images[order[1]], "top")
            a = os.path.basename(paths[order[0]])
            b = os.path.basename(paths[order[1]])
            sep = " | " if direction == "horizontal" else "\n─\n"
            label = f"[{a}]{sep}[{b}] ({direction})"
            candidates.append((score, order, direction, label))

    candidates.sort(key=lambda x: x[0])
    best_score = candidates[0][0]

    print("Edge seam scores (lower = better match):")
    for score, _, _, label in candidates:
        marker = " ✓" if score == best_score else ""
        print(f"  {score:6.1f}  {label}{marker}")

    _, best_order, best_direction, _ = candidates[0]
    return [paths[i] for i in best_order], [images[i] for i in best_order], best_direction


# ── Main concat logic ─────────────────────────────────────────────────────────

def concat_images(image_paths, output_path, direction="auto", order="auto"):
    opened = [Image.open(p) for p in image_paths]
    input_formats = [img.format for img in opened]
    images = [img.convert("RGB") for img in opened]

    fix_order     = (order != "auto")
    fix_direction = None if direction == "auto" else direction

    use_edge_match = (
        _HAS_NUMPY
        and len(image_paths) == 2
        and (not fix_order or fix_direction is None)  # at least one thing to decide
    )

    if use_edge_match:
        print("Order + direction: edge-color seam matching")
        image_paths, images, direction = find_best_arrangement_2(
            image_paths, images, fix_order=fix_order, fix_direction=fix_direction
        )
    else:
        if not fix_order:
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
            direction = auto_direction_fallback(images)
            print(f"Direction: auto (orientation heuristic) → {direction}")
        else:
            direction = fix_direction

        if not _HAS_NUMPY and len(image_paths) == 2:
            print("(install numpy for edge-color seam matching)")

    print(f"\nLayout: {direction}")

    # Detect encoding params from the first input image
    first = image_paths[0]
    if input_formats[0] == "JPEG":
        qtables, quality, subsampling = _jpeg_params(first)
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
            if qtables:
                canvas.save(output_path, "JPEG", qtables=qtables, subsampling=subsampling)
            else:
                canvas.save(output_path, "JPEG", quality=quality, subsampling=subsampling)
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
        print(f"\nEncoding: {enc}, subsampling={['4:4:4','4:2:2','4:2:0'][subsampling]}")
    else:
        print(f"\nEncoding: {out_ext.lstrip('.')} (lossless)")
    print(f"Saved: {output_path}")


_FORMAT_TO_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}

def _detect_format(path):
    """Return Pillow format string (e.g. 'JPEG', 'PNG') detected from file bytes."""
    with Image.open(path) as img:
        return img.format

def _output_ext(inputs):
    """Return a common extension if all inputs share the same actual format, else .jpg."""
    fmts = {_detect_format(p) for p in inputs}
    if len(fmts) == 1:
        fmt = fmts.pop()
        return _FORMAT_TO_EXT.get(fmt, os.path.splitext(inputs[0])[1].lower() or ".jpg")
    return ".jpg"


def make_output_path(inputs):
    """Auto-generate output path next to the first input, with dedup guard."""
    out_dir = os.path.dirname(os.path.abspath(inputs[0]))
    ext = _output_ext(inputs)
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


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate JPEG images preserving encoding params.",
        usage="%(prog)s img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg] [--direction h|v|auto] [--order auto|as-given]",
    )
    parser.add_argument("images",      nargs="+", help="Input image paths")
    parser.add_argument("--output",    default=None, help="Output path (auto-generated if omitted)")
    parser.add_argument("--direction", choices=["horizontal", "vertical", "auto"], default="auto")
    parser.add_argument("--order",     choices=["auto", "as-given"],               default="auto")
    args = parser.parse_args()

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        print(f"Error: files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    output = args.output or make_output_path(args.images)
    concat_images(args.images, output, args.direction, args.order)


if __name__ == "__main__":
    main()
