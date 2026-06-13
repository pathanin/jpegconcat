#!/usr/bin/env python3
"""
concat_jpeg.py — Concatenate JPEG images while preserving original encoding parameters.

Usage:
    python3 concat_jpeg.py --images img1.jpg img2.jpg [img3.jpg ...] --output out.jpg
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
import struct
import sys

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


# ── JPEG parameter detection ──────────────────────────────────────────────────

def estimate_jpeg_quality(path):
    with open(path, "rb") as f:
        data = f.read()
    tables = []
    i = 0
    while i < len(data) - 3:
        if data[i] == 0xFF and data[i + 1] == 0xDB:
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            seg = data[i + 4 : i + 2 + length]
            j = 0
            while j < len(seg):
                prec_id = seg[j]; j += 1
                precision = (prec_id >> 4) + 1
                tbl = []
                for _ in range(64):
                    if precision == 1:
                        tbl.append(seg[j]); j += 1
                    else:
                        tbl.append(struct.unpack(">H", seg[j : j + 2])[0]); j += 2
                tables.append(tbl)
        i += 1
    if not tables:
        return 85
    std_luma = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]
    t = tables[0]
    s = sum(t[k] * 100 / std_luma[k] for k in range(64)) / 64
    q = (200 - s) / 2 if s <= 100 else 5000 / s
    return max(1, min(95, round(q)))


def detect_subsampling(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data) - 1:
        if data[i] == 0xFF and data[i + 1] in (0xC0, 0xC1, 0xC2):
            ncomp = data[i + 9]
            if ncomp < 3:
                return 2
            y_samp  = data[i + 11]
            y_h, y_v   = (y_samp >> 4), (y_samp & 0xF)
            cb_samp = data[i + 14]
            cb_h, cb_v = (cb_samp >> 4), (cb_samp & 0xF)
            if y_h == cb_h and y_v == cb_v:
                return 0
            if y_v == cb_v:
                return 1
            return 2
        i += 1
    return 2


# ── Filename / EXIF ordering (fallback) ───────────────────────────────────────

def exif_datetime(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "DateTimeOriginal":
                    return val
    except Exception:
        pass
    return None


def sort_key(path):
    name = os.path.basename(path)
    nums = tuple(int(n) for n in re.findall(r"\d+", name))
    exif_dt = exif_datetime(path) or ""
    mtime = os.path.getmtime(path)
    return (nums, exif_dt, mtime)


def auto_sort(paths):
    return sorted(paths, key=sort_key)


def auto_direction_fallback(images):
    """Portrait majority → horizontal; landscape majority → vertical."""
    portrait = sum(1 for img in images if img.height >= img.width)
    landscape = len(images) - portrait
    return "horizontal" if portrait >= landscape else "vertical"


# ── Edge-color seam matching ──────────────────────────────────────────────────

def _edge_strip(arr, side):
    """
    Extract a (EDGE_LEN, 3) float32 strip from one side of an image array (H, W, 3).
    Averages EDGE_DEPTH pixels inward to reduce compression-artifact noise, then
    resamples to EDGE_LEN so strips from different-sized images are comparable.
    """
    if side == "right":
        strip = arr[:, -EDGE_DEPTH:, :].mean(axis=1)   # (H, 3)
    elif side == "left":
        strip = arr[:, :EDGE_DEPTH,  :].mean(axis=1)   # (H, 3)
    elif side == "bottom":
        strip = arr[-EDGE_DEPTH:, :, :].mean(axis=0)   # (W, 3)
    else:  # top
        strip = arr[:EDGE_DEPTH,  :, :].mean(axis=0)   # (W, 3)

    cur = strip.shape[0]
    if cur != EDGE_LEN:
        idx = np.linspace(0, cur - 1, EDGE_LEN)
        lo  = np.floor(idx).astype(int)
        hi  = np.minimum(lo + 1, cur - 1)
        t   = (idx - lo)[:, None]
        strip = strip[lo] * (1 - t) + strip[hi] * t
    return strip


def _seam_mad(arr_a, side_a, arr_b, side_b):
    """Mean absolute difference between two touching edges (lower = better seam)."""
    return float(np.mean(np.abs(_edge_strip(arr_a, side_a) - _edge_strip(arr_b, side_b))))


def find_best_arrangement_2(paths, images, fix_order, fix_direction):
    """
    Try every allowed seam connection for 2 images and return
    (ordered_paths, ordered_images, direction) for the best match.

    fix_order=True     → keep as-given order, only test directions
    fix_direction=str  → keep that direction, only test orderings
    """
    arrs      = [np.array(img, dtype=np.float32) for img in images]
    orders     = [[0, 1]] if fix_order else [[0, 1], [1, 0]]
    directions = [fix_direction] if fix_direction else ["horizontal", "vertical"]

    candidates = []
    for order in orders:
        for direction in directions:
            if direction == "horizontal":
                score = _seam_mad(arrs[order[0]], "right", arrs[order[1]], "left")
            else:
                score = _seam_mad(arrs[order[0]], "bottom", arrs[order[1]], "top")
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
    images = [Image.open(p).convert("RGB") for p in image_paths]

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
            original = image_paths[:]
            image_paths = auto_sort(image_paths)
            images = [Image.open(p).convert("RGB") for p in image_paths]
            if image_paths != original:
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

    first = image_paths[0]
    if first.lower().endswith((".jpg", ".jpeg")):
        quality     = estimate_jpeg_quality(first)
        subsampling = detect_subsampling(first)
    else:
        quality     = 85
        subsampling = 2

    canvas.save(output_path, "JPEG", quality=quality, subsampling=subsampling)

    input_sizes = [os.path.getsize(p) for p in image_paths]
    output_size = os.path.getsize(output_path)
    combined    = sum(input_sizes)

    print(f"\nSize report:")
    for p, s in zip(image_paths, input_sizes):
        print(f"  {os.path.basename(p)}: {s // 1024} KB")
    print(f"  ─────────────────────────")
    print(f"  Combined input:  {combined    // 1024} KB")
    print(f"  Output:          {output_size // 1024} KB  ({output_size / combined:.2f}x)")
    print(f"\nEncoding: quality={quality}, subsampling={['4:4:4','4:2:2','4:2:0'][subsampling]}")
    print(f"Saved: {output_path}")


def make_output_path(inputs):
    """Auto-generate output path next to the first input, with dedup guard."""
    out_dir = os.path.dirname(os.path.abspath(inputs[0]))
    candidate = os.path.join(out_dir, "concat.jpg")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(out_dir, f"concat_{n}.jpg")
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
