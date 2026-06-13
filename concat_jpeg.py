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


# ── JPEG parameter detection ──────────────────────────────────────────────────

def _jpeg_params(path):
    """
    Return (quality, subsampling) via a single streaming JPEG marker scan.
    Reads only the header (stops at SOS / pixel data), never loading the full file.
    """
    quality = 85
    subsampling = 2
    have_q = have_s = False

    with open(path, "rb") as f:
        if f.read(2) != b'\xff\xd8':
            return quality, subsampling

        while not (have_q and have_s):
            b = f.read(2)
            if len(b) < 2 or b[0] != 0xFF:
                break
            marker = b[1]
            # standalone markers (no length field)
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            # SOS — pixel data starts here; nothing useful follows in the header
            if marker == 0xDA:
                break
            lb = f.read(2)
            if len(lb) < 2:
                break
            length = struct.unpack(">H", lb)[0]
            seg = f.read(length - 2)

            if marker == 0xDB and not have_q:  # DQT
                j = 0
                while j < len(seg):
                    prec_id = seg[j]; j += 1
                    precision = (prec_id >> 4) + 1
                    table_id = prec_id & 0x0F
                    tbl = []
                    for _ in range(64):
                        if precision == 1:
                            tbl.append(seg[j]); j += 1
                        else:
                            tbl.append(struct.unpack(">H", seg[j:j+2])[0]); j += 2
                    if table_id == 0:  # luma table → compute quality
                        s = sum(tbl[k] * 100 / _STD_LUMA[k] for k in range(64)) / 64
                        q = (200 - s) / 2 if s <= 100 else 5000 / s
                        quality = max(1, min(95, round(q)))
                        have_q = True
                        break

            elif marker in (0xC0, 0xC1, 0xC2) and not have_s:  # SOF0/1/2
                # seg layout: precision(1) height(2) width(2) ncomp(1) [id samp qtbl]×ncomp
                if len(seg) >= 11 and seg[5] >= 3:
                    y_h,  y_v  = seg[7] >> 4, seg[7] & 0xF
                    cb_h, cb_v = seg[10] >> 4, seg[10] & 0xF
                    if y_h == cb_h and y_v == cb_v:
                        subsampling = 0
                    elif y_v == cb_v:
                        subsampling = 1
                    else:
                        subsampling = 2
                have_s = True

    return quality, subsampling


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
        quality, subsampling = _jpeg_params(first)
    else:
        quality, subsampling = 85, 2

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
