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

Note: The lossless jpegtran fast-path uses -copy all to preserve metadata from
the first source image (EXIF, ICC profiles, XMP). The Pillow fallback also
preserves EXIF from the first source image.
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
    from PIL import Image, ImageOps
except ImportError:
    print("Pillow not installed. Run: pip install Pillow --break-system-packages")
    sys.exit(1)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import importlib.util as _ilu
    _HAS_FLASK = _ilu.find_spec("flask") is not None
except Exception:
    _HAS_FLASK = False

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

def _parse_sof_subsampling(data):
    """Extract chroma subsampling from raw JPEG bytes (SOF marker scan).

    Returns 0/1/2 for 4:4:4 / 4:2:2 / 4:2:0, or 2 (default) on failure.
    """
    try:
        if data[0:2] != b'\xff\xd8':
            return 2
        pos = 2
        while pos < len(data) - 1:
            if data[pos] != 0xFF:
                return 2
            marker = data[pos + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if marker == 0xDA:
                break
            if pos + 3 >= len(data):
                break
            seg_len = (data[pos + 2] << 8) | data[pos + 3]
            if seg_len < 2:
                break
            seg_start = pos + 4
            seg_end = min(pos + 2 + seg_len, len(data))
            if marker in (0xC0, 0xC1, 0xC2):
                seg = data[seg_start:seg_end]
                if len(seg) >= 15 and seg[5] >= 3:
                    y_h,  y_v  = seg[7] >> 4, seg[7] & 0xF
                    cb_h, cb_v = seg[10] >> 4, seg[10] & 0xF
                    if y_h == cb_h and y_v == cb_v:
                        return 0
                    elif y_v == cb_v:
                        return 1
                    else:
                        return 2
                break
            pos += 2 + seg_len
    except Exception:
        pass
    return 2


def _jpeg_params(path):
    """
    Return (qtables, quality, subsampling, is_grayscale) for a JPEG file.

    Reads the file exactly once — binary data is parsed for subsampling,
    Pillow's header-only open extracts quantization tables.

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

    try:
        with open(path, "rb") as f:
            raw = f.read()

        subsampling = _parse_sof_subsampling(raw)
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

    return qtables, quality, subsampling, is_grayscale


def _jpeg_params_from_image(img):
    """Extract JPEG params from an already-opened PIL Image (avoids re-reading disk).

    For JPEG images, still does a binary read for SOF subsampling since Pillow
    doesn't expose it cleanly. For non-JPEG, returns defaults.
    """
    qtables = None
    quality = 85
    subsampling = 2
    is_grayscale = (img.mode == "L")

    if img.format == "JPEG" and img.quantization:
        qtables = img.quantization
        tbl = qtables.get(0, [])
        if len(tbl) == 64:
            s = sum(tbl[k] * 100 / _STD_LUMA[k] for k in range(64)) / 64
            q = (200 - s) / 2 if s <= 100 else 5000 / s
            quality = max(1, min(95, round(q)))

    return qtables, quality, subsampling, is_grayscale


# ── jpegtran lossless fast-path ───────────────────────────────────────────────

def _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling, all_subsampling=None):
    """
    Attempt lossless DCT-level concatenation via jpegtran -drop.

    Returns True on success, False if any precondition is unmet.

    NOTE: jpegtran is called with -copy all, so metadata (EXIF, ICC
    profiles, XMP) from the first source image is preserved in the output.
    Metadata from subsequent source images is discarded, which is expected
    for a composite.

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

    if all_subsampling is not None:
        if any(s != subsampling for s in all_subsampling):
            return False
    else:
        for path in image_paths[1:]:
            _, _, s, _ = _jpeg_params(path)
            if s != subsampling:
                return False

    mcu_w, mcu_h = _MCU_DIMS.get(subsampling, (8, 8))

    if direction == "horizontal":
        if not all(img.height % mcu_h == 0 for img in images):
            return False
        x = 0
        for img in images[:-1]:
            x += img.width
            if x % mcu_w != 0:
                return False
    else:
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
        # -copy all: metadata from the first source image is preserved
        current = canvas_path
        x = y = 0
        for idx, (path, img) in enumerate(zip(image_paths, images)):
            out_path = os.path.join(td, f"step_{idx}.jpg")
            result = subprocess.run(
                [jpegtran, "-copy", "all", "-drop", f"+{x}+{y}", path, current],
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


def _make_blend_mask(w, h, direction):
    """Linear gradient mask for cross-fade blending in the overlap zone.

    direction='horizontal': 0 (shows prev) on left  → 255 (shows next) on right
    direction='vertical':   0 (shows prev) on top   → 255 (shows next) on bottom
    PIL composite: mask=0 → image2 (prev), mask=255 → image1 (next)
    """
    if direction == "horizontal":
        row = Image.new("L", (w, 1))
        pix = row.load()
        denom = max(w - 1, 1)
        for x in range(w):
            pix[x, 0] = round(255 * x / denom)
        return row.resize((w, h), Image.NEAREST)
    else:
        col = Image.new("L", (1, h))
        pix = col.load()
        denom = max(h - 1, 1)
        for y in range(h):
            pix[0, y] = round(255 * y / denom)
        return col.resize((w, h), Image.NEAREST)


def _suggest_overlap(imgs, direction):
    """
    Suggest an overlap in pixels by finding where the blend zone has the
    flattest (most background-like) content in both images.

    Scans up to 30% of the join dimension in ~60 steps and picks the overlap
    where combined edge variance is lowest — that zone falls on neutral
    backgrounds rather than on subjects or text.

    Falls back to 10% of join dimension when numpy is unavailable.
    """
    if direction == "horizontal":
        dim = min(img.width for img in imgs)
    else:
        dim = min(img.height for img in imgs)

    fallback = max(30, round(dim * 0.10))
    if not _HAS_NUMPY or len(imgs) < 2:
        return fallback

    img_a, img_b = imgs[0], imgs[1]
    max_ov = round(dim * 0.30)
    if max_ov < 10:
        return fallback

    PERP = 64           # thumbnail size in the non-join dimension
    N    = 60           # number of candidate steps
    step = max(1, max_ov // N)

    try:
        if direction == "horizontal":
            a = np.asarray(
                img_a.crop((img_a.width - max_ov, 0, img_a.width, img_a.height))
                     .resize((max_ov, PERP), Image.LANCZOS), dtype=np.float32)
            b = np.asarray(
                img_b.crop((0, 0, max_ov, img_b.height))
                     .resize((max_ov, PERP), Image.LANCZOS), dtype=np.float32)
            def _zone_score(ov):
                return float(np.std(a[:, max_ov - ov:]) + np.std(b[:, :ov]))
        else:
            a = np.asarray(
                img_a.crop((0, img_a.height - max_ov, img_a.width, img_a.height))
                     .resize((PERP, max_ov), Image.LANCZOS), dtype=np.float32)
            b = np.asarray(
                img_b.crop((0, 0, img_b.width, max_ov))
                     .resize((PERP, max_ov), Image.LANCZOS), dtype=np.float32)
            def _zone_score(ov):
                return float(np.std(a[max_ov - ov:]) + np.std(b[:ov]))

        scores = [(ov, _zone_score(ov)) for ov in range(step, max_ov + 1, step)]
        if not scores:
            return fallback
        min_score = min(s for _, s in scores)
        # Among candidates within 5% of the minimum, prefer the smallest overlap.
        # This avoids greedily choosing large overlaps when the variance curve is flat.
        # Floor at fallback (10%) so dissimilar images still get a tasteful default.
        threshold = min_score * 1.05
        best_ov = min(ov for ov, s in scores if s <= threshold)
        return max(fallback, best_ov)
    except Exception:
        return fallback


def _visual_balance(img, direction):
    """
    Grayscale brightness asymmetry across the image midpoint.

    Horizontal: right_mean - left_mean.  Positive → content heavier on right.
    Vertical:   bottom_mean - top_mean.  Positive → content heavier on bottom.

    Used to determine layout order: the image with higher score (more content on
    the "inner" side) goes first — left for horizontal, top for vertical — so
    subjects face inward toward the join seam rather than out toward the edge.
    """
    w, h = img.size
    scale = max(1, max(w, h) // 128)
    thumb = img.convert('L').resize((max(1, w // scale), max(1, h // scale)))
    arr = np.asarray(thumb, dtype=np.float32)
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

def concat_images(image_paths, output_path=None, direction="auto", order="auto", fit=False, overlap=0, overlap_blend="hard"):
    """Concatenate images preserving source encoding parameters.

    output_path may be None — the path is auto-generated next to the first input.
    """
    _first = Image.open(image_paths[0])
    _first_format = _first.format
    _first_exif = _first.info.get('exif') if _first_format == "JPEG" else None

    if output_path is None:
        fmt = _first.format or "JPEG"
        out_dir = os.path.dirname(os.path.abspath(image_paths[0]))
        ext = _FORMAT_TO_EXT.get(fmt, ".jpg")
        stem = "concat"
        candidate = os.path.join(out_dir, f"{stem}{ext}")
        n = 2
        while os.path.exists(candidate):
            candidate = os.path.join(out_dir, f"{stem}_{n}{ext}")
            n += 1
        output_path = candidate
    _first.close()

    images = []
    input_formats = []
    for p in image_paths:
        with Image.open(p) as img:
            input_formats.append(img.format)
            images.append(ImageOps.exif_transpose(img).convert("RGB"))

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

    if fit:
        if direction == "horizontal":
            target_h = max(img.height for img in images)
            images = [
                img.resize((round(img.width * target_h / img.height), target_h), Image.LANCZOS)
                if img.height != target_h else img
                for img in images
            ]
        else:
            target_w = max(img.width for img in images)
            images = [
                img.resize((target_w, round(img.height * target_w / img.width)), Image.LANCZOS)
                if img.width != target_w else img
                for img in images
            ]
        print(f"Fit: resized images to share {'height' if direction == 'horizontal' else 'width'}")

    # Detect encoding params from the first input image
    first = image_paths[0]
    qtables = quality = subsampling = None
    is_grayscale = False
    if input_formats[0] == "JPEG":
        qtables, quality, subsampling, is_grayscale = _jpeg_params(first)
    else:
        qtables, quality, subsampling = None, 85, 2

    all_subsampling = None
    if _HAS_NUMPY and len(image_paths) > 1 and input_formats[0] == "JPEG":
        all_subsampling = [subsampling]
        for i in range(1, len(image_paths)):
            if input_formats[i] == "JPEG":
                _, _, s, _ = _jpeg_params(image_paths[i])
                all_subsampling.append(s)
            else:
                all_subsampling.append(2)

    if overlap > 0:
        min_join_dim = (min(img.width for img in images) if direction == "horizontal"
                        else min(img.height for img in images))
        if overlap >= min_join_dim:
            overlap = max(0, min_join_dim - 1)
            print(f"Warning: overlap clamped to {overlap}px")

    lossless = overlap == 0 and _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling, all_subsampling=all_subsampling)

    # ── Pillow re-encode fallback ─────────────────────────────────────────────
    if not lossless:
        if direction == "horizontal":
            total_w = sum(img.width  for img in images) - overlap * (len(images) - 1)
            total_h = max(img.height for img in images)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            x = 0
            for i, img in enumerate(images):
                if i > 0 and overlap > 0:
                    x -= overlap
                    if overlap_blend == "fade":
                        blend_h    = min(img.height, canvas.height)
                        left_crop  = canvas.crop((x, 0, x + overlap, blend_h))
                        right_crop = img.crop((0, 0, overlap, blend_h))
                        mask = _make_blend_mask(overlap, blend_h, "horizontal")
                        canvas.paste(Image.composite(right_crop, left_crop, mask), (x, 0))
                        if img.width > overlap:
                            canvas.paste(img.crop((overlap, 0, img.width, img.height)), (x + overlap, 0))
                    else:
                        canvas.paste(img, (x, 0))
                else:
                    canvas.paste(img, (x, 0))
                x += img.width
        else:
            total_w = max(img.width  for img in images)
            total_h = sum(img.height for img in images) - overlap * (len(images) - 1)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            y = 0
            for i, img in enumerate(images):
                if i > 0 and overlap > 0:
                    y -= overlap
                    if overlap_blend == "fade":
                        blend_w    = min(img.width, canvas.width)
                        top_crop   = canvas.crop((0, y, blend_w, y + overlap))
                        bot_crop   = img.crop((0, 0, blend_w, overlap))
                        mask = _make_blend_mask(blend_w, overlap, "vertical")
                        canvas.paste(Image.composite(bot_crop, top_crop, mask), (0, y))
                        if img.height > overlap:
                            canvas.paste(img.crop((0, overlap, img.width, img.height)), (0, y + overlap))
                    else:
                        canvas.paste(img, (0, y))
                else:
                    canvas.paste(img, (0, y))
                y += img.height

        out_ext = os.path.splitext(output_path)[1].lower()
        if out_ext in (".jpg", ".jpeg"):
            # Preserve EXIF from the first source image (in the final arrangement)
            exif_data = _first_exif

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



# ── Web UI ────────────────────────────────────────────────────────────────────

# Embedded single-file frontend — served directly from memory, no files on disk.
_WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Stitcher</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #ffffff;
  color: #1d1d1f;
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

button { font-family: inherit; }

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 52px;
  background: #f5f5f7;
  border-bottom: 1px solid #e0e0e0;
  flex-shrink: 0;
}

h1 {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.374px;
  color: #1d1d1f;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 14px;
}

#status {
  font-size: 14px;
  font-weight: 400;
  letter-spacing: -0.224px;
  color: #7a7a7a;
}

/* Fit toggle — Apple-style switch */
.fit-toggle-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 400;
  letter-spacing: -0.224px;
  color: #1d1d1f;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}

.fit-toggle-input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
  pointer-events: none;
}

.fit-toggle-switch {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
  background: #e0e0e0;
  border-radius: 10px;
  transition: background 0.15s;
  flex-shrink: 0;
}

.fit-toggle-switch::after {
  content: '';
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  background: #fff;
  border-radius: 50%;
  transition: transform 0.15s;
  box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}

.fit-toggle-input:checked + .fit-toggle-switch {
  background: #0066cc;
}

.fit-toggle-input:checked + .fit-toggle-switch::after {
  transform: translateX(16px);
}

#stitch-btn {
  padding: 11px 22px;
  background: #0066cc;
  color: #fff;
  border: none;
  border-radius: 9999px;
  font-family: inherit;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.374px;
  line-height: 1;
  cursor: pointer;
  transition: opacity 0.12s, transform 0.08s;
  white-space: nowrap;
}

#stitch-btn:hover:not(:disabled) { opacity: 0.85; }
#stitch-btn:active:not(:disabled) { transform: scale(0.95); }
#stitch-btn:disabled {
  background: #e0e0e0;
  color: #999;
  cursor: default;
}
#stitch-btn:focus-visible {
  outline: 2px solid #0071e3;
  outline-offset: 2px;
}

#app-body {
  flex: 1;
  display: flex;
  overflow: hidden;
  min-height: 0;
}

#app-body.layout-side  { flex-direction: row; }
#app-body.layout-stack { flex-direction: column; }

#editor-pane {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px;
  overflow: auto;
  flex-shrink: 0;
  background: #ffffff;
}

#app-body.has-preview.layout-side  #editor-pane { max-width: 42%; }
#app-body.has-preview.layout-stack #editor-pane { max-height: 42%; }
#app-body:not(.has-preview) #editor-pane { flex: 1; }

#preview-pane {
  display: none;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 20px;
  min-width: 0;
  min-height: 0;
  border-left: 1px solid #e0e0e0;
  background: #fafafc;
}

#app-body.has-preview #preview-pane { display: flex; }
#app-body.layout-stack #preview-pane { border-left: none; border-top: 1px solid #e0e0e0; }

#preview-img-wrap {
  flex: 1;
  align-self: stretch;
  min-height: 0;
  min-width: 0;
  position: relative;
}

#preview-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 6px;
  background: #ffffff;
}

#download-btn {
  display: inline-block;
  padding: 11px 22px;
  background: transparent;
  color: #0066cc;
  border: 1px solid #0066cc;
  border-radius: 9999px;
  font-family: inherit;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.374px;
  line-height: 1;
  cursor: pointer;
  text-decoration: none;
  transition: background 0.12s, transform 0.08s;
  flex-shrink: 0;
  white-space: nowrap;
}

#download-btn:hover { background: rgba(0,102,204,0.06); }
#download-btn:active { transform: scale(0.95); }
#download-btn:focus-visible {
  outline: 2px solid #0071e3;
  outline-offset: 2px;
}

#drop-zone {
  border: 2px dashed #e0e0e0;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #7a7a7a;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
  user-select: none;
  min-width: 300px;
  min-height: 200px;
}

#app-body:not(.has-preview) #editor-pane {
  align-items: stretch;
  justify-content: flex-start;
}

#app-body:not(.has-preview) #drop-zone {
  flex: 1;
}

#app-body:not(.has-preview) #grid-outer {
  flex: 1;
  overflow: auto;
}

#app-body:not(.has-preview) .cell img {
  max-width: 320px;
  max-height: 320px;
}

#drop-zone.drag-over {
  border-color: #0066cc;
  color: #0066cc;
  background: rgba(0,102,204,0.04);
}

#drop-zone svg { flex-shrink: 0; }

#drop-zone .hint {
  font-size: 14px;
  color: #7a7a7a;
}

#file-input { display: none; }

#grid-outer {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

#grid-scroll {
  display: flex;
  align-items: center;
  gap: 6px;
}

#grid {
  display: inline-grid;
  gap: 6px;
  background: transparent;
  padding: 6px;
}

.cell {
  position: relative;
  overflow: hidden;
  border-radius: 8px;
  background: #ffffff;
  border: 1px solid #e0e0e0;
  cursor: grab;
}

.cell:active { cursor: grabbing; }
.cell.dragging { opacity: 0.35; }

.cell img {
  display: block;
  max-width: 200px;
  max-height: 200px;
  width: auto;
  height: auto;
  pointer-events: none;
  user-select: none;
}

.cell.empty {
  min-width: 80px;
  min-height: 80px;
  border: 2px dashed #e0e0e0;
  background: #fafafc;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7a7a7a;
  font-size: 1.3rem;
  cursor: default;
  transition: border-color 0.1s, color 0.1s, background 0.1s;
}

.cell.empty.drag-over-empty {
  border-color: #0066cc;
  color: #0066cc;
  background: rgba(0,102,204,0.06);
}

.drop-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2;
}

.drop-overlay .dz {
  position: absolute;
  opacity: 0;
  transition: opacity 0.08s;
  background: rgba(0,102,204,0.35);
}

.drop-overlay .dz.active { opacity: 1; }

.drop-overlay .dz-center { inset: 0; }
.drop-overlay .dz-left   { top: 0; left: 0; bottom: 0; width: 28%; }
.drop-overlay .dz-right  { top: 0; right: 0; bottom: 0; width: 28%; }
.drop-overlay .dz-top    { top: 0; left: 0; right: 0; height: 28%; }
.drop-overlay .dz-bottom { bottom: 0; left: 0; right: 0; height: 28%; }

.btn-remove {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 20px;
  height: 20px;
  background: rgba(0,0,0,0.55);
  border: none;
  border-radius: 4px;
  color: #ccc;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.12s, background 0.12s, color 0.12s;
  z-index: 4;
}

.cell:hover .btn-remove { opacity: 1; }
.btn-remove:hover { background: #dc2626; color: #fff; }

.cell-toolbar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 26px;
  background: rgba(0,0,0,0.65);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1px;
  opacity: 0;
  transition: opacity 0.12s;
  z-index: 4;
  border-radius: 0 0 4px 4px;
}

.cell:hover .cell-toolbar { opacity: 1; }

.btn-transform {
  width: 24px;
  height: 22px;
  background: transparent;
  border: none;
  color: #ccc;
  font-size: 13px;
  cursor: pointer;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.1s, color 0.1s;
  flex-shrink: 0;
}

.btn-transform:hover { background: rgba(255,255,255,0.15); color: #fff; }

/* Overlap control */
#overlap-control {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 400;
  letter-spacing: -0.224px;
  color: #1d1d1f;
  white-space: nowrap;
}

#overlap-slider {
  -webkit-appearance: none;
  appearance: none;
  width: 96px;
  height: 4px;
  border-radius: 2px;
  background: #e0e0e0;
  outline: none;
  cursor: pointer;
  transition: background 0.15s;
}
#overlap-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #0066cc;
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
#overlap-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border: none;
  border-radius: 50%;
  background: #0066cc;
  cursor: pointer;
}

#overlap-val {
  min-width: 46px;
  font-variant-numeric: tabular-nums;
  color: #7a7a7a;
  text-align: right;
}

.overlap-step-btn {
  width: 22px;
  height: 22px;
  padding: 0;
  font-size: 15px;
  font-family: inherit;
  font-weight: 400;
  line-height: 1;
  background: transparent;
  border: 1px solid #c0c0c0;
  border-radius: 4px;
  color: #555;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.overlap-step-btn:hover { border-color: #0066cc; color: #0066cc; }
.overlap-step-btn:active { background: rgba(0,102,204,0.08); }
.zoom-level-btn {
  padding: 1px 5px;
  font-size: 11px;
  border: 1px solid #c0c0c0;
  border-radius: 3px;
  background: transparent;
  color: #666;
  cursor: pointer;
}
.zoom-level-btn.active { background: #0066cc; color: #fff; border-color: #0066cc; }

#overlap-auto-btn {
  padding: 3px 8px;
  font-size: 12px;
  font-family: inherit;
  font-weight: 400;
  background: transparent;
  border: 1px solid #c0c0c0;
  border-radius: 4px;
  color: #555;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s;
  line-height: 1;
}
#overlap-auto-btn:hover { border-color: #0066cc; color: #0066cc; }
#overlap-auto-btn.active { border-color: #0066cc; color: #0066cc; background: rgba(0,102,204,0.06); }

.toolbar-sep {
  width: 1px;
  height: 14px;
  background: #555;
  margin: 0 2px;
  flex-shrink: 0;
}

.add-btn {
  background: transparent;
  border: 1px dashed #e0e0e0;
  border-radius: 6px;
  color: #7a7a7a;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
  flex-shrink: 0;
}

.add-btn:hover {
  background: #0066cc;
  color: #fff;
  border-color: #0066cc;
}

#add-left-btn {
  width: 22px;
  align-self: stretch;
}

#add-right-btn {
  width: 22px;
  align-self: stretch;
}

#add-top-btn {
  height: 22px;
}

#add-bottom-btn {
  height: 22px;
}

#processing {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  z-index: 100;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 14px;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.374px;
  color: #1d1d1f;
}

#processing.visible { display: flex; }

@keyframes spin { to { transform: rotate(360deg); } }

.spinner {
  width: 24px;
  height: 24px;
  border: 2.5px solid #e0e0e0;
  border-top-color: #0066cc;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #aaa; }
</style>
</head>
<body>

<header>
  <h1>Image Stitcher</h1>
  <div class="header-right">
    <span id="status">Drop images to start</span>
    <label class="fit-toggle-label">
      <input type="checkbox" id="fit-toggle" class="fit-toggle-input">
      <span class="fit-toggle-switch"></span>
      <span>Fit images to layout</span>
    </label>
    <div id="overlap-control" style="display:none">
      <span>Overlap</span>
      <input type="range" id="overlap-slider" min="0" max="800" value="0" step="1">
      <button id="overlap-dec-btn" class="overlap-step-btn" title="Decrease by 1px">−</button>
      <span id="overlap-val">0 px</span>
      <button id="overlap-inc-btn" class="overlap-step-btn" title="Increase by 1px">+</button>
      <button id="zoom-toggle-btn" class="overlap-step-btn" title="Zoom into the seam edge" style="width:auto;padding:0 7px;font-size:12px">Zoom</button>
      <button id="overlap-auto-btn" class="active" title="Re-detect best overlap">Auto</button>
      <label class="fit-toggle-label" title="Cross-fade blend instead of hard paste">
        <input type="checkbox" id="overlap-fade-toggle" class="fit-toggle-input">
        <span class="fit-toggle-switch"></span>
        <span>Fade</span>
      </label>
    </div>
    <button id="stitch-btn" disabled>Stitch &amp; Preview</button>
  </div>
</header>

<div id="app-body">
  <div id="editor-pane">
    <div id="drop-zone" role="button" tabindex="0" aria-label="Drop images or click to browse">
      <svg width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
      </svg>
      <span>Drop images anywhere</span>
      <span class="hint">or click to browse &middot; JPEG, PNG, WebP</span>
      <input type="file" id="file-input" accept="image/*" multiple>
    </div>

    <div id="grid-outer">
      <button id="add-top-btn" class="add-btn" title="Add row above">+</button>
      <div id="grid-scroll">
        <button id="add-left-btn" class="add-btn" title="Add column before">+</button>
        <div id="grid"></div>
        <button id="add-right-btn" class="add-btn" title="Add column after">+</button>
      </div>
      <button id="add-bottom-btn" class="add-btn" title="Add row below">+</button>
    </div>
  </div>

  <div id="preview-pane">
    <div id="preview-img-wrap">
      <img id="preview-img" alt="Stitched preview">
    </div>
    <div id="zoom-panel" style="display:none;margin-top:6px">
      <canvas id="zoom-canvas" style="width:100%;display:block;image-rendering:pixelated;border-radius:5px;background:#2c2c2e;cursor:grab"></canvas>
      <div style="display:flex;align-items:center;gap:5px;margin-top:4px;font-size:11px;color:#888">
        <span>Zoom:</span>
        <button class="zoom-level-btn" data-zoom="2">2×</button>
        <button class="zoom-level-btn active" data-zoom="3">3×</button>
        <button class="zoom-level-btn" data-zoom="4">4×</button>
        <span style="margin-left:6px">Drag to pan · blue = overlap bounds</span>
      </div>
    </div>
    <a id="download-btn" download="stitched.jpg">Download</a>
  </div>
</div>

<div id="processing">
  <div class="spinner"></div>
  <span>Stitching&hellip;</span>
</div>

<script>
'use strict';

let grid = [];
let dragSrc = null;
let dragEnterCount = 0;
let _rafPending = false;
let _lastDragX = 0, _lastDragY = 0;

const appBody       = document.getElementById('app-body');
const dropZoneEl    = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const gridOuter     = document.getElementById('grid-outer');
const gridEl        = document.getElementById('grid');
const addLeftBtn    = document.getElementById('add-left-btn');
const addRightBtn   = document.getElementById('add-right-btn');
const addTopBtn     = document.getElementById('add-top-btn');
const addBottomBtn  = document.getElementById('add-bottom-btn');
const stitchBtn     = document.getElementById('stitch-btn');
const statusEl      = document.getElementById('status');
const processingEl  = document.getElementById('processing');
const previewImg    = document.getElementById('preview-img');
const downloadBtn   = document.getElementById('download-btn');
const overlapCtrl   = document.getElementById('overlap-control');
const overlapSlider = document.getElementById('overlap-slider');
const overlapValEl  = document.getElementById('overlap-val');
const overlapAutoBtn    = document.getElementById('overlap-auto-btn');
const overlapDecBtn     = document.getElementById('overlap-dec-btn');
const overlapIncBtn     = document.getElementById('overlap-inc-btn');
const overlapFadeToggle = document.getElementById('overlap-fade-toggle');
const zoomToggleBtn     = document.getElementById('zoom-toggle-btn');
const zoomPanel         = document.getElementById('zoom-panel');
const zoomCanvas        = document.getElementById('zoom-canvas');

let overlapIsAuto   = true;
let _zoomActive     = false;
let _zoomLevel      = 3;
let _zoomOffsetX    = null;   // null = re-center on next render
let _zoomOffsetY    = null;
let _lastZoomParams = null;

function _setOverlapPx(px) {
  overlapSlider.value = px;
  overlapValEl.textContent = px + ' px';
}

overlapSlider.addEventListener('input', () => {
  overlapIsAuto = false;
  overlapAutoBtn.classList.remove('active');
  overlapValEl.textContent = overlapSlider.value + ' px';
  scheduleLivePreview();
});

overlapAutoBtn.addEventListener('click', () => {
  overlapIsAuto = true;
  overlapAutoBtn.classList.add('active');
});

function _stepOverlap(delta) {
  const v = Math.min(parseInt(overlapSlider.max), Math.max(0, parseInt(overlapSlider.value) + delta));
  overlapSlider.value = v;
  overlapIsAuto = false;
  overlapAutoBtn.classList.remove('active');
  overlapValEl.textContent = v + ' px';
  scheduleLivePreview();
}
overlapDecBtn.addEventListener('click', () => _stepOverlap(-1));
overlapIncBtn.addEventListener('click', () => _stepOverlap( 1));

overlapFadeToggle.addEventListener('change', () => { scheduleLivePreview(); });

zoomToggleBtn.addEventListener('click', () => {
  _zoomActive  = !_zoomActive;
  _zoomOffsetX = null;
  _zoomOffsetY = null;
  zoomToggleBtn.classList.toggle('active', _zoomActive);
  if (!_zoomActive) zoomPanel.style.display = 'none';
  scheduleLivePreview();
});

document.querySelectorAll('.zoom-level-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const newLevel = parseInt(btn.dataset.zoom);
    if (_zoomOffsetX !== null) {
      // Keep the same viewport center when changing magnification
      const zc  = zoomCanvas;
      const cxSrc = _zoomOffsetX + Math.ceil(zc.width  / _zoomLevel) / 2;
      const cySrc = _zoomOffsetY + Math.ceil(zc.height / _zoomLevel) / 2;
      _zoomLevel   = newLevel;
      _zoomOffsetX = Math.round(cxSrc - Math.ceil(zc.width  / _zoomLevel) / 2);
      _zoomOffsetY = Math.round(cySrc - Math.ceil(zc.height / _zoomLevel) / 2);
    } else {
      _zoomLevel = newLevel;
    }
    document.querySelectorAll('.zoom-level-btn').forEach(b => b.classList.toggle('active', b === btn));
    if (_lastZoomParams) _renderZoomStrip(_lastZoomParams);
  });
});

let _zoomDragX = null, _zoomDragY = null;
zoomCanvas.addEventListener('mousedown', e => {
  _zoomDragX = e.clientX;
  _zoomDragY = e.clientY;
  zoomCanvas.style.cursor = 'grabbing';
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (_zoomDragX === null) return;
  const dx = e.clientX - _zoomDragX;
  const dy = e.clientY - _zoomDragY;
  _zoomDragX = e.clientX;
  _zoomDragY = e.clientY;
  if (_zoomOffsetX !== null) {
    _zoomOffsetX = Math.round(_zoomOffsetX - dx / _zoomLevel);
    _zoomOffsetY = Math.round(_zoomOffsetY - dy / _zoomLevel);
  }
  if (_lastZoomParams) _renderZoomStrip(_lastZoomParams);
});
window.addEventListener('mouseup', () => {
  if (_zoomDragX !== null) { _zoomDragX = null; _zoomDragY = null; zoomCanvas.style.cursor = 'grab'; }
});

const uid = () => Math.random().toString(36).slice(2, 9);

async function makeThumbnail(file, maxPx = 400) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const scale = Math.min(1, maxPx / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const cv = Object.assign(document.createElement('canvas'), { width: w, height: h });
  cv.getContext('2d').drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return cv.toDataURL('image/jpeg', 0.75);
}

const nRows     = () => grid.length;
const nCols     = () => (grid.length ? grid[0].length : 0);
const hasImages = () => grid.some(row => row.some(Boolean));
const imgCount  = () => grid.reduce((n, r) => n + r.filter(Boolean).length, 0);

function isFileDrag(e) {
  return e.dataTransfer && [...e.dataTransfer.types].includes('Files');
}

function padGrid() {
  const cols = Math.max(...grid.map(r => r.length), 0);
  for (const row of grid) while (row.length < cols) row.push(null);
}

function insertCol(after) {
  for (const row of grid) row.splice(after + 1, 0, null);
}

function insertRow(after) {
  grid.splice(after + 1, 0, Array(nCols()).fill(null));
}

function removeCell(r, c) {
  const cell = grid[r][c];
  if (cell) delete _bitmapCache[cell.id];
  grid[r][c] = null;
  compact();
}

function compact() {
  grid = grid.filter(row => row.some(Boolean));
  if (!grid.length) return;
  padGrid();
  const cols = nCols();
  const dead = Array.from({length: cols}, (_, i) => i)
    .filter(c => grid.every(row => !row[c]));
  if (dead.length) {
    const keep = new Set(Array.from({length: cols}, (_, i) => i).filter(i => !dead.includes(i)));
    for (let r = 0; r < grid.length; r++) {
      grid[r] = grid[r].filter((_, i) => keep.has(i));
    }
  }
}

function swapCells(r1, c1, r2, c2) {
  [grid[r1][c1], grid[r2][c2]] = [grid[r2][c2], grid[r1][c1]];
}

function moveCellToZone(srcR, srcC, dstR, dstC, zone) {
  if (zone === 'center') { swapCells(srcR, srcC, dstR, dstC); return; }
  const srcCell = grid[srcR][srcC];
  const dstCell = grid[dstR][dstC];
  grid[srcR][srcC] = null;
  compact();
  // Re-locate dstCell after compact may have shifted indices
  let nr = -1, nc = -1;
  outer: for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < grid[r].length; c++)
      if (grid[r][c] === dstCell) { nr = r; nc = c; break outer; }
  if (nr === -1) { grid.push([srcCell]); return; }
  if (zone === 'left')   { insertCol(nc - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'right')  { insertCol(nc);     grid[nr][nc + 1] = srcCell; }
  else if (zone === 'top')    { insertRow(nr - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'bottom') { insertRow(nr);     grid[nr + 1][nc] = srcCell; }
}

async function applyTransform(r, c, type) {
  const cell = grid[r][c];
  if (!cell) return;
  const objURL = URL.createObjectURL(cell.file);
  const img = new Image();
  img.src = objURL;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  URL.revokeObjectURL(objURL);
  const rotate = type === 'rotate-cw' || type === 'rotate-ccw';
  const canvas = document.createElement('canvas');
  canvas.width  = rotate ? img.height : img.width;
  canvas.height = rotate ? img.width  : img.height;
  const ctx = canvas.getContext('2d');
  ctx.save();
  if      (type === 'rotate-cw')  { ctx.translate(canvas.width, 0);  ctx.rotate( Math.PI / 2); }
  else if (type === 'rotate-ccw') { ctx.translate(0, canvas.height); ctx.rotate(-Math.PI / 2); }
  else if (type === 'flip-h')     { ctx.translate(canvas.width, 0);  ctx.scale(-1,  1); }
  else if (type === 'flip-v')     { ctx.translate(0, canvas.height); ctx.scale( 1, -1); }
  ctx.drawImage(img, 0, 0);
  ctx.restore();
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
  const newFile = new File([blob], cell.name, { type: 'image/jpeg' });
  const thumbURL = await makeThumbnail(newFile);
  grid[r][c] = { ...cell, dataURL: thumbURL, file: newFile };
  render();
}

async function makeCells(files) {
  const imgs = [...files].filter(f => f.type.startsWith('image/'));
  return Promise.all(imgs.map(async f => ({
    id: uid(), name: f.name, file: f, dataURL: await makeThumbnail(f),
  })));
}

async function initFromFiles(files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  grid = [cells];
  render();
}

async function dropFilesOnCell(r, c, zone, files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  const first = cells[0];
  if (zone === 'center') {
    grid[r][c] = first;
    for (let i = 1; i < cells.length; i++) { insertCol(c + i - 1); grid[r][c + i] = cells[i]; }
  } else if (zone === 'right')  { insertCol(c);     grid[r][c + 1] = first; }
  else if   (zone === 'left')   { insertCol(c - 1); grid[r][c]     = first; }
  else if   (zone === 'bottom') { insertRow(r);     grid[r + 1][c] = first; }
  else if   (zone === 'top')    { insertRow(r - 1); grid[r][c]     = first; }
  render();
}

function render() {
  if (!hasImages()) {
    gridOuter.style.display = 'none';
    dropZoneEl.style.display = 'flex';
    stitchBtn.disabled = true;
    statusEl.textContent = 'Drop images to start';
    return;
  }
  dropZoneEl.style.display = 'none';
  gridOuter.style.display = 'flex';
  const rows = nRows(), cols = nCols();
  gridEl.style.gridTemplateColumns = `repeat(${cols}, auto)`;
  gridEl.style.gridTemplateRows    = `repeat(${rows}, auto)`;
  gridEl.innerHTML = '';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid[r][c];
      gridEl.appendChild(cell ? buildCell(cell, r, c) : buildEmptyCell(r, c));
    }
  }
  const n = imgCount();
  statusEl.textContent = `${n} image${n !== 1 ? 's' : ''} · ${rows}×${cols} grid`;
  stitchBtn.disabled = n < 2;
  overlapCtrl.style.display = n >= 2 ? 'flex' : 'none';
}

function buildCell(cell, r, c) {
  const el = document.createElement('div');
  el.className = 'cell';
  el.draggable = true;
  el.dataset.r = r;
  el.dataset.c = c;

  const img = document.createElement('img');
  img.src = cell.dataURL;
  img.alt = cell.name;
  el.appendChild(img);

  const overlay = document.createElement('div');
  overlay.className = 'drop-overlay';
  for (const z of ['center','left','right','top','bottom']) {
    const dz = document.createElement('div');
    dz.className = `dz dz-${z}`;
    overlay.appendChild(dz);
  }
  el.appendChild(overlay);

  const rmBtn = document.createElement('button');
  rmBtn.className = 'btn-remove';
  rmBtn.title = 'Remove';
  rmBtn.textContent = '×';
  rmBtn.addEventListener('click', e => { e.stopPropagation(); removeCell(r, c); render(); });
  el.appendChild(rmBtn);

  const toolbar = document.createElement('div');
  toolbar.className = 'cell-toolbar';
  toolbar.innerHTML =
    '<button class="btn-transform" title="Rotate CCW" aria-label="Rotate counter-clockwise">↺</button>' +
    '<button class="btn-transform" title="Rotate CW" aria-label="Rotate clockwise">↻</button>' +
    '<button class="btn-transform" title="Flip horizontal" aria-label="Flip horizontal">↔</button>' +
    '<button class="btn-transform" title="Flip vertical" aria-label="Flip vertical">↕</button>';
  const [rotateCCW, rotateCW, flipH, flipV] = toolbar.children;
  rotateCCW.addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'rotate-ccw'); });
  rotateCW .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'rotate-cw'); });
  flipH    .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'flip-h'); });
  flipV    .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'flip-v'); });
  el.appendChild(toolbar);

  el.addEventListener('dragstart', e => {
    dragSrc = { r, c };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', `${r},${c}`);
    requestAnimationFrame(() => el.classList.add('dragging'));
  });
  el.addEventListener('dragend', () => {
    el.classList.remove('dragging');
    dragSrc = null;
    clearAllHighlights();
  });
  el.addEventListener('dragover', e => {
    e.preventDefault();
    if (dragSrc) activateZone(overlay, zoneAt(e.clientX, e.clientY, el));
  });
  el.addEventListener('dragleave', e => {
    if (dragSrc && !el.contains(e.relatedTarget)) deactivateZone(overlay);
  });
  el.addEventListener('drop', e => {
    e.preventDefault();
    deactivateZone(overlay);
    if (e.dataTransfer.files.length) return;
    if (!dragSrc) return;
    const { r: sr, c: sc } = dragSrc;
    if (sr !== r || sc !== c) { moveCellToZone(sr, sc, r, c, zoneAt(e.clientX, e.clientY, el)); render(); }
  });

  return el;
}

function buildEmptyCell(r, c) {
  const el = document.createElement('div');
  el.className = 'cell empty';
  el.dataset.r = r;
  el.dataset.c = c;
  el.textContent = '+';
  el.addEventListener('dragover', e => { e.preventDefault(); });
  el.addEventListener('click', () => {
    fileInput.dataset.targetR = r;
    fileInput.dataset.targetC = c;
    fileInput.click();
  });
  return el;
}

function zoneAt(clientX, clientY, el) {
  const rect = el.getBoundingClientRect();
  const nx = (clientX - (rect.left + rect.right)  / 2) / (rect.width  / 2);
  const ny = (clientY - (rect.top  + rect.bottom) / 2) / (rect.height / 2);
  // Small central region snaps to center (replace), everywhere else snaps to nearest side
  if (Math.abs(nx) < 0.33 && Math.abs(ny) < 0.33) return 'center';
  return Math.abs(nx) >= Math.abs(ny)
    ? (nx > 0 ? 'right' : 'left')
    : (ny > 0 ? 'bottom' : 'top');
}

function activateZone(overlay, zone) {
  for (const dz of overlay.querySelectorAll('.dz')) {
    dz.classList.toggle('active', dz.classList.contains(`dz-${zone}`));
  }
}
function deactivateZone(overlay) {
  overlay.querySelectorAll('.dz.active').forEach(d => d.classList.remove('active'));
}
function clearAllHighlights() {
  document.querySelectorAll('.dz.active').forEach(d => d.classList.remove('active'));
  document.querySelectorAll('.cell.empty.drag-over-empty').forEach(d => d.classList.remove('drag-over-empty'));
  dropZoneEl.classList.remove('drag-over');
}

function findNearestZone(clientX, clientY) {
  if (!hasImages()) return { type: 'empty' };
  const allCells = document.querySelectorAll('.cell[data-r][data-c]');
  if (!allCells.length) return { type: 'empty' };
  function distToRect(rect) {
    const dx = Math.max(0, rect.left - clientX, clientX - rect.right);
    const dy = Math.max(0, rect.top  - clientY, clientY - rect.bottom);
    return Math.hypot(dx, dy);
  }
  let nearest = null, minDist = Infinity;
  for (const el of allCells) {
    const rect = el.getBoundingClientRect();
    const d = distToRect(rect);
    if (d < minDist) { minDist = d; nearest = { el, rect }; }
  }
  const r = +nearest.el.dataset.r;
  const c = +nearest.el.dataset.c;
  if (nearest.el.classList.contains('empty')) return { type: 'cell', r, c, zone: 'center', el: nearest.el };
  return { type: 'cell', r, c, zone: zoneAt(clientX, clientY, nearest.el), el: nearest.el };
}

function updateDropHighlight(clientX, clientY) {
  clearAllHighlights();
  const target = findNearestZone(clientX, clientY);
  if (target.type === 'empty') { dropZoneEl.classList.add('drag-over'); return; }
  if (target.el.classList.contains('empty')) {
    target.el.classList.add('drag-over-empty');
  } else {
    const overlay = target.el.querySelector('.drop-overlay');
    if (overlay) activateZone(overlay, target.zone);
  }
}

document.addEventListener('dragenter', e => {
  if (!isFileDrag(e)) return;
  dragEnterCount++;
});
document.addEventListener('dragleave', e => {
  if (!isFileDrag(e)) return;
  dragEnterCount--;
  if (dragEnterCount <= 0) {
    dragEnterCount = 0;
    clearAllHighlights();
  }
});
document.addEventListener('dragover', e => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  _lastDragX = e.clientX;
  _lastDragY = e.clientY;
  if (!_rafPending) {
    _rafPending = true;
    requestAnimationFrame(() => {
      updateDropHighlight(_lastDragX, _lastDragY);
      _rafPending = false;
    });
  }
});
document.addEventListener('drop', e => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  dragEnterCount = 0;
  clearAllHighlights();
  const target = findNearestZone(e.clientX, e.clientY);
  if (target.type === 'empty') initFromFiles(e.dataTransfer.files);
  else dropFilesOnCell(target.r, target.c, target.zone, e.dataTransfer.files);
});
document.addEventListener('dragend', () => {
  dragEnterCount = 0;
  clearAllHighlights();
});

dropZoneEl.addEventListener('click', () => fileInput.click());
dropZoneEl.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener('change', e => {
  if (!e.target.files.length) { fileInput.value = ''; return; }
  const tr = fileInput.dataset.targetR;
  const tc = fileInput.dataset.targetC;
  if (tr !== undefined && tc !== undefined) {
    dropFilesOnCell(+tr, +tc, 'center', e.target.files);
    delete fileInput.dataset.targetR;
    delete fileInput.dataset.targetC;
  } else {
    initFromFiles(e.target.files);
  }
  fileInput.value = '';
});

addLeftBtn.addEventListener('click', () => { if (hasImages()) { insertCol(-1); render(); } });
addRightBtn.addEventListener('click', () => { if (hasImages()) { insertCol(nCols() - 1); render(); } });
addTopBtn.addEventListener('click', () => { if (hasImages()) { insertRow(-1); render(); } });
addBottomBtn.addEventListener('click', () => { if (hasImages()) { insertRow(nRows() - 1); render(); } });

function _buildFormData() {
  const formData = new FormData();
  const layoutIds = grid.map(row => row.map(cell => cell ? cell.id : null));
  formData.append('layout', JSON.stringify(layoutIds));
  formData.append('fit', document.getElementById('fit-toggle').checked);
  formData.append('overlap', overlapIsAuto ? 'auto' : overlapSlider.value);
  formData.append('overlap_blend', overlapFadeToggle.checked ? 'fade' : 'hard');
  const seen = new Set();
  for (const row of grid) {
    for (const cell of row) {
      if (cell && !seen.has(cell.id)) {
        formData.append(cell.id, cell.file, cell.name);
        seen.add(cell.id);
      }
    }
  }
  return formData;
}

let _previewObjectURL = null;

// ── Live client-side preview ──────────────────────────────────────────────────
const _bitmapCache = {};  // cell.id → ImageBitmap

function _renderZoomStrip({ cv, dw0, ov, cvW, cvH }) {
  const ZOOM   = _zoomLevel;
  const zc     = zoomCanvas;
  const panelW = zoomPanel.offsetWidth || 600;
  zc.width     = panelW;
  zc.height    = 200;

  // How many source pixels fit in this viewport at the chosen zoom level
  const vpW = Math.ceil(zc.width  / ZOOM);
  const vpH = Math.ceil(zc.height / ZOOM);

  // On first render (or after toggle/reset), center the view on the seam
  const seamL = dw0 - ov;
  if (_zoomOffsetX === null) {
    _zoomOffsetX = Math.round(seamL + ov / 2 - vpW / 2);
    _zoomOffsetY = Math.round((cvH - vpH) / 2);
  }

  const srcX = Math.max(0, Math.min(cvW - vpW, _zoomOffsetX));
  const srcY = Math.max(0, Math.min(cvH - vpH, _zoomOffsetY));

  const zctx = zc.getContext('2d');
  zctx.clearRect(0, 0, zc.width, zc.height);
  zctx.imageSmoothingEnabled = false;
  zctx.drawImage(cv, srcX, srcY, vpW, vpH, 0, 0, zc.width, zc.height);

  // Mark overlap left/right boundaries
  zctx.strokeStyle = 'rgba(0,180,255,0.75)';
  zctx.lineWidth   = 2;
  for (const natX of [seamL, dw0]) {
    const mx = (natX - srcX) * ZOOM;
    if (mx >= 0 && mx <= zc.width) {
      zctx.beginPath(); zctx.moveTo(mx, 0); zctx.lineTo(mx, zc.height); zctx.stroke();
    }
  }
  zoomPanel.style.display = 'block';
}

async function _getBitmap(cell) {
  if (!_bitmapCache[cell.id]) {
    _bitmapCache[cell.id] = await createImageBitmap(cell.file, { imageOrientation: 'from-image' });
  }
  return _bitmapCache[cell.id];
}

async function updateLivePreview() {
  const rows = nRows(), cols = nCols();
  const is1Row = rows === 1 && cols >= 2;
  const is1Col = cols === 1 && rows >= 2;
  if (!hasImages() || (!is1Row && !is1Col)) return;

  const direction = is1Row ? 'horizontal' : 'vertical';
  const cells = is1Row
    ? grid[0].filter(Boolean)
    : grid.map(r => r[0]).filter(Boolean);
  if (cells.length < 2) return;

  let bitmaps;
  try { bitmaps = await Promise.all(cells.map(_getBitmap)); }
  catch (_) { return; }

  const overlap = Math.max(0, parseInt(overlapSlider.value) || 0);
  const isFade  = overlapFadeToggle.checked;

  // Natural canvas size
  let natW, natH;
  if (direction === 'horizontal') {
    natW = bitmaps.reduce((s, b) => s + b.width,  0) - overlap * (bitmaps.length - 1);
    natH = Math.max(...bitmaps.map(b => b.height));
  } else {
    natW = Math.max(...bitmaps.map(b => b.width));
    natH = bitmaps.reduce((s, b) => s + b.height, 0) - overlap * (bitmaps.length - 1);
  }
  if (natW <= 0 || natH <= 0) return;

  // Scale down to ≤2000px on longest side for speed
  const scale = Math.min(1, 2000 / Math.max(natW, natH));
  const cvW = Math.max(1, Math.round(natW * scale));
  const cvH = Math.max(1, Math.round(natH * scale));
  const ov  = Math.round(overlap * scale);

  const cv  = document.createElement('canvas');
  cv.width  = cvW;
  cv.height = cvH;
  const ctx = cv.getContext('2d');

  if (direction === 'horizontal') {
    let x = 0;
    for (let i = 0; i < bitmaps.length; i++) {
      const bmp = bitmaps[i];
      const dw  = Math.round(bmp.width  * scale);
      const dh  = Math.round(bmp.height * scale);
      if (i > 0) x -= ov;
      if (isFade && i > 0 && ov > 0) {
        // Fade: draw the incoming image with a left-to-right alpha ramp
        const tmp = document.createElement('canvas');
        tmp.width = dw; tmp.height = dh;
        const tc  = tmp.getContext('2d');
        tc.drawImage(bmp, 0, 0, dw, dh);
        const grad = tc.createLinearGradient(0, 0, ov, 0);
        grad.addColorStop(0, 'rgba(0,0,0,0)');
        grad.addColorStop(1, 'rgba(0,0,0,1)');
        tc.globalCompositeOperation = 'destination-in';
        tc.fillStyle = grad;
        tc.fillRect(0, 0, ov, dh);
        ctx.drawImage(tmp, x, 0);
      } else {
        ctx.drawImage(bmp, x, 0, dw, dh);
      }
      x += dw;
    }
  } else {
    let y = 0;
    for (let i = 0; i < bitmaps.length; i++) {
      const bmp = bitmaps[i];
      const dw  = Math.round(bmp.width  * scale);
      const dh  = Math.round(bmp.height * scale);
      if (i > 0) y -= ov;
      if (isFade && i > 0 && ov > 0) {
        const tmp = document.createElement('canvas');
        tmp.width = dw; tmp.height = dh;
        const tc  = tmp.getContext('2d');
        tc.drawImage(bmp, 0, 0, dw, dh);
        const grad = tc.createLinearGradient(0, 0, 0, ov);
        grad.addColorStop(0, 'rgba(0,0,0,0)');
        grad.addColorStop(1, 'rgba(0,0,0,1)');
        tc.globalCompositeOperation = 'destination-in';
        tc.fillStyle = grad;
        tc.fillRect(0, 0, dw, ov);
        ctx.drawImage(tmp, 0, y);
      } else {
        ctx.drawImage(bmp, 0, y, dw, dh);
      }
      y += dh;
    }
  }

  // Zoom view of seam edge (horizontal layouts only)
  if (_zoomActive && direction === 'horizontal' && ov > 0) {
    const dw0 = Math.round(bitmaps[0].width * scale);
    _lastZoomParams = { cv, dw0, ov, cvW, cvH };
    _renderZoomStrip(_lastZoomParams);
  } else if (!_zoomActive) {
    zoomPanel.style.display = 'none';
  }

  cv.toBlob(blob => {
    if (!blob) return;
    if (_previewObjectURL) URL.revokeObjectURL(_previewObjectURL);
    _previewObjectURL = URL.createObjectURL(blob);
    previewImg.onload = () => {
      const landscape = cvW >= cvH;
      appBody.classList.toggle('layout-side',  !landscape);
      appBody.classList.toggle('layout-stack',  landscape);
      appBody.classList.add('has-preview');
    };
    previewImg.src = _previewObjectURL;
    statusEl.textContent = `${cells.length} images · ${overlap}px overlap`;
  }, 'image/jpeg', 0.92);
}

let _livePreviewPending = false;
function scheduleLivePreview() {
  if (_livePreviewPending) return;
  _livePreviewPending = true;
  requestAnimationFrame(() => { _livePreviewPending = false; updateLivePreview(); });
}

stitchBtn.addEventListener('click', async () => {
  if (!hasImages()) return;
  processingEl.classList.add('visible');
  stitchBtn.textContent = 'Stitching…';
  stitchBtn.disabled = true;
  try {
    const res = await fetch('/preview', { method: 'POST', body: _buildFormData() });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(`Server error ${res.status}: ${msg}`);
    }
    const appliedOverlap = res.headers.get('X-Overlap-Px');
    if (appliedOverlap !== null) {
      const px = parseInt(appliedOverlap, 10);
      _setOverlapPx(px);
      if (overlapIsAuto) overlapAutoBtn.classList.add('active');
    }
    const blob = await res.blob();
    if (_previewObjectURL) URL.revokeObjectURL(_previewObjectURL);
    _previewObjectURL = URL.createObjectURL(blob);
    previewImg.onload = () => {
      const landscape = previewImg.naturalWidth >= previewImg.naturalHeight;
      appBody.classList.toggle('layout-side', !landscape);
      appBody.classList.toggle('layout-stack', landscape);
      appBody.classList.add('has-preview');
    };
    previewImg.src = _previewObjectURL;
    const kb = Math.round(blob.size / 1024);
    statusEl.textContent = `Stitched · ${kb} KB`;
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    processingEl.classList.remove('visible');
    stitchBtn.textContent = 'Stitch & Preview';
    stitchBtn.disabled = imgCount() < 2;
  }
});

downloadBtn.addEventListener('click', async e => {
  e.preventDefault();
  if (!hasImages()) return;
  const prev = downloadBtn.textContent;
  downloadBtn.textContent = 'Downloading…';
  downloadBtn.style.pointerEvents = 'none';
  try {
    const res = await fetch('/stitch', { method: 'POST', body: _buildFormData() });
    if (!res.ok) throw new Error(res.statusText);
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    Object.assign(document.createElement('a'), { href: url, download: 'stitched.jpg' }).click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (err) {
    statusEl.textContent = 'Download error: ' + err.message;
  } finally {
    downloadBtn.textContent = prev;
    downloadBtn.style.pointerEvents = '';
  }
});
</script>
</body>
</html>"""


def _stitch_grid(layout, paths, td, fit=False, overlap=0, overlap_blend="hard"):
    """
    Composite a 2D grid of images into a single JPEG.

    layout  — 2D list of file IDs (strings) or None for empty cells
    paths   — {id: absolute_path}
    td      — temp directory for the output file

    Column widths = max image width per column.
    Row heights   = max image height per row.
    Empty cells   = black fill.
    Encoding params taken from the first non-None cell (top-left scan order).
    """
    first_path = None
    num_rows = len(layout)
    num_cols = max(len(r) for r in layout)
    for row in layout:
        while len(row) < num_cols:
            row.append(None)

    col_widths  = [0] * num_cols
    row_heights = [0] * num_rows
    grid = []
    for r, row in enumerate(layout):
        grid_row = []
        for c, fid in enumerate(row):
            if fid and fid in paths:
                img = ImageOps.exif_transpose(Image.open(paths[fid])).convert("RGB")
                if first_path is None:
                    first_path = paths[fid]
                col_widths[c]  = max(col_widths[c],  img.width)
                row_heights[r] = max(row_heights[r], img.height)
                grid_row.append(img)
            else:
                grid_row.append(None)
        grid.append(grid_row)

    if first_path is None:
        return None

    if fit:
        if num_rows == 1:
            # Single-row horizontal: scale each image to the tallest row height
            target_h = row_heights[0]
            for c, img in enumerate(grid[0]):
                if img is not None and img.height != target_h:
                    new_w = round(img.width * target_h / img.height)
                    grid[0][c] = img.resize((new_w, target_h), Image.LANCZOS)
            col_widths = [
                grid[0][c].width if grid[0][c] is not None else 0
                for c in range(num_cols)
            ]
        else:
            # Vertical single-column or multi-row grid: scale each image to its column width
            for r, row in enumerate(grid):
                for c, img in enumerate(row):
                    if img is not None:
                        target_w = col_widths[c]
                        if img.width != target_w:
                            new_h = round(img.height * target_w / img.width)
                            grid[r][c] = img.resize((target_w, new_h), Image.LANCZOS)
            row_heights = [0] * num_rows
            for r, row in enumerate(grid):
                for c, img in enumerate(row):
                    if img is not None:
                        row_heights[r] = max(row_heights[r], img.height)

    # Clamp overlap to something sensible for 1D layouts
    if overlap > 0:
        if num_rows == 1 and num_cols > 1:
            overlap = min(overlap, min(col_widths) - 1)
        elif num_cols == 1 and num_rows > 1:
            overlap = min(overlap, min(row_heights) - 1)
        else:
            overlap = 0  # non-1D grid: ignore overlap
        overlap = max(0, overlap)

    if overlap > 0 and num_rows == 1:
        total_w = sum(col_widths) - overlap * (num_cols - 1)
        total_h = row_heights[0] if row_heights else 0
    elif overlap > 0 and num_cols == 1:
        total_w = col_widths[0] if col_widths else 0
        total_h = sum(row_heights) - overlap * (num_rows - 1)
    else:
        total_w = sum(col_widths)
        total_h = sum(row_heights)

    if total_w <= 0 or total_h <= 0:
        return None

    MAX_CANVAS_PX = 16384
    if total_w > MAX_CANVAS_PX or total_h > MAX_CANVAS_PX:
        return None

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))

    if overlap > 0 and num_rows == 1:
        x = 0
        for c in range(num_cols):
            img = grid[0][c]
            if img is None:
                x += col_widths[c]
                continue
            if c > 0:
                x -= overlap
                if overlap_blend == "fade":
                    blend_h    = min(img.height, canvas.height)
                    left_crop  = canvas.crop((x, 0, x + overlap, blend_h))
                    right_crop = img.crop((0, 0, overlap, blend_h))
                    mask = _make_blend_mask(overlap, blend_h, "horizontal")
                    canvas.paste(Image.composite(right_crop, left_crop, mask), (x, 0))
                    if img.width > overlap:
                        canvas.paste(img.crop((overlap, 0, img.width, img.height)), (x + overlap, 0))
                else:
                    canvas.paste(img, (x, 0))
            else:
                canvas.paste(img, (x, 0))
            x += col_widths[c]
    elif overlap > 0 and num_cols == 1:
        y = 0
        for r in range(num_rows):
            img = grid[r][0]
            if img is None:
                y += row_heights[r]
                continue
            if r > 0:
                y -= overlap
                if overlap_blend == "fade":
                    blend_w  = min(img.width, canvas.width)
                    top_crop = canvas.crop((0, y, blend_w, y + overlap))
                    bot_crop = img.crop((0, 0, blend_w, overlap))
                    mask = _make_blend_mask(blend_w, overlap, "vertical")
                    canvas.paste(Image.composite(bot_crop, top_crop, mask), (0, y))
                    if img.height > overlap:
                        canvas.paste(img.crop((0, overlap, img.width, img.height)), (0, y + overlap))
                else:
                    canvas.paste(img, (0, y))
            else:
                canvas.paste(img, (0, y))
            y += row_heights[r]
    else:
        y_off = 0
        for r, row in enumerate(grid):
            x_off = 0
            for c, img in enumerate(row):
                if img is not None:
                    canvas.paste(img, (x_off, y_off))
                x_off += col_widths[c]
            y_off += row_heights[r]

    output_path = os.path.join(td, "output.jpg")
    qtables, quality, subsampling, is_grayscale = _jpeg_params(first_path)

    if is_grayscale:
        canvas = canvas.convert("L")
        if qtables:
            canvas.save(output_path, "JPEG", qtables=qtables)
        else:
            canvas.save(output_path, "JPEG", quality=quality)
    elif qtables:
        canvas.save(output_path, "JPEG", qtables=qtables, subsampling=subsampling)
    else:
        canvas.save(output_path, "JPEG", quality=quality, subsampling=subsampling)

    return output_path


def _run_web_server(port=5001):
    """Launch the drag-and-drop web UI and open it in the default browser."""
    if not _HAS_FLASK:
        print(
            "Flask is required for the web UI.\n"
            "Install it with:  pip3 install flask --break-system-packages\n"
            "Or install via Homebrew:  brew upgrade jpegconcat"
        )
        sys.exit(1)

    import io, json as _json, threading, webbrowser
    from flask import Flask, abort, request, send_file

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB total upload limit
    # Suppress Flask startup banner
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return _WEB_UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    def _validate_layout():
        layout_raw = request.form.get("layout")
        if not layout_raw:
            abort(400, "Missing layout")
        try:
            layout = _json.loads(layout_raw)
            if not isinstance(layout, list):
                raise ValueError
            layout = [row for row in layout if isinstance(row, list) and any(c is not None for c in row)]
        except (ValueError, _json.JSONDecodeError):
            abort(400, "Invalid layout")
        if not layout:
            abort(400, "Empty layout")
        num_cols = max(len(r) for r in layout)
        for row in layout:
            while len(row) < num_cols:
                row.append(None)
        non_empty = [c for c in range(num_cols) if any(r[c] is not None for r in layout)]
        if not non_empty:
            abort(400, "No images in layout")
        return [[row[c] for c in non_empty] for row in layout]

    def _stitch_to_buf():
        layout = _validate_layout()
        fit = request.form.get("fit", "false").lower() == "true"
        overlap_raw   = request.form.get("overlap", "auto")
        overlap_blend = request.form.get("overlap_blend", "hard")

        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for key, f in request.files.items():
                safe_key = re.sub(r'[^A-Za-z0-9_-]', '_', key)[:64]
                ext = os.path.splitext(f.filename)[1] or ".jpg"
                ext = re.sub(r'[^A-Za-z0-9.]', '', ext)[:16]
                dest = os.path.join(td, safe_key + ext)
                f.save(dest)
                paths[key] = dest

            if overlap_raw == "auto":
                n_rows = len(layout)
                n_cols = max(len(r) for r in layout)
                direction = "horizontal" if n_rows == 1 else "vertical"
                imgs = []
                for row in layout:
                    for fid in row:
                        if fid and fid in paths:
                            try:
                                imgs.append(
                                    ImageOps.exif_transpose(Image.open(paths[fid])).convert("RGB")
                                )
                            except Exception:
                                pass
                overlap = _suggest_overlap(imgs, direction) if len(imgs) >= 2 else 0
            else:
                try:
                    overlap = max(0, int(overlap_raw))
                except (ValueError, TypeError):
                    overlap = 0

            output = _stitch_grid(layout, paths, td, fit=fit, overlap=overlap, overlap_blend=overlap_blend)
            if output is None:
                abort(500, "Nothing to stitch")
            buf = io.BytesIO(open(output, "rb").read())
        buf.seek(0)
        return buf, overlap, overlap_blend

    @app.route("/stitch", methods=["POST"])
    def stitch():
        buf, _, __ = _stitch_to_buf()
        return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="stitched.jpg")

    @app.route("/preview", methods=["POST"])
    def preview():
        from flask import make_response
        buf, overlap, overlap_blend = _stitch_to_buf()
        resp = make_response(send_file(buf, mimetype="image/jpeg", as_attachment=False))
        resp.headers["X-Overlap-Px"]    = str(overlap)
        resp.headers["X-Overlap-Blend"] = overlap_blend
        return resp

    url = f"http://localhost:{port}"
    print(f"Image Stitcher → {url}  (Ctrl-C to quit)")

    def _open():
        import time; time.sleep(0.6)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    try:
        app.run(debug=False, port=port)
    except OSError as exc:
        if "Address already in use" in str(exc):
            print(f"Port {port} is in use. Try: jpegconcat --web --port {port + 1}")
            sys.exit(1)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate JPEG images preserving encoding params.",
        usage=(
            "%(prog)s img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg]"
            " [--direction h|v|auto] [--order auto|as-given]\n"
            "       %(prog)s --web [--port PORT]"
        ),
    )
    _DIR_ALIASES = {"h": "horizontal", "v": "vertical", "a": "auto"}

    parser.add_argument("images",       nargs="*", help="Input image paths")
    parser.add_argument("--output", "-o", default=None, help="Output path (auto-generated if omitted)")
    parser.add_argument("--direction", "-d",
                        choices=["horizontal", "vertical", "auto", "h", "v", "a"],
                        default="auto",
                        metavar="{horizontal|h, vertical|v, auto|a}",
                        help="Layout direction (default: auto)")
    parser.add_argument("--order",      choices=["auto", "as-given"],  default="auto")
    parser.add_argument("--fit", action="store_true",
                        help="Resize images to share the layout dimension before stitching "
                             "(height for horizontal, width for vertical)")
    parser.add_argument("--overlap", type=int, default=0, metavar="PX",
                        help="Pixels to overlap between adjacent images (default: 0)")
    parser.add_argument("--overlap-blend", choices=["hard", "fade"], default="hard",
                        help="Overlap blend mode: hard=direct paste (default), fade=cross-fade")
    parser.add_argument("--web",  "-w", action="store_true",
                        help="Launch the drag-and-drop web UI in your browser")
    parser.add_argument("--port",       type=int, default=5001,
                        help="Port for --web mode (default: 5001)")
    args = parser.parse_args()

    if args.web:
        _run_web_server(port=args.port)
        return

    if not args.images:
        parser.error("the following arguments are required: images  (or use --web for the browser UI)")

    direction = _DIR_ALIASES.get(args.direction, args.direction)

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        print(f"Error: files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    # Defer output-path computation to concat_images to avoid opening
    # input files twice (once here for format detection, once for processing).
    if args.overlap < 0:
        print("Error: --overlap must be >= 0", file=sys.stderr)
        sys.exit(1)

    concat_images(args.images, args.output, direction, args.order, fit=args.fit, overlap=args.overlap, overlap_blend=args.overlap_blend)


if __name__ == "__main__":
    main()
