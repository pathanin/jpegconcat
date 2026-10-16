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
import concurrent.futures
import io
import os
import re
import shutil
import struct
import subprocess
import sys

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
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None, 85, 2, False
    return _jpeg_params_from_bytes(raw)


def _jpeg_params_from_bytes(raw):
    """Same as _jpeg_params, but from bytes already read into memory (avoids a
    second disk read when the caller already has the raw bytes, e.g. because
    it also needs them to decode pixels)."""
    qtables = None
    quality = 85
    subsampling = 2
    is_grayscale = False

    try:
        subsampling = _parse_sof_subsampling(raw)
        _img = Image.open(io.BytesIO(raw))
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


def _decode_rgb(raw):
    """Decode image bytes to an upright RGB Image.

    Equivalent to ImageOps.exif_transpose(Image.open(...)).convert("RGB"),
    minus the two full-resolution copies that chain makes when the image is
    already upright and already RGB (the common case for camera JPEGs).
    """
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.getexif().get(0x0112, 1) != 1:
        img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# Pillow releases the GIL while decoding/encoding, so a thread pool decodes
# several large images concurrently.
_DECODE_WORKERS = min(8, os.cpu_count() or 1)


def _decode_many(raws):
    """_decode_rgb over a list of byte strings, in parallel, preserving order."""
    if len(raws) <= 1:
        return [_decode_rgb(r) for r in raws]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(_DECODE_WORKERS, len(raws))) as ex:
        return list(ex.map(_decode_rgb, raws))


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

    # Build a blank black canvas JPEG with the source's quantization tables
    if direction == "horizontal":
        canvas_size = (sum(img.width for img in images), max(img.height for img in images))
    else:
        canvas_size = (max(img.width for img in images), sum(img.height for img in images))

    canvas_buf = io.BytesIO()
    blank = Image.new("RGB", canvas_size, (0, 0, 0))
    if qtables:
        blank.save(canvas_buf, "JPEG", qtables=qtables, subsampling=subsampling)
    else:
        blank.save(canvas_buf, "JPEG", quality=85, subsampling=subsampling)
    current = canvas_buf.getvalue()
    del blank, canvas_buf

    # Drop each source image into the canvas at its offset (DCT-level, no
    # re-encode). The growing canvas is piped through stdin/stdout rather
    # than staged in temp files.
    # -copy all: metadata from the first source image is preserved
    x = y = 0
    for path, img in zip(image_paths, images):
        result = subprocess.run(
            [jpegtran, "-copy", "all", "-drop", f"+{x}+{y}", path],
            input=current, capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            return False
        current = result.stdout
        if direction == "horizontal":
            x += img.width
        else:
            y += img.height

    with open(output_path, "wb") as fout:
        fout.write(current)

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


def _paste_overlap(canvas, img, pos, overlap, overlap_blend, direction):
    """Paste img onto canvas at pos, optionally cross-fading the leading
    `overlap`-px band (along direction) with whatever is already on the
    canvas there. Shared by concat_images and _stitch_grid.

    overlap=0 (e.g. the first image in a sequence) always does a plain paste.
    """
    x, y = pos
    if overlap > 0 and overlap_blend == "fade":
        if direction == "horizontal":
            blend_h   = min(img.height, canvas.height - y)
            prev_crop = canvas.crop((x, y, x + overlap, y + blend_h))
            new_crop  = img.crop((0, 0, overlap, blend_h))
            mask      = _make_blend_mask(overlap, blend_h, "horizontal")
            canvas.paste(Image.composite(new_crop, prev_crop, mask), (x, y))
            if img.width > overlap:
                canvas.paste(img.crop((overlap, 0, img.width, img.height)), (x + overlap, y))
        else:
            blend_w   = min(img.width, canvas.width - x)
            prev_crop = canvas.crop((x, y, x + blend_w, y + overlap))
            new_crop  = img.crop((0, 0, blend_w, overlap))
            mask      = _make_blend_mask(blend_w, overlap, "vertical")
            canvas.paste(Image.composite(new_crop, prev_crop, mask), (x, y))
            if img.height > overlap:
                canvas.paste(img.crop((0, overlap, img.width, img.height)), (x, y + overlap))
    else:
        canvas.paste(img, (x, y))


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
    # Read every input exactly once; headers, encoding params and pixels all
    # come from these bytes.
    raws = []
    for p in image_paths:
        with open(p, "rb") as f:
            raws.append(f.read())
    raw_of = dict(zip(image_paths, raws))
    heads = [Image.open(io.BytesIO(r)) for r in raws]  # header-only, no pixel decode

    _first = heads[0]
    _first_format = _first.format
    _first_exif = _first.info.get('exif') if _first_format == "JPEG" else None
    if _first_exif and _first.getexif().get(0x0112, 1) != 1:
        # Pixels get exif_transpose'd below, so the copied EXIF must not ask
        # viewers to rotate them a second time.
        _ex = Image.Exif()  # fresh copy: getexif() is cached on the image
        _ex.load(_first_exif)
        _ex[0x0112] = 1
        _first_exif = _ex.tobytes()

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

    input_formats = [h.format for h in heads]
    # jpegtran -drop works on the stored (unrotated, unresized) DCT data, so
    # the lossless path is only valid when the decoded pixels match the file.
    pixels_differ_from_file = any(h.getexif().get(0x0112, 1) != 1 for h in heads)

    preserve_order  = (order != "auto")
    fix_direction = None if direction == "auto" else direction

    use_edge_match = (
        _HAS_NUMPY
        and len(image_paths) == 2
        and (not preserve_order or fix_direction is None)  # at least one thing to decide
    )

    # Ordering, direction and the lossless path only need dimensions, so
    # full-resolution decoding is deferred until pixels are actually needed
    # (seam matching, a --fit resize, or the Pillow re-encode). Header
    # dimensions equal decoded ones only when no EXIF rotation applies.
    decoded = use_edge_match or pixels_differ_from_file
    images = _decode_many(raws) if decoded else heads

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
        pre_fit_sizes = [img.size for img in images]
        if direction == "horizontal":
            needs_resize = len({img.height for img in images}) > 1
        else:
            needs_resize = len({img.width for img in images}) > 1
        if needs_resize and not decoded:
            images = _decode_many([raw_of[p] for p in image_paths])
            decoded = True
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
        if [img.size for img in images] != pre_fit_sizes:
            pixels_differ_from_file = True

    # Detect encoding params from the first input image
    first = image_paths[0]
    qtables = quality = subsampling = None
    is_grayscale = False
    if input_formats[0] == "JPEG":
        qtables, quality, subsampling, is_grayscale = _jpeg_params_from_bytes(raw_of[first])
    else:
        qtables, quality, subsampling = None, 85, 2

    all_subsampling = None
    if len(image_paths) > 1 and input_formats[0] == "JPEG":
        all_subsampling = [subsampling]
        for i in range(1, len(image_paths)):
            if input_formats[i] == "JPEG":
                all_subsampling.append(_parse_sof_subsampling(raw_of[image_paths[i]]))
            else:
                all_subsampling.append(2)

    if overlap > 0:
        min_join_dim = (min(img.width for img in images) if direction == "horizontal"
                        else min(img.height for img in images))
        if overlap >= min_join_dim:
            overlap = max(0, min_join_dim - 1)
            print(f"Warning: overlap clamped to {overlap}px")

    if overlap > 0:
        print("Note: overlap disables the lossless jpegtran fast-path; re-encoding via Pillow.")
    lossless = overlap == 0 and not pixels_differ_from_file and _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling, all_subsampling=all_subsampling)

    # ── Pillow re-encode fallback ─────────────────────────────────────────────
    if not lossless:
        if not decoded:
            images = _decode_many([raw_of[p] for p in image_paths])
        if direction == "horizontal":
            total_w = sum(img.width  for img in images) - overlap * (len(images) - 1)
            total_h = max(img.height for img in images)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            x = 0
            for i, img in enumerate(images):
                if i > 0 and overlap > 0:
                    x -= overlap
                _paste_overlap(canvas, img, (x, 0), overlap if i > 0 else 0, overlap_blend, "horizontal")
                x += img.width
        else:
            total_w = max(img.width  for img in images)
            total_h = sum(img.height for img in images) - overlap * (len(images) - 1)
            canvas  = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            y = 0
            for i, img in enumerate(images):
                if i > 0 and overlap > 0:
                    y -= overlap
                _paste_overlap(canvas, img, (0, y), overlap if i > 0 else 0, overlap_blend, "vertical")
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
  flex-wrap: wrap;
  gap: 8px 14px;
  padding: 8px 24px;
  min-height: 52px;
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

.header-left {
  display: flex;
  align-items: baseline;
  gap: 14px;
  min-width: 0;
}

.capsule {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 14px;
  padding: 6px 12px;
  background: #fafafc;
  border: 1px solid #e0e0e0;
  border-radius: 11px;
}

#overlap-control {
  padding-left: 14px;
  border-left: 1px solid #e0e0e0;
}

.header-right {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px 14px;
  margin-left: auto;
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

.fit-toggle-input:focus-visible + .fit-toggle-switch {
  outline: 2px solid #0071e3;
  outline-offset: 2px;
}

.ghost-btn, #quit-btn, #download-btn {
  height: 36px;
  padding: 0 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 14px;
  letter-spacing: -0.224px;
  line-height: 1;
  flex-shrink: 0;
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
  align-items: safe center;
  justify-content: safe center;
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
  min-height: 64px;
  min-width: 0;
  position: relative;
}

#preview-img, #preview-live {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 8px;
  background: #ffffff;
}

#preview-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  align-self: stretch;
  flex-shrink: 0;
}

#preview-caption {
  font-size: 14px;
  letter-spacing: -0.224px;
  color: #7a7a7a;
  min-width: 0;
}

#preview-pane[data-state="final"] #preview-caption { color: #1d1d1f; }
#preview-pane[data-state="pending"] #preview-img,
#preview-pane[data-state="pending"] #preview-live { opacity: 0.35; transition: opacity 0.15s; }

#status.error {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #1d1d1f;
  font-weight: 600;
}
#status.error::before {
  content: '!';
  flex-shrink: 0;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #1d1d1f;
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  line-height: 16px;
  text-align: center;
}
body.loading, body.loading * { cursor: progress !important; }

.ghost-btn {
  background: transparent;
  color: #0066cc;
  border: 1px solid #0066cc;
  border-radius: 9999px;
  font-family: inherit;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.12s, transform 0.08s;
}
.ghost-btn:hover { background: rgba(0,102,204,0.06); }
.ghost-btn:active { transform: scale(0.95); }
.ghost-btn:focus-visible { outline: 2px solid #0071e3; outline-offset: 2px; }

#download-btn {
  background: #0066cc;
  color: #fff;
  border: none;
  border-radius: 9999px;
  font-family: inherit;
  font-weight: 400;
  cursor: pointer;
  transition: opacity 0.12s, transform 0.08s;
  white-space: nowrap;
}

#download-btn:hover:not(:disabled) { opacity: 0.85; }
#download-btn:disabled { cursor: progress; }
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
  place-items: center;
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
  cursor: pointer;
  transition: border-color 0.1s, color 0.1s, background 0.1s;
}

.cell.empty:hover {
  border-color: #0066cc;
  color: #0066cc;
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
  width: 24px;
  height: 24px;
  background: rgba(0,0,0,0.55);
  border: none;
  border-radius: 8px;
  color: rgba(255,255,255,0.85);
  font-size: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.12s, background 0.12s, color 0.12s;
  z-index: 4;
}

.cell:hover .btn-remove,
.cell:focus-within .btn-remove { opacity: 1; }
.cell:focus-visible { outline: 2px solid #0071e3; outline-offset: 2px; }
.btn-remove:hover { background: rgba(0,0,0,0.85); color: #fff; }

.cell-toolbar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 32px;
  background: rgba(0,0,0,0.65);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.12s;
  z-index: 4;
  border-radius: 0 0 7px 7px;
}

.cell:hover .cell-toolbar,
.cell:focus-within .cell-toolbar { opacity: 1; }

#toast {
  position: fixed;
  left: 50%;
  bottom: 76px;
  transform: translateX(-50%) translateY(12px);
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 8px 8px 18px;
  background: rgba(245,245,247,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid #e0e0e0;
  border-radius: 9999px;
  font-size: 14px;
  letter-spacing: -0.224px;
  color: #1d1d1f;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 0.15s, transform 0.15s, visibility 0s 0.15s;
  z-index: 150;
}
#toast.visible {
  opacity: 1;
  visibility: visible;
  transform: translateX(-50%);
  pointer-events: auto;
  transition: opacity 0.15s, transform 0.15s;
}
#toast kbd { font-family: inherit; color: #7a7a7a; }

.btn-transform {
  width: 32px;
  height: 28px;
  background: transparent;
  border: none;
  color: rgba(255,255,255,0.85);
  font-size: 15px;
  cursor: pointer;
  border-radius: 8px;
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
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  color: #1d1d1f;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s, transform 0.08s;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.overlap-step-btn:hover { border-color: #0066cc; color: #0066cc; }

/* Small controls keep their look but get a ~44px hit area. */
.overlap-step-btn, #overlap-auto-btn, .zoom-level-btn, .add-btn { position: relative; }
.overlap-step-btn::before, #overlap-auto-btn::before, .zoom-level-btn::before {
  content: ''; position: absolute; inset: -11px -3px;
}
.btn-remove::before { content: ''; position: absolute; inset: -10px; }
#add-left-btn::before, #add-right-btn::before { content: ''; position: absolute; inset: 0 -11px; }
#add-top-btn::before, #add-bottom-btn::before { content: ''; position: absolute; inset: -11px 0; }
.overlap-step-btn:active { transform: scale(0.95); }
.overlap-step-btn.active { border-color: #0066cc; color: #0066cc; background: rgba(0,102,204,0.06); }
.overlap-step-btn:focus-visible, #overlap-auto-btn:focus-visible, .zoom-level-btn:focus-visible {
  outline: 2px solid #0071e3; outline-offset: 2px;
}
.zoom-level-btn {
  font-family: inherit;
  transition: border-color 0.12s, color 0.12s, transform 0.08s;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: transparent;
  color: #1d1d1f;
  cursor: pointer;
}
.zoom-level-btn.active { background: #0066cc; color: #fff; border-color: #0066cc; }

#overlap-auto-btn, .zoom-level-btn, .overlap-step-btn.text-btn {
  width: auto;
  height: 22px;
  padding: 0 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  line-height: 1;
  flex-shrink: 0;
}
#overlap-auto-btn:active, .zoom-level-btn:active { transform: scale(0.95); }
#zoom-panel { align-self: stretch; flex-shrink: 0; }
#zoom-canvas {
  width: 100%;
  display: block;
  height: clamp(80px, 18vh, 200px);
  image-rendering: pixelated;
  border-radius: 8px;
  background: #1d1d1f;
  cursor: grab;
}
.zoom-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  font-size: 12px;
  color: #7a7a7a;
}
.zoom-hint { margin-left: 8px; }

#overlap-auto-btn {
  font-family: inherit;
  font-weight: 400;
  background: transparent;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  color: #1d1d1f;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s, transform 0.08s;
}
#overlap-auto-btn:hover { border-color: #0066cc; color: #0066cc; }
#overlap-auto-btn.active { border-color: #0066cc; color: #0066cc; background: rgba(0,102,204,0.06); }

.add-btn {
  background: transparent;
  border: 1px dashed #e0e0e0;
  border-radius: 8px;
  color: #7a7a7a;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s, transform 0.08s;
  flex-shrink: 0;
}
.add-btn:active { transform: scale(0.95); }
.add-btn:focus-visible { outline: 2px solid #0071e3; outline-offset: 2px; }

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

#quit-btn {
  background: transparent;
  color: #1d1d1f;
  border: 1px solid #e0e0e0;
  border-radius: 9999px;
  font-family: inherit;
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s, background 0.12s, transform 0.08s;
  white-space: nowrap;
}
#quit-btn:hover:not(:disabled) { border-color: #1d1d1f; }
#quit-btn.armed { background: #1d1d1f; border-color: #1d1d1f; color: #fff; }
#quit-btn:active:not(:disabled) { transform: scale(0.95); }
#quit-btn:disabled { color: #7a7a7a; cursor: default; }
#quit-btn:focus-visible {
  outline: 2px solid #0071e3;
  outline-offset: 2px;
}

#stopped {
  display: none;
  position: fixed;
  inset: 0;
  background: #f5f5f7;
  z-index: 200;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  color: #1d1d1f;
  text-align: center;
  padding: 24px;
}
#stopped.visible { display: flex; }
#stopped strong { font-size: 21px; font-weight: 600; letter-spacing: -0.4px; }
#stopped span { font-size: 15px; color: #7a7a7a; }

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
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7a7a7a; }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>Image Stitcher</h1>
    <span id="status" aria-live="polite">Drop images to start</span>
  </div>
  <div class="header-right">
    <div id="options" class="capsule" style="display:none">
    <label class="fit-toggle-label" title="Scale images to the same height (row) or width (column)">
      <input type="checkbox" id="fit-toggle" class="fit-toggle-input">
      <span class="fit-toggle-switch"></span>
      <span>Fit images to layout</span>
    </label>
    <div id="overlap-control" style="display:none">
      <label for="overlap-slider">Overlap</label>
      <input type="range" id="overlap-slider" min="0" max="800" value="0" step="1">
      <button id="overlap-dec-btn" class="overlap-step-btn" title="Decrease by 1px" aria-label="Decrease overlap by 1 pixel">−</button>
      <span id="overlap-val">0 px</span>
      <button id="overlap-inc-btn" class="overlap-step-btn" title="Increase by 1px" aria-label="Increase overlap by 1 pixel">+</button>
      <button id="zoom-toggle-btn" class="overlap-step-btn text-btn" title="Zoom into the seam edge" aria-pressed="false">Zoom</button>
      <button id="overlap-auto-btn" title="Detect the best overlap for these images">Auto</button>
      <label class="fit-toggle-label" title="Cross-fade blend instead of hard paste">
        <input type="checkbox" id="overlap-fade-toggle" class="fit-toggle-input">
        <span class="fit-toggle-switch"></span>
        <span>Fade</span>
      </label>
    </div>
    </div>
    <button id="add-images-btn" class="ghost-btn" style="display:none" title="Add more images to the end">Add Images</button>
    <button id="quit-btn" title="Stop the Image Stitcher server (click twice)">Quit</button>
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
      <canvas id="preview-live" aria-label="Live preview" style="display:none"></canvas>
    </div>
    <div id="zoom-panel" style="display:none">
      <canvas id="zoom-canvas" aria-label="Magnified view of the seam"></canvas>
      <div class="zoom-bar">
        <span>Magnify</span>
        <button class="zoom-level-btn" data-zoom="2">2×</button>
        <button class="zoom-level-btn active" data-zoom="3">3×</button>
        <button class="zoom-level-btn" data-zoom="4">4×</button>
        <span class="zoom-hint">Drag to pan · blue lines mark the overlap</span>
      </div>
    </div>
    <div id="preview-footer">
      <span id="preview-caption"></span>
      <button id="download-btn" type="button" title="Save the full-quality JPEG (⌘↵)">Download</button>
    </div>
  </div>
</div>

<div id="toast" role="status" aria-live="polite">
  <span id="toast-msg"></span>
  <button id="toast-undo" class="ghost-btn">Undo <kbd>⌘Z</kbd></button>
</div>

<div id="stopped" role="status" aria-live="polite">
  <strong>Image Stitcher has stopped</strong>
  <span>You can close this tab. Run <code>jpegconcat</code> to start it again.</span>
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
let _cellDragoverRafPending = false;
let _pendingCellDragover = null;

const appBody       = document.getElementById('app-body');
const dropZoneEl    = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const gridOuter     = document.getElementById('grid-outer');
const gridEl        = document.getElementById('grid');
const addLeftBtn    = document.getElementById('add-left-btn');
const addRightBtn   = document.getElementById('add-right-btn');
const addTopBtn     = document.getElementById('add-top-btn');
const addBottomBtn  = document.getElementById('add-bottom-btn');
const addImagesBtn  = document.getElementById('add-images-btn');
const optionsEl     = document.getElementById('options');
const statusEl      = document.getElementById('status');
const processingEl  = document.getElementById('processing');
const previewImg    = document.getElementById('preview-img');
const previewLive   = document.getElementById('preview-live');
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
  overlapValEl.textContent = overlapSlider.value + ' px';
  onLayoutChanged();
});

// Auto is a one-shot detection, not a mode: the server suggests a value,
// which then lands on the slider like any manual setting. (A sticky "auto"
// would re-detect on Download and could save something the preview never
// showed.)
overlapAutoBtn.addEventListener('click', async () => {
  if (imgCount() < 2) return;
  _statusError = null;
  overlapAutoBtn.disabled = true;
  overlapAutoBtn.textContent = 'Detecting…';
  try {
    const fd = _buildFormData();
    fd.set('overlap', 'auto');
    const res = await fetch('/preview', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
    const px = parseInt(res.headers.get('X-Overlap-Px'), 10);
    if (!Number.isNaN(px)) _setOverlapPx(px);
    onLayoutChanged();
  } catch (err) {
    showError('Overlap detection failed: ' + err.message);
  } finally {
    overlapAutoBtn.disabled = false;
    overlapAutoBtn.textContent = 'Auto';
  }
});

document.getElementById('fit-toggle').addEventListener('change', () => onLayoutChanged());

function _stepOverlap(delta) {
  const v = Math.min(parseInt(overlapSlider.max), Math.max(0, parseInt(overlapSlider.value) + delta));
  _setOverlapPx(v);
  onLayoutChanged();
}
overlapDecBtn.addEventListener('click', () => _stepOverlap(-1));
overlapIncBtn.addEventListener('click', () => _stepOverlap( 1));

overlapFadeToggle.addEventListener('change', () => onLayoutChanged());

zoomToggleBtn.addEventListener('click', () => {
  _zoomActive  = !_zoomActive;
  _zoomOffsetX = null;
  _zoomOffsetY = null;
  zoomToggleBtn.classList.toggle('active', _zoomActive);
  zoomToggleBtn.setAttribute('aria-pressed', _zoomActive);
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
let _zoomMoveRafPending = false;
let _lastZoomMoveX = 0, _lastZoomMoveY = 0;
zoomCanvas.addEventListener('mousedown', e => {
  _zoomDragX = e.clientX;
  _zoomDragY = e.clientY;
  zoomCanvas.style.cursor = 'grabbing';
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (_zoomDragX === null) return;
  _lastZoomMoveX = e.clientX;
  _lastZoomMoveY = e.clientY;
  if (_zoomMoveRafPending) return;
  _zoomMoveRafPending = true;
  requestAnimationFrame(() => {
    _zoomMoveRafPending = false;
    if (_zoomDragX === null) return;
    const dx = _lastZoomMoveX - _zoomDragX;
    const dy = _lastZoomMoveY - _zoomDragY;
    _zoomDragX = _lastZoomMoveX;
    _zoomDragY = _lastZoomMoveY;
    if (_zoomOffsetX !== null) {
      _zoomOffsetX = Math.round(_zoomOffsetX - dx / _zoomLevel);
      _zoomOffsetY = Math.round(_zoomOffsetY - dy / _zoomLevel);
    }
    if (_lastZoomParams) _renderZoomStrip(_lastZoomParams);
  });
});
window.addEventListener('mouseup', () => {
  if (_zoomDragX !== null) { _zoomDragX = null; _zoomDragY = null; zoomCanvas.style.cursor = 'grab'; }
});

const uid = () => Math.random().toString(36).slice(2, 9);

// Thumbnails are blob: URLs rather than base64 data: URLs — toBlob encodes
// off the main thread and the DOM doesn't carry ~100 KB attribute strings.
// Each cell owns its URL; _disposeCell revokes it.
async function _thumbFromSource(src, maxPx) {
  const scale = Math.min(1, maxPx / Math.max(src.width, src.height));
  const w = Math.max(1, Math.round(src.width * scale));
  const h = Math.max(1, Math.round(src.height * scale));
  const cv = Object.assign(document.createElement('canvas'), { width: w, height: h });
  cv.getContext('2d').drawImage(src, 0, 0, w, h);
  const blob = await new Promise(res => cv.toBlob(res, 'image/jpeg', 0.75));
  return URL.createObjectURL(blob);
}


// Releases everything a cell owns (thumbnail URL, cached preview bitmap).
// Call whenever a cell object is dropped from the grid or replaced.
function _disposeCell(cell) {
  if (!cell) return;
  _evictBitmap(cell.id);
  if (cell.dataURL && cell.dataURL.startsWith('blob:')) URL.revokeObjectURL(cell.dataURL);
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

// One level of undo for Remove. The removed cell isn't disposed until
// the undo window closes, so restoring it needs no re-decode.
let _undo = null;   // { grid, cell, timer }
const UNDO_MS = 8000;
const toastEl  = document.getElementById('toast');
const toastMsg = document.getElementById('toast-msg');

function _finalizeUndo() {
  if (!_undo) return;
  clearTimeout(_undo.timer);
  _disposeCell(_undo.cell);
  _undo = null;
  toastEl.classList.remove('visible');
}

function removeCell(r, c) {
  _finalizeUndo();
  const cell = grid[r][c];
  if (!cell) return;
  _undo = { grid: grid.map(row => row.slice()), cell, timer: setTimeout(_finalizeUndo, UNDO_MS) };
  toastMsg.textContent = `Removed ${cell.name}`;
  toastEl.classList.add('visible');
  grid[r][c] = null;
  compact();
}

function undoRemove() {
  if (!_undo) return;
  clearTimeout(_undo.timer);
  grid = _undo.grid;
  _undo = null;
  toastEl.classList.remove('visible');
  render();
}
document.getElementById('toast-undo').addEventListener('click', undoRemove);

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

// Returns true when the grid's row/col shape is unchanged (a plain swap),
// so the caller can patch just the two affected cells instead of a full
// render(); returns false when rows/cols were inserted or removed and a
// full render() is required.
function moveCellToZone(srcR, srcC, dstR, dstC, zone) {
  if (zone === 'center') { swapCells(srcR, srcC, dstR, dstC); return true; }
  const srcCell = grid[srcR][srcC];
  const dstCell = grid[dstR][dstC];
  grid[srcR][srcC] = null;
  compact();
  // Re-locate dstCell after compact may have shifted indices
  let nr = -1, nc = -1;
  outer: for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < grid[r].length; c++)
      if (grid[r][c] === dstCell) { nr = r; nc = c; break outer; }
  if (nr === -1) { grid.push([srcCell]); return false; }
  if (zone === 'left')   { insertCol(nc - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'right')  { insertCol(nc);     grid[nr][nc + 1] = srcCell; }
  else if (zone === 'top')    { insertRow(nr - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'bottom') { insertRow(nr);     grid[nr + 1][nc] = srcCell; }
  return false;
}

async function applyTransform(r, c, type) {
  _finalizeUndo();  // undo snapshots hold cell objects this would replace
  const cell = grid[r][c];
  if (!cell) return;
  const img = await createImageBitmap(cell.file, { imageOrientation: 'from-image' });
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
  img.close();
  // The thumbnail comes straight from the transformed canvas rather than
  // re-decoding the freshly encoded JPEG.
  const [blob, thumbURL] = await Promise.all([
    new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92)),
    _thumbFromSource(canvas, 400),
  ]);
  const newFile = new File([blob], cell.name, { type: 'image/jpeg' });
  // The grid may have changed while we were encoding; only patch if this
  // cell is still where it was.
  if (grid[r] && grid[r][c] === cell) {
    _disposeCell(cell);
    grid[r][c] = { ...cell, dataURL: thumbURL, file: newFile };
  } else {
    URL.revokeObjectURL(thumbURL);
    return;
  }
  updateCellInPlace(r, c);
  onLayoutChanged();
}

// Patches a single cell's thumbnail <img> in place instead of tearing down
// and rebuilding the whole grid (which forces the browser to re-layout and
// re-decode every other cell's thumbnail too). Only valid when the cell's
// position and the grid's shape haven't changed — falls back to a full
// render() if the expected element isn't found.
function updateCellInPlace(r, c) {
  const cell = grid[r][c];
  if (!cell) { render(); return; }
  const img = gridEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"] img`);
  if (!img) { render(); return; }
  img.src = cell.dataURL;
  img.alt = cell.name;
}

// One full-resolution decode per added image yields both the grid
// thumbnail and the (pre-warmed) live-preview bitmap, so the first slider
// drag never stalls on a decode. Costs some memory up front, in exchange
// for smoothness.
// Decodes the image files into cells, with a loading message while it
// runs. Unreadable files are skipped and reported rather than failing the
// whole batch.
async function makeCells(files) {
  _finalizeUndo();  // undo would silently drop the images being added
  _statusError = null;
  const all = [...files];
  const imgs = all.filter(f => f.type.startsWith('image/'));
  if (!imgs.length) {
    if (all.length) showError('Those files aren’t images. Use JPEG, PNG or WebP.');
    return [];
  }
  statusEl.classList.remove('error');
  statusEl.textContent = `Loading ${imgs.length} image${imgs.length !== 1 ? 's' : ''}…`;
  document.body.classList.add('loading');
  try {
    const results = await Promise.allSettled(imgs.map(_makeCell));
    const bad = imgs.filter((_, i) => results[i].status === 'rejected').map(f => f.name);
    const skipped = all.length - imgs.length;
    if (bad.length) showError(`Couldn’t read ${bad.join(', ')}`);
    else if (skipped) showError(`Skipped ${skipped} file${skipped !== 1 ? 's' : ''} that aren’t images`);
    else updateStatus();
    return results.filter(r => r.status === 'fulfilled').map(r => r.value);
  } finally {
    document.body.classList.remove('loading');
  }
}

async function _makeCell(f) {
  const full = await createImageBitmap(f, { imageOrientation: 'from-image' });
  const cell = { id: uid(), name: f.name, file: f };
  let entry;
  try {
    // Both start their synchronous draws from `full` before it is closed.
    const thumbP = _thumbFromSource(full, 400);
    entry = _previewEntryFrom(full);
    cell.dataURL = await thumbP;
  } catch (err) {
    full.close();
    if (entry) entry.then(e => e.bmp.close(), () => {});
    throw err;
  }
  _bitmapCache[cell.id] = entry;
  entry.catch(() => { if (_bitmapCache[cell.id] === entry) delete _bitmapCache[cell.id]; });
  return cell;
}

// Adds images after the existing ones: down a single column, otherwise
// along the first row.
async function appendFiles(files) {
  if (!hasImages()) return initFromFiles(files);
  const cells = await makeCells(files);
  if (!cells.length) return;
  if (nCols() === 1 && nRows() > 1) for (const c of cells) grid.push([c]);
  else for (const c of cells) { insertCol(nCols() - 1); grid[0][nCols() - 1] = c; }
  render();
}

async function initFromFiles(files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  _finalizeUndo();  // the old grid is going away; nothing to undo into
  for (const row of grid) for (const old of row) _disposeCell(old);
  grid = [cells];
  render();
}

async function dropFilesOnCell(r, c, zone, files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  const first = cells[0];
  if (zone === 'center') {
    _disposeCell(grid[r][c]);
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
    overlapCtrl.style.display = 'none';
    addImagesBtn.style.display = 'none';
    optionsEl.style.display = 'none';
    onLayoutChanged();
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
  addImagesBtn.style.display = '';
  optionsEl.style.display = n >= 2 ? 'flex' : 'none';
  // Overlap only applies to single-row or single-column grids — _stitch_grid
  // silently zeroes it for any other shape, so don't show a control that
  // would have no effect.
  const overlapApplies = rows === 1 || cols === 1;
  overlapCtrl.style.display = (n >= 2 && overlapApplies) ? 'flex' : 'none';
  // The seam zoom only draws for side-by-side layouts.
  const zoomApplies = rows === 1;
  zoomToggleBtn.style.display = zoomApplies ? '' : 'none';
  if (!zoomApplies && _zoomActive) {
    _zoomActive = false;
    zoomToggleBtn.classList.remove('active');
    zoomToggleBtn.setAttribute('aria-pressed', 'false');
    zoomPanel.style.display = 'none';
  }
  onLayoutChanged();
}

// One status line, rebuilt from current state rather than overwritten by
// whichever code path ran last.
let _statusError = null;

// Errors stay in the status line (in bold) until the next action starts.
function showError(msg) {
  _statusError = msg;
  updateStatus();
}

function updateStatus() {
  statusEl.classList.toggle('error', !!_statusError);
  if (_statusError) { statusEl.textContent = _statusError; return; }
  if (!hasImages()) { statusEl.textContent = 'Drop images to start'; return; }
  const n = imgCount();
  let s = `${n} image${n !== 1 ? 's' : ''} · ${nRows()}×${nCols()} grid`;
  if (overlapCtrl.style.display !== 'none') {
    s += ` · ${overlapSlider.value}px overlap`;
  }
  statusEl.textContent = s;
}

const previewPane    = document.getElementById('preview-pane');
const previewCaption = document.getElementById('preview-caption');

// 'live'  — fast client-side composite (downscaled, not the final file)
// 'final' — the server's real output for the current layout
// 'pending' — an old result is shown while the new one renders
function _setPreviewState(state, text) {
  previewPane.dataset.state = state;
  previewCaption.textContent = text || {
    live:    'Quick preview (reduced size) · Download saves full quality',
    pending: 'Rendering…',
  }[state] || '';
}

const _liveSupported = () => (nRows() === 1 || nCols() === 1) && imgCount() >= 2;

// Called after anything that changes what the output would be: layout,
// rotation/flip, Fit, overlap. Keeps the preview pane from showing a
// result that Download would no longer produce.
function onLayoutChanged() {
  updateStatus();
  if (imgCount() < 2) {
    _previewGen++;  // drop any in-flight live paint
    appBody.classList.remove('has-preview', 'layout-side', 'layout-stack');
    return;
  }
  if (_liveSupported()) scheduleLivePreview();
  else if (appBody.classList.contains('has-preview')) {
    // Back to the layout that was last stitched: the shown result is current.
    if (_matchesLastServerResult(_requestSignature())) {
      _setPreviewState('final', _lastServerResult.caption || ' ');
      return;
    }
    _setPreviewState('pending');
  }
  // Grids with no client-side live preview get the server render instead,
  // debounced so a burst of edits costs one request.
  if (!_liveSupported()) scheduleServerPreview();
}

function buildCell(cell, r, c) {
  const el = document.createElement('div');
  el.className = 'cell';
  el.draggable = true;
  el.tabIndex = 0;
  el.setAttribute('aria-label',
    `${cell.name}, row ${r + 1}, column ${c + 1}. R to rotate, H or V to flip, Delete to remove.`);
  el.addEventListener('keydown', e => {
    if (e.target !== el || e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.toLowerCase();
    const transform = key === 'r' ? (e.shiftKey ? 'rotate-ccw' : 'rotate-cw')
                    : key === 'h' ? 'flip-h' : key === 'v' ? 'flip-v' : null;
    if (transform) { e.preventDefault(); applyTransform(r, c, transform); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      removeCell(r, c);
      render();
      // Keep keyboard focus in the grid.
      const next = gridEl.querySelector('.cell:not(.empty)');
      if (next) next.focus();
    }
  });
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
  rmBtn.title = 'Remove (Delete)';
  rmBtn.setAttribute('aria-label', 'Remove');
  rmBtn.tabIndex = -1;
  rmBtn.textContent = '×';
  rmBtn.addEventListener('click', e => { e.stopPropagation(); removeCell(r, c); render(); });
  el.appendChild(rmBtn);

  const toolbar = document.createElement('div');
  toolbar.className = 'cell-toolbar';
  toolbar.innerHTML =
    '<button class="btn-transform" tabindex="-1" title="Rotate counter-clockwise (⇧R)" aria-label="Rotate counter-clockwise">↺</button>' +
    '<button class="btn-transform" tabindex="-1" title="Rotate clockwise (R)" aria-label="Rotate clockwise">↻</button>' +
    '<button class="btn-transform" tabindex="-1" title="Flip horizontal (H)" aria-label="Flip horizontal">↔</button>' +
    '<button class="btn-transform" tabindex="-1" title="Flip vertical (V)" aria-label="Flip vertical">↕</button>';
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
    if (!dragSrc) return;
    _pendingCellDragover = { overlay, clientX: e.clientX, clientY: e.clientY, el };
    if (_cellDragoverRafPending) return;
    _cellDragoverRafPending = true;
    requestAnimationFrame(() => {
      _cellDragoverRafPending = false;
      if (!_pendingCellDragover) return;
      const p = _pendingCellDragover;
      _pendingCellDragover = null;
      activateZone(p.overlay, zoneAt(p.clientX, p.clientY, p.el));
    });
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
    if (sr !== r || sc !== c) {
      _finalizeUndo();
      const sameShape = moveCellToZone(sr, sc, r, c, zoneAt(e.clientX, e.clientY, el));
      if (sameShape) { updateCellInPlace(sr, sc); updateCellInPlace(r, c); onLayoutChanged(); }
      else render();
    }
  });

  return el;
}

function buildEmptyCell(r, c) {
  const el = document.createElement('div');
  el.className = 'cell empty';
  el.dataset.r = r;
  el.dataset.c = c;
  el.textContent = '+';
  el.tabIndex = 0;
  el.setAttribute('role', 'button');
  el.setAttribute('aria-label', `Add image at row ${r + 1}, column ${c + 1}`);
  el.addEventListener('dragover', e => { e.preventDefault(); });
  el.addEventListener('click', () => openPicker(r, c));
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker(r, c); }
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

// Opens the file picker aimed at an empty cell, or (no target) at the end
// of the grid. The target is always reset, so cancelling a cell's picker
// can't redirect a later browse into that cell.
function openPicker(r, c) {
  if (r === undefined) { delete fileInput.dataset.targetR; delete fileInput.dataset.targetC; }
  else { fileInput.dataset.targetR = r; fileInput.dataset.targetC = c; }
  fileInput.click();
}

dropZoneEl.addEventListener('click', () => openPicker());
dropZoneEl.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker(); } });
document.getElementById('add-images-btn').addEventListener('click', () => openPicker());
fileInput.addEventListener('change', e => {
  if (!e.target.files.length) { fileInput.value = ''; return; }
  const tr = fileInput.dataset.targetR;
  const tc = fileInput.dataset.targetC;
  if (tr !== undefined && tc !== undefined) {
    dropFilesOnCell(+tr, +tc, 'center', e.target.files);
    delete fileInput.dataset.targetR;
    delete fileInput.dataset.targetC;
  } else {
    appendFiles(e.target.files);
  }
  fileInput.value = '';
});

addLeftBtn.addEventListener('click', () => { if (hasImages()) { _finalizeUndo(); insertCol(-1); render(); } });
addRightBtn.addEventListener('click', () => { if (hasImages()) { _finalizeUndo(); insertCol(nCols() - 1); render(); } });
addTopBtn.addEventListener('click', () => { if (hasImages()) { _finalizeUndo(); insertRow(-1); render(); } });
addBottomBtn.addEventListener('click', () => { if (hasImages()) { _finalizeUndo(); insertRow(nRows() - 1); render(); } });

function _buildFormData() {
  const formData = new FormData();
  const layoutIds = grid.map(row => row.map(cell => cell ? cell.id : null));
  formData.append('layout', JSON.stringify(layoutIds));
  formData.append('fit', document.getElementById('fit-toggle').checked);
  formData.append('overlap', overlapSlider.value);
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

// The last full-resolution result from the server, keyed on everything that
// feeds _buildFormData. Download reuses it instead of re-uploading every
// image and re-stitching an identical JPEG. (Never the client-side live
// composite — that one is downscaled.)
let _lastServerResult = null;  // { key, files, blob }

function _requestSignature() {
  const files = [];
  const layoutIds = grid.map(row => row.map(cell => {
    if (cell) files.push(cell.file);
    return cell ? cell.id : null;
  }));
  const key = JSON.stringify([
    layoutIds,
    document.getElementById('fit-toggle').checked,
    overlapSlider.value,
    overlapFadeToggle.checked,
  ]);
  return { key, files };
}

function _matchesLastServerResult(sig) {
  const last = _lastServerResult;
  return !!last && last.key === sig.key && last.files.length === sig.files.length
    && last.files.every((f, i) => f === sig.files[i]);
}

// ── Live client-side preview ──────────────────────────────────────────────────
const _bitmapCache = {};  // cell.id → ImageBitmap

function _renderZoomStrip({ cv, dw0, ov, cvW, cvH }) {
  const ZOOM   = _zoomLevel;
  const zc     = zoomCanvas;
  // Measure after showing: a hidden panel has no size. The CSS box sets
  // the size; the backing store just matches it so pixels stay square.
  zoomPanel.style.display = 'block';
  zc.width     = Math.max(1, zc.clientWidth);
  zc.height    = Math.max(1, zc.clientHeight);

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
  // Accent line on a white under-stroke so it reads on dark and light photos.
  for (const natX of [seamL, dw0]) {
    const mx = (natX - srcX) * ZOOM;
    if (mx >= 0 && mx <= zc.width) {
      for (const [style, w] of [['#ffffff', 4], ['#0066cc', 2]]) {
        zctx.strokeStyle = style; zctx.lineWidth = w;
        zctx.beginPath(); zctx.moveTo(mx, 0); zctx.lineTo(mx, zc.height); zctx.stroke();
      }
    }
  }
  zoomPanel.style.display = 'block';
}

// The live composite is capped at LIVE_MAX_PX on its longest side, so no
// single image is ever drawn larger than that. Caching a bitmap pre-shrunk
// to that bound (plus the natural size, which the layout math needs) means
// each rAF tick resamples a few megapixels instead of full camera frames.
const LIVE_MAX_PX = 2000;

// Turns a full-resolution bitmap into a preview-cache entry, taking
// ownership of `full` (closed here unless it is small enough to keep).
async function _previewEntryFrom(full) {
  const w = full.width, h = full.height;
  const s = LIVE_MAX_PX / Math.max(w, h);
  if (s >= 1) return { bmp: full, w, h };
  // Downscale through a canvas: GPU-backed and far cheaper than
  // createImageBitmap's resizeQuality option, which resamples on the CPU.
  try {
    const cv = Object.assign(document.createElement('canvas'), {
      width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)),
    });
    const c = cv.getContext('2d');
    c.imageSmoothingQuality = 'high';
    c.drawImage(full, 0, 0, cv.width, cv.height);
    return { bmp: await createImageBitmap(cv), w, h };
  } finally { full.close(); }
}

async function _loadPreviewBitmap(file) {
  return _previewEntryFrom(await createImageBitmap(file, { imageOrientation: 'from-image' }));
}

function _getBitmap(cell) {
  // Cache the promise so concurrent callers share one decode.
  if (!_bitmapCache[cell.id]) {
    const p = _loadPreviewBitmap(cell.file);
    _bitmapCache[cell.id] = p;
    p.catch(() => { if (_bitmapCache[cell.id] === p) delete _bitmapCache[cell.id]; });
  }
  return _bitmapCache[cell.id];
}

function _evictBitmap(id) {
  const p = _bitmapCache[id];
  if (!p) return;
  delete _bitmapCache[id];
  p.then(e => e.bmp.close(), () => {});
}

// Reused across frames instead of allocating fresh canvases every tick.
// The live composite is drawn straight into the on-screen canvas, so slider
// feedback lands on the next frame with no JPEG encode/decode round trip.
const _liveCanvas = previewLive;
const _fadeCanvas = document.createElement('canvas');

// Draws one image whose leading `ov` px (along `horizontal` or vertical)
// cross-fade over what is already on ctx. Only the overlap band goes
// through the scratch canvas; the rest is a plain drawImage.
function _drawFaded(ctx, e, x, y, dw, dh, ov, horizontal) {
  const bmp = e.bmp;
  const fx = bmp.width / dw, fy = bmp.height / dh;   // dest px → bitmap px
  const bw = horizontal ? ov : dw, bh = horizontal ? dh : ov;
  if (_fadeCanvas.width !== bw || _fadeCanvas.height !== bh) {
    _fadeCanvas.width = bw; _fadeCanvas.height = bh;
  }
  const tc = _fadeCanvas.getContext('2d');
  tc.globalCompositeOperation = 'source-over';
  tc.clearRect(0, 0, bw, bh);
  tc.drawImage(bmp, 0, 0, bw * fx, bh * fy, 0, 0, bw, bh);
  const grad = horizontal ? tc.createLinearGradient(0, 0, ov, 0) : tc.createLinearGradient(0, 0, 0, ov);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,1)');
  tc.globalCompositeOperation = 'destination-in';
  tc.fillStyle = grad;
  tc.fillRect(0, 0, bw, bh);
  ctx.drawImage(_fadeCanvas, x, y);
  if (horizontal && dw > ov) {
    ctx.drawImage(bmp, ov * fx, 0, bmp.width - ov * fx, bmp.height, x + ov, y, dw - ov, dh);
  } else if (!horizontal && dh > ov) {
    ctx.drawImage(bmp, 0, ov * fy, bmp.width, bmp.height - ov * fy, x, y + ov, dw, dh - ov);
  }
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

  const gen = _previewGen;
  let entries;
  try { entries = await Promise.all(cells.map(_getBitmap)); }
  catch (_) { return; }
  // A server result (or another layout) replaced the preview while we were
  // waiting on bitmaps — don't paint a stale low-res composite over it.
  if (gen !== _previewGen) return;

  const isFade  = overlapFadeToggle.checked;
  const horizontal = direction === 'horizontal';

  // Output size of each image. Mirrors _stitch_grid's fit: scale every
  // image to the tallest height (row) or widest width (column).
  if (document.getElementById('fit-toggle').checked) {
    const maxH = Math.max(...entries.map(e => e.h));
    const maxW = Math.max(...entries.map(e => e.w));
    entries = entries.map(e => horizontal
      ? { ...e, w: e.h === maxH ? e.w : Math.round(e.w * maxH / e.h), h: maxH }
      : { ...e, w: maxW, h: e.w === maxW ? e.h : Math.round(e.h * maxW / e.w) });
  }

  // _stitch_grid clamps overlap to one less than the smallest image, so
  // the slider never offers a value that wouldn't be applied.
  const minSide = Math.min(...entries.map(e => horizontal ? e.w : e.h));
  const maxOv = Math.max(0, minSide - 1);
  if (+overlapSlider.max !== maxOv) overlapSlider.max = maxOv;
  if ((parseInt(overlapSlider.value) || 0) > maxOv) _setOverlapPx(maxOv);
  const overlap = Math.min(Math.max(0, parseInt(overlapSlider.value) || 0), maxOv);

  // Natural canvas size (from full-resolution dimensions)
  let natW, natH;
  if (horizontal) {
    natW = entries.reduce((s, e) => s + e.w, 0) - overlap * (entries.length - 1);
    natH = Math.max(...entries.map(e => e.h));
  } else {
    natW = Math.max(...entries.map(e => e.w));
    natH = entries.reduce((s, e) => s + e.h, 0) - overlap * (entries.length - 1);
  }
  if (natW <= 0 || natH <= 0) return;

  // Scale down to ≤LIVE_MAX_PX on longest side for speed
  const scale = Math.min(1, LIVE_MAX_PX / Math.max(natW, natH));
  const cvW = Math.max(1, Math.round(natW * scale));
  const cvH = Math.max(1, Math.round(natH * scale));
  const ov  = Math.round(overlap * scale);

  const cv  = _liveCanvas;
  if (cv.width !== cvW || cv.height !== cvH) { cv.width = cvW; cv.height = cvH; }
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cvW, cvH);

  let pos = 0;
  for (let i = 0; i < entries.length; i++) {
    const e  = entries[i];
    const dw = Math.round(e.w * scale);
    const dh = Math.round(e.h * scale);
    if (i > 0) pos -= ov;
    const x = horizontal ? pos : 0, y = horizontal ? 0 : pos;
    if (isFade && i > 0 && ov > 0) _drawFaded(ctx, e, x, y, dw, dh, ov, horizontal);
    else ctx.drawImage(e.bmp, x, y, dw, dh);
    pos += horizontal ? dw : dh;
  }

  // Zoom view of seam edge (horizontal layouts only)
  if (_zoomActive && direction === 'horizontal' && ov > 0) {
    const dw0 = Math.round(entries[0].w * scale);
    _lastZoomParams = { cv, dw0, ov, cvW, cvH };
    _renderZoomStrip(_lastZoomParams);
  } else {
    zoomPanel.style.display = 'none';
  }

  _liveDrawCount++;
  _showPreview(previewLive, cvW >= cvH);
  _setPreviewState('live');
  updateStatus();
}

// Counts live-canvas paints. A server result whose <img> finishes decoding
// after a newer live paint must not cover it.
let _liveDrawCount = 0;

// Bumped whenever the preview pane gets authoritative content (a server
// result), invalidating any live composite still in flight.
let _previewGen = 0;

function _showPreview(el, landscape) {
  previewLive.style.display = el === previewLive ? '' : 'none';
  previewImg.style.display  = el === previewImg  ? '' : 'none';
  appBody.classList.toggle('layout-side',  !landscape);
  appBody.classList.toggle('layout-stack',  landscape);
  appBody.classList.add('has-preview');
}

let _livePreviewPending = false;
function scheduleLivePreview() {
  if (_livePreviewPending) return;
  _livePreviewPending = true;
  requestAnimationFrame(() => { _livePreviewPending = false; updateLivePreview(); });
}

function _fmtSize(bytes) {
  return bytes >= 1024 * 1024 ? (bytes / 1024 / 1024).toFixed(1) + ' MB'
                              : Math.round(bytes / 1024) + ' KB';
}

// Full-quality server render, used as the preview for grids the live
// composite can't draw. Runs automatically after edits; a sequence number
// drops responses that a newer request has superseded.
let _serverPreviewTimer = null;
let _serverPreviewSeq = 0;
function scheduleServerPreview() {
  clearTimeout(_serverPreviewTimer);
  if (imgCount() < 2) return;
  _serverPreviewTimer = setTimeout(runServerPreview, 400);
}

async function runServerPreview() {
  if (imgCount() < 2) return;
  const seq = ++_serverPreviewSeq;
  const sig = _requestSignature();
  if (appBody.classList.contains('has-preview')) _setPreviewState('pending');
  try {
    const res = await fetch('/preview', { method: 'POST', body: _buildFormData() });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(`Server error ${res.status}: ${msg}`);
    }
    const blob = await res.blob();
    if (seq !== _serverPreviewSeq) return;
    _lastServerResult = { ...sig, blob };
    _previewGen++;
    const liveDrawsAtResult = _liveDrawCount;
    if (_previewObjectURL) URL.revokeObjectURL(_previewObjectURL);
    _previewObjectURL = URL.createObjectURL(blob);
    previewImg.onload = () => {
      if (_liveDrawCount !== liveDrawsAtResult || seq !== _serverPreviewSeq) return;
      const w = previewImg.naturalWidth, h = previewImg.naturalHeight;
      _lastServerResult.caption = `${w} × ${h} px · ${_fmtSize(blob.size)}`;
      _showPreview(previewImg, w >= h);
      if (_matchesLastServerResult(_requestSignature())) _setPreviewState('final', _lastServerResult.caption);
      else _setPreviewState('pending');
    };
    previewImg.src = _previewObjectURL;
  } catch (err) {
    if (seq !== _serverPreviewSeq) return;
    console.error(err);
    showError('Preview failed: ' + err.message);
  }
}

document.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (!mod) return;
  if (e.key === 'z' && !e.shiftKey && _undo) { e.preventDefault(); undoRemove(); }
  else if (e.key === 'Enter' && imgCount() >= 2) { e.preventDefault(); downloadBtn.click(); }
  else if (e.key === 'o') { e.preventDefault(); openPicker(); }
});

const quitBtn   = document.getElementById('quit-btn');
const stoppedEl = document.getElementById('stopped');
// Two-step: the first click arms the button, a second within 3 s quits.
// (An in-page confirm, not window.confirm.)
let _quitArmTimer = null;
function _disarmQuit() {
  clearTimeout(_quitArmTimer);
  _quitArmTimer = null;
  quitBtn.classList.remove('armed');
  quitBtn.textContent = 'Quit';
}
quitBtn.addEventListener('blur', () => { if (_quitArmTimer) _disarmQuit(); });
quitBtn.addEventListener('click', async () => {
  if (!_quitArmTimer) {
    quitBtn.classList.add('armed');
    quitBtn.textContent = 'Click again to quit';
    _quitArmTimer = setTimeout(_disarmQuit, 3000);
    return;
  }
  clearTimeout(_quitArmTimer);
  _quitArmTimer = null;
  quitBtn.classList.remove('armed');
  quitBtn.disabled = true;
  quitBtn.textContent = 'Quitting…';
  try {
    const res = await fetch('/quit', { method: 'POST', headers: { 'X-Requested-With': 'fetch' } });
    if (!res.ok) throw Object.assign(new Error(`Server error ${res.status}`), { fromServer: true });
  } catch (err) {
    // A network error just means the server went away mid-response — that's
    // the goal. Only an actual HTTP error response means quitting failed.
    if (err.fromServer) {
      showError('Quit failed: ' + err.message);
      quitBtn.disabled = false;
      quitBtn.textContent = 'Quit';
      return;
    }
  }
  stoppedEl.classList.add('visible');
  // Browsers only honour this for tabs with a single history entry (as
  // opened by webbrowser.open) or script-opened ones; otherwise it's a
  // no-op and the "stopped" overlay above stays as the fallback.
  window.close();
});

// Names the download after the first image, e.g. beach.jpg → beach_stitched.jpg.
function _downloadName() {
  const first = grid.flat().find(Boolean);
  const base = first ? first.name.replace(/\.[^.]*$/, '') : '';
  return (base || 'stitched') + (base ? '_stitched' : '') + '.jpg';
}

downloadBtn.addEventListener('click', async () => {
  if (!hasImages() || downloadBtn.disabled) return;
  _statusError = null;
  const prev = downloadBtn.textContent;
  downloadBtn.textContent = 'Downloading…';
  downloadBtn.disabled = true;
  try {
    const sig = _requestSignature();
    let blob;
    if (_matchesLastServerResult(sig)) {
      blob = _lastServerResult.blob;
    } else {
      processingEl.classList.add('visible');
      const res = await fetch('/stitch', { method: 'POST', body: _buildFormData() });
      if (!res.ok) throw new Error(res.statusText);
      blob = await res.blob();
      _lastServerResult = { ...sig, blob };
    }
    const url  = URL.createObjectURL(blob);
    Object.assign(document.createElement('a'), { href: url, download: _downloadName() }).click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (err) {
    showError('Download failed: ' + err.message);
  } finally {
    processingEl.classList.remove('visible');
    downloadBtn.textContent = prev;
    downloadBtn.disabled = false;
  }
});
</script>
</body>
</html>"""


def _stitch_grid(layout, paths, td, fit=False, overlap=0, overlap_blend="hard", preopened=None, preopened_params=None):
    """
    Composite a 2D grid of images into a single JPEG.

    layout     — 2D list of file IDs (strings) or None for empty cells
    paths      — {id: absolute_path}; only read for ids missing from preopened
    td         — unused (kept for signature compatibility; the output is
                 produced in memory)
    preopened  — optional {id: Image} of already-decoded (EXIF-transposed, RGB)
                 images, used instead of re-opening from paths to avoid a
                 second full-resolution decode (e.g. after auto-overlap
                 suggestion already opened them).
    preopened_params — optional {id: (qtables, quality, subsampling, is_grayscale)}
                 precomputed via _jpeg_params_from_bytes while the caller
                 already had the raw bytes in hand, so this function doesn't
                 need to re-read the first cell's file from disk.

    Column widths = max image width per column.
    Row heights   = max image height per row.
    Empty cells   = black fill.
    Encoding params taken from the first non-None cell (top-left scan order).

    Returns (output_buffer, applied_overlap) — output_buffer is a BytesIO
    positioned at 0, or None if there was nothing to stitch. applied_overlap
    reflects any clamping or non-1D zeroing this function did to the
    requested overlap, so callers can report what was actually used rather
    than what was asked for.
    """
    preopened = preopened or {}
    preopened_params = preopened_params or {}
    first_path = None
    first_fid = None
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
                img = preopened.get(fid)
                if img is None:
                    with open(paths[fid], "rb") as fh:
                        img = _decode_rgb(fh.read())
                if first_path is None:
                    first_path = paths[fid]
                    first_fid = fid
                col_widths[c]  = max(col_widths[c],  img.width)
                row_heights[r] = max(row_heights[r], img.height)
                grid_row.append(img)
            else:
                grid_row.append(None)
        grid.append(grid_row)

    if first_path is None:
        return None, 0

    # Compute fit's target dimensions up front (cheap arithmetic) so the
    # oversize-canvas check below can reject requests before any LANCZOS
    # resampling runs. The actual resize happens later, after that check.
    fit_new_w = None  # per-column new width, horizontal fit only
    fit_new_h = None  # per-cell new height matrix, vertical fit only
    if fit:
        if num_rows == 1:
            # Single-row horizontal: scale each image to the tallest row height
            target_h = row_heights[0]
            fit_new_w = [0] * num_cols
            for c, img in enumerate(grid[0]):
                if img is not None:
                    fit_new_w[c] = (
                        round(img.width * target_h / img.height) if img.height != target_h else img.width
                    )
            col_widths = list(fit_new_w)
        else:
            # Vertical single-column or multi-row grid: scale each image to its column width
            fit_new_h = [[None] * num_cols for _ in range(num_rows)]
            new_row_heights = [0] * num_rows
            for r, row in enumerate(grid):
                for c, img in enumerate(row):
                    if img is not None:
                        target_w = col_widths[c]
                        h = round(img.height * target_w / img.width) if img.width != target_w else img.height
                        fit_new_h[r][c] = h
                        new_row_heights[r] = max(new_row_heights[r], h)
            row_heights = new_row_heights

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
        return None, 0

    MAX_CANVAS_PX = 16384
    if total_w > MAX_CANVAS_PX or total_h > MAX_CANVAS_PX:
        return None, 0

    # Now that the request has cleared the oversize check, actually perform
    # the fit resampling using the dimensions computed above.
    if fit:
        if num_rows == 1:
            target_h = row_heights[0]
            for c, img in enumerate(grid[0]):
                if img is not None and img.height != target_h:
                    grid[0][c] = img.resize((fit_new_w[c], target_h), Image.LANCZOS)
        else:
            for r, row in enumerate(grid):
                for c, img in enumerate(row):
                    if img is not None and img.width != col_widths[c]:
                        grid[r][c] = img.resize((col_widths[c], fit_new_h[r][c]), Image.LANCZOS)

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
            _paste_overlap(canvas, img, (x, 0), overlap if c > 0 else 0, overlap_blend, "horizontal")
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
            _paste_overlap(canvas, img, (0, y), overlap if r > 0 else 0, overlap_blend, "vertical")
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

    if first_fid in preopened_params:
        qtables, quality, subsampling, is_grayscale = preopened_params[first_fid]
    else:
        qtables, quality, subsampling, is_grayscale = _jpeg_params(first_path)

    output_buf = io.BytesIO()
    if is_grayscale:
        canvas = canvas.convert("L")
        if qtables:
            canvas.save(output_buf, "JPEG", qtables=qtables)
        else:
            canvas.save(output_buf, "JPEG", quality=quality)
    elif qtables:
        canvas.save(output_buf, "JPEG", qtables=qtables, subsampling=subsampling)
    else:
        canvas.save(output_buf, "JPEG", quality=quality, subsampling=subsampling)

    output_buf.seek(0)
    return output_buf, overlap


def _create_web_app(on_quit=None):
    """Build the Flask app for the web UI.

    on_quit — called (with no args) when the UI's Quit button POSTs /quit;
              the caller decides how to actually stop serving.
    """
    import json as _json
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
        # Default "0" (no overlap) so callers that omit the field — anything
        # other than the bundled UI, which always sends it explicitly — get
        # plain concatenation rather than a silently auto-applied overlap.
        overlap_raw   = request.form.get("overlap", "0")
        overlap_blend = request.form.get("overlap_blend", "hard")

        # Only uploads the layout actually references are read, and they stay
        # in memory: no temp-file write + re-read round trip. Each one is
        # decoded once (in parallel) and its JPEG encoding params are taken
        # from the same bytes, so _stitch_grid never touches the disk.
        referenced = []
        for row in layout:
            for fid in row:
                if fid and fid in request.files and fid not in referenced:
                    referenced.append(fid)
        raws = [request.files[fid].read() for fid in referenced]
        try:
            decoded = _decode_many(raws)
        except Exception:
            abort(400, "Could not decode one of the images")
        opened = dict(zip(referenced, decoded))
        opened_params = {fid: _jpeg_params_from_bytes(raw) for fid, raw in zip(referenced, raws)}
        paths = {fid: request.files[fid].filename or fid for fid in referenced}
        del raws

        if overlap_raw == "auto":
            n_rows = len(layout)
            direction = "horizontal" if n_rows == 1 else "vertical"
            imgs = [opened[fid] for row in layout for fid in row if fid in opened]
            overlap = _suggest_overlap(imgs, direction) if len(imgs) >= 2 else 0
        else:
            try:
                overlap = max(0, int(overlap_raw))
            except (ValueError, TypeError):
                overlap = 0

        buf, overlap = _stitch_grid(
            layout, paths, None, fit=fit, overlap=overlap, overlap_blend=overlap_blend,
            preopened=opened, preopened_params=opened_params,
        )
        if buf is None:
            abort(500, "Nothing to stitch")
        return buf, overlap, overlap_blend

    @app.route("/stitch", methods=["POST"])
    def stitch():
        from flask import make_response
        buf, overlap, overlap_blend = _stitch_to_buf()
        resp = make_response(send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="stitched.jpg"))
        resp.headers["X-Overlap-Px"]    = str(overlap)
        resp.headers["X-Overlap-Blend"] = overlap_blend
        return resp

    @app.route("/preview", methods=["POST"])
    def preview():
        from flask import make_response
        buf, overlap, overlap_blend = _stitch_to_buf()
        resp = make_response(send_file(buf, mimetype="image/jpeg", as_attachment=False))
        resp.headers["X-Overlap-Px"]    = str(overlap)
        resp.headers["X-Overlap-Blend"] = overlap_blend
        return resp

    @app.route("/quit", methods=["POST"])
    def quit_server():
        # A custom header can't be sent cross-origin without a CORS preflight
        # (which this app never approves), so other web pages can't use a
        # plain form POST to stop the server. The Origin check backs that up.
        if request.headers.get("X-Requested-With") != "fetch":
            abort(403)
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            abort(403)
        if on_quit:
            on_quit()
        return "", 204

    return app


def _run_web_server(port=5001):
    """Launch the drag-and-drop web UI and open it in the default browser."""
    if not _HAS_FLASK:
        print(
            "Flask is required for the web UI.\n"
            "Install it with:  pip3 install flask --break-system-packages\n"
            "Or install via Homebrew:  brew upgrade jpegconcat"
        )
        sys.exit(1)

    import threading, time, webbrowser
    from werkzeug.serving import make_server

    server = None

    def _shutdown_soon():
        # Let the /quit response flush before the serve loop stops.
        time.sleep(0.2)
        server.shutdown()

    app = _create_web_app(
        on_quit=lambda: threading.Thread(target=_shutdown_soon, daemon=True).start()
    )

    try:
        server = make_server("127.0.0.1", port, app, threaded=True)
    except SystemExit:
        # werkzeug prints its own "Address already in use" and exits; add our hint.
        print(f"Try: jpegconcat --web --port {port + 1}")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"Image Stitcher → {url}  (Quit button or Ctrl-C to stop)")

    def _open():
        time.sleep(0.6)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("Image Stitcher stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate JPEG images preserving encoding params.",
        usage=(
            "%(prog)s img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg]"
            " [--direction h|v|auto] [--order auto|as-given]\n"
            "       %(prog)s [--port PORT]   (no images: open the web UI)"
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
                        help="Launch the drag-and-drop web UI (default when no images are given)")
    parser.add_argument("--port",       type=int, default=5001,
                        help="Port for --web mode (default: 5001)")
    args = parser.parse_args()

    if args.web or not args.images:
        _run_web_server(port=args.port)
        return

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
