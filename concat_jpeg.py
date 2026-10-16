#!/usr/bin/env python3
"""
concat_jpeg.py — Concatenate JPEG images while preserving original encoding parameters.

Usage:
    python3 concat_jpeg.py img1.jpg img2.jpg [img3.jpg ...] [--output out.jpg]
                           [--direction horizontal|vertical|auto]
                           [--order auto|as-given]

Auto-detection (both on by default):
  order:     for 2 images (with numpy), the image with more content on its inner
             side goes first, unless one seam scores ≥5× better (a panorama);
             otherwise sorts by embedded numeric sequence in filename, then EXIF
             timestamp, then mtime
  direction: portrait-majority → horizontal, otherwise vertical

Metadata: every output path (jpegtran, Pillow, web UI) carries the first
image's EXIF (orientation reset, stale pixel dimensions dropped) and its ICC
colour profile. Metadata from the other images is discarded.
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
from pathlib import Path

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
        if getattr(_img, "quantization", None):
            qtables = _img.quantization
            tbl = qtables.get(0, [])
            if len(tbl) == 64:
                s = sum(tbl[k] * 100 / _STD_LUMA[k] for k in range(64)) / 64
                q = (200 - s) / 2 if s <= 100 else 5000 / s
                quality = max(1, min(95, round(q)))
    except Exception:
        pass

    return qtables, quality, subsampling, is_grayscale


def _source_meta(raw):
    """Return (exif, icc) from image bytes, ready to copy into the output.

    EXIF orientation is reset to 1 because pixels are decoded upright, and the
    pixel-dimension tags are dropped because they describe the source, not
    the composite. Either value is None when absent.
    """
    try:
        head = Image.open(io.BytesIO(raw))
    except Exception:
        return None, None
    icc = head.info.get("icc_profile") or None
    exif = head.info.get("exif") or None
    if exif:
        try:
            ex = Image.Exif()  # fresh copy: getexif() is cached on the image
            ex.load(exif)
            if ex.get(0x0112, 1) != 1:
                ex[0x0112] = 1
            sub = ex.get_ifd(0x8769)
            sub.pop(0xA002, None)  # PixelXDimension
            sub.pop(0xA003, None)  # PixelYDimension
            exif = ex.tobytes()
        except Exception:
            # Unparseable EXIF must not fail the stitch. Raw bytes are only
            # safe to copy when they don't ask viewers to rotate the pixels.
            try:
                upright = head.getexif().get(0x0112, 1) == 1
            except Exception:
                upright = False
            exif = exif if upright else None
    return exif, icc


def _save_jpeg(canvas, dest, params, grayscale, exif=None, icc=None):
    """Encode canvas as JPEG with the source's encoding params.

    params    — (qtables, quality, subsampling, is_grayscale) of the source
                whose encoding the output copies (the first image).
    grayscale — whether the output is grayscale (all inputs were).
    """
    qtables, quality, subsampling, src_grayscale = params
    kw = {}
    if exif:
        kw["exif"] = exif
    # A profile only describes output of the same colour model as its source.
    if icc and src_grayscale == grayscale:
        kw["icc_profile"] = icc
    if grayscale:
        canvas = canvas.convert("L")
    else:
        kw["subsampling"] = subsampling
    if qtables:
        kw["qtables"] = qtables
    else:
        kw["quality"] = quality
    canvas.save(dest, "JPEG", **kw)


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

def _try_lossless(image_paths, images, input_formats, output_path, direction, qtables, subsampling,
                  all_subsampling=None, exif=None, icc=None):
    """
    Attempt lossless DCT-level concatenation via jpegtran -drop.

    Returns True on success, False if any precondition is unmet.

    Metadata: jpegtran -copy all copies markers from its *input* file, which
    is the blank canvas here, never the dropped images. So `exif` and `icc`
    (the first image's) are written into that canvas and carried through.

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
    meta_kw = {k: v for k, v in (("exif", exif), ("icc_profile", icc)) if v}
    if qtables:
        blank.save(canvas_buf, "JPEG", qtables=qtables, subsampling=subsampling, **meta_kw)
    else:
        blank.save(canvas_buf, "JPEG", quality=85, subsampling=subsampling, **meta_kw)
    current = canvas_buf.getvalue()
    del blank, canvas_buf

    # Drop each source image into the canvas at its offset (DCT-level, no
    # re-encode). The growing canvas is piped through stdin/stdout rather
    # than staged in temp files. -copy all keeps the canvas's metadata.
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


def _orientation_direction(images):
    """Portrait-majority → horizontal (side by side), otherwise vertical."""
    portrait_count = sum(1 for img in images if img.height >= img.width)
    return "horizontal" if portrait_count * 2 >= len(images) else "vertical"


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
    direction = fix_direction or _orientation_direction(images)
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
    Raises CanvasTooLarge if the result exceeds what a JPEG can hold.
    """
    # Read every input exactly once; headers, encoding params and pixels all
    # come from these bytes.
    raws = []
    for p in image_paths:
        with open(p, "rb") as f:
            raws.append(f.read())
    raw_of = dict(zip(image_paths, raws))
    heads = [Image.open(io.BytesIO(r)) for r in raws]  # header-only, no pixel decode
    head_of = dict(zip(image_paths, heads))

    if output_path is None:
        output_path = _make_output_path(image_paths, heads)
    jpeg_out = os.path.splitext(output_path)[1].lower() in (".jpg", ".jpeg")

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
    # (seam matching or the Pillow re-encode). Header dimensions equal decoded
    # ones only when no EXIF rotation applies.
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
            direction = _orientation_direction(images)
            print(f"Direction: auto (orientation heuristic) → {direction}")
        else:
            direction = fix_direction

        if not _HAS_NUMPY and len(image_paths) == 2:
            print("(install numpy for edge-color seam matching)")

    print(f"\nLayout: {direction}")
    horizontal = direction == "horizontal"

    if fit:
        # The compositor does the resizing; here it only rules out jpegtran.
        if len({img.height if horizontal else img.width for img in images}) > 1:
            pixels_differ_from_file = True
        print(f"Fit: resized images to share {'height' if horizontal else 'width'}")

    # Encoding params and metadata come from the first image (in final order).
    first_raw = raw_of[image_paths[0]]
    params = _jpeg_params_from_bytes(first_raw)
    qtables, quality, subsampling, _ = params
    exif, icc = _source_meta(first_raw)
    grayscale = all(head_of[p].mode == "L" for p in image_paths)

    if overlap > 0:
        print("Note: overlap disables the lossless jpegtran fast-path; re-encoding via Pillow.")
    lossless = (
        jpeg_out and overlap == 0 and not pixels_differ_from_file
        and _try_lossless(
            image_paths, images, [head_of[p].format for p in image_paths], output_path,
            direction, qtables, subsampling,
            all_subsampling=[_parse_sof_subsampling(raw_of[p]) for p in image_paths],
            exif=exif, icc=None if params[3] else icc,  # the canvas is always colour
        )
    )

    # ── Pillow re-encode fallback ─────────────────────────────────────────────
    if not lossless:
        if not decoded:
            images = _decode_many([raw_of[p] for p in image_paths])
        grid = [images] if horizontal else [[img] for img in images]
        canvas, applied = _composite_grid(
            grid, fit=fit, overlap=overlap, overlap_blend=overlap_blend,
            max_px=_JPEG_MAX_PX if jpeg_out else None,
        )
        if applied < overlap:
            print(f"Warning: overlap clamped to {applied}px")
        if jpeg_out:
            _save_jpeg(canvas, output_path, params, grayscale, exif, icc)
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

    if jpeg_out:
        if lossless:
            enc = "lossless (jpegtran DCT)"
        elif qtables:
            enc = f"quality≈{quality} (exact source tables)"
        else:
            enc = f"quality={quality}"
        if grayscale and not lossless:
            ss_label = "grayscale"
        else:
            ss_label = ['4:4:4', '4:2:2', '4:2:0'][subsampling]
        print(f"\nEncoding: {enc}, subsampling={ss_label}")
    else:
        print(f"\nEncoding: {os.path.splitext(output_path)[1].lstrip('.').lower()} (lossless)")
    print(f"Saved: {output_path}")


# ── Grid compositing (shared by the CLI and the web UI) ───────────────────────

MAX_CANVAS_PX = 16384  # web UI cap per side; app.js mirrors it
_JPEG_MAX_PX = 65500   # what libjpeg can encode per side


class CanvasTooLarge(ValueError):
    """The composite would exceed the allowed size; the message says by how much."""


def _composite_grid(grid, fit=False, overlap=0, overlap_blend="hard", max_px=None):
    """
    Paste a rectangular 2D list of RGB Images (None = empty cell) onto one canvas.

    Column widths = max image width per column.
    Row heights   = max image height per row.
    Empty space   = black fill.
    fit           — single row: scale every image to the tallest height;
                    otherwise scale each image to its column's width.
    overlap       — only for a single row or column; clamped to one less than
                    the smallest join dimension, zeroed for other shapes.

    Returns (canvas, applied_overlap), or (None, 0) if there's nothing to draw.
    Raises CanvasTooLarge when a side would exceed max_px (None = no cap).
    """
    num_rows = len(grid)
    num_cols = max((len(r) for r in grid), default=0)
    col_widths  = [0] * num_cols
    row_heights = [0] * num_rows
    for r, row in enumerate(grid):
        for c, img in enumerate(row):
            if img is not None:
                col_widths[c]  = max(col_widths[c],  img.width)
                row_heights[r] = max(row_heights[r], img.height)
    if not any(col_widths):
        return None, 0

    # Compute fit's target sizes up front (cheap arithmetic) so the size cap
    # below can reject a request before any LANCZOS resampling runs.
    fit_size = {}  # (r, c) → new (w, h)
    if fit:
        if num_rows == 1:
            target_h = row_heights[0]
            for c, img in enumerate(grid[0]):
                if img is not None and img.height != target_h:
                    fit_size[0, c] = (round(img.width * target_h / img.height), target_h)
                    col_widths[c] = fit_size[0, c][0]
        else:
            new_row_heights = [0] * num_rows
            for r, row in enumerate(grid):
                for c, img in enumerate(row):
                    if img is None:
                        continue
                    h = img.height
                    if img.width != col_widths[c]:
                        h = round(img.height * col_widths[c] / img.width)
                        fit_size[r, c] = (col_widths[c], h)
                    new_row_heights[r] = max(new_row_heights[r], h)
            row_heights = new_row_heights

    horizontal_1d = num_rows == 1 and num_cols > 1
    vertical_1d   = num_cols == 1 and num_rows > 1
    if overlap > 0:
        if horizontal_1d:
            overlap = min(overlap, min(col_widths) - 1)
        elif vertical_1d:
            overlap = min(overlap, min(row_heights) - 1)
        else:
            overlap = 0
        overlap = max(0, overlap)

    total_w = sum(col_widths)  - (overlap * (num_cols - 1) if horizontal_1d else 0)
    total_h = sum(row_heights) - (overlap * (num_rows - 1) if vertical_1d else 0)
    if total_w <= 0 or total_h <= 0:
        return None, 0
    if max_px is not None and (total_w > max_px or total_h > max_px):
        raise CanvasTooLarge(
            f"Output would be {total_w} × {total_h} px; the limit is {max_px} px per side."
        )

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    y = 0
    for r, row in enumerate(grid):
        x = 0
        for c, img in enumerate(row):
            if img is not None:
                if (r, c) in fit_size:
                    img = img.resize(fit_size[r, c], Image.LANCZOS)
                if horizontal_1d and c > 0:
                    x -= overlap
                    _paste_overlap(canvas, img, (x, y), overlap, overlap_blend, "horizontal")
                elif vertical_1d and r > 0:
                    y -= overlap
                    _paste_overlap(canvas, img, (x, y), overlap, overlap_blend, "vertical")
                else:
                    canvas.paste(img, (x, y))
            elif horizontal_1d and c > 0:
                x -= overlap
            elif vertical_1d and r > 0:
                y -= overlap
            x += col_widths[c]
        y += row_heights[r]
    return canvas, overlap


# ── Web UI ────────────────────────────────────────────────────────────────────

# The browser UI lives in web/ next to this file: index.html (markup),
# style.css and app.js, served as-is by _create_web_app.
_WEB_DIR = Path(__file__).resolve().parent / "web"


_TURN_CW = {1: Image.Transpose.ROTATE_270, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_90}


def _apply_orientation(img, rot=0, flip=False):
    """Mirror left-right if flip, then turn rot quarter turns clockwise (the
    order app.js uses for its thumbnails and live preview)."""
    if flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rot:
        img = img.transpose(_TURN_CW[rot])
    return img


def _stitch_grid(layout, paths=None, fit=False, overlap=0, overlap_blend="hard",
                 preopened=None, preopened_params=None, preopened_meta=None, max_px=MAX_CANVAS_PX):
    """
    Composite a 2D grid of images into a single JPEG (see _composite_grid for
    the layout rules).

    layout     — 2D list of file IDs (strings) or None for empty cells; not modified
    paths      — {id: path}; read only for ids missing from the preopened dicts
    preopened  — optional {id: Image} of already-decoded (upright, RGB) images
    preopened_params — optional {id: _jpeg_params_from_bytes(raw)}
    preopened_meta   — optional {id: _source_meta(raw)}

    Encoding params, EXIF and ICC come from the first non-empty cell (row-major).
    The output is grayscale only when every image is.

    Returns (output_buffer, applied_overlap): a BytesIO positioned at 0, or
    None if there was nothing to stitch. Raises CanvasTooLarge past max_px.
    """
    paths = paths or {}
    images = dict(preopened or {})
    params = dict(preopened_params or {})
    meta = dict(preopened_meta or {})

    def _load(fid):
        """Fill in whatever the caller didn't preopen, from one disk read."""
        raw = None
        if fid in paths:
            with open(paths[fid], "rb") as fh:
                raw = fh.read()
        if fid not in images and raw is not None:
            images[fid] = _decode_rgb(raw)
        params.setdefault(fid, _jpeg_params_from_bytes(raw) if raw else (None, 85, 2, False))
        meta.setdefault(fid, _source_meta(raw) if raw else (None, None))

    num_cols = max(len(r) for r in layout)
    used = []
    for row in layout:
        for fid in row:
            if fid is not None and fid not in used and (fid in images or fid in paths):
                if fid not in images or fid not in params or fid not in meta:
                    _load(fid)
                used.append(fid)
    if not used:
        return None, 0

    grid = [[images.get(fid) if fid is not None else None for fid in row] + [None] * (num_cols - len(row))
            for row in layout]
    canvas, overlap = _composite_grid(grid, fit, overlap, overlap_blend, max_px)
    if canvas is None:
        return None, 0

    first = used[0]
    grayscale = all(params[fid][3] for fid in used)
    output_buf = io.BytesIO()
    _save_jpeg(canvas, output_buf, params[first], grayscale, *meta[first])
    output_buf.seek(0)
    return output_buf, overlap


def _create_web_app(on_quit=None):
    """Build the Flask app for the web UI.

    on_quit — called (with no args) when the UI's Quit button POSTs /quit;
              the caller decides how to actually stop serving.
    """
    import json as _json
    from flask import Flask, abort, request, send_file, send_from_directory
    from werkzeug.exceptions import HTTPException

    app = Flask(__name__, static_folder=str(_WEB_DIR), static_url_path="/static")
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB total upload limit
    # Suppress Flask startup banner
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return send_from_directory(_WEB_DIR, "index.html")

    # Errors go back as plain text, which the UI shows as-is in its status line.
    @app.errorhandler(HTTPException)
    def _plain_error(e):
        return e.description or e.name, e.code, {"Content-Type": "text/plain; charset=utf-8"}

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
        if not all(c is None or isinstance(c, str) for row in layout for c in row):
            abort(400, "Invalid layout")
        num_cols = max(len(r) for r in layout)
        layout = [row + [None] * (num_cols - len(row)) for row in layout]
        non_empty = [c for c in range(num_cols) if any(r[c] is not None for r in layout)]
        return [[row[c] for c in non_empty] for row in layout]

    def _parse_transforms():
        """{id: (rot, flip)} from the optional `transforms` field: mirror
        left-right if flip, then rot quarter turns clockwise."""
        raw = request.form.get("transforms")
        if not raw:
            return {}
        try:
            data = _json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError
            out = {}
            for fid, t in data.items():
                rot, flip = t.get("rot", 0), t.get("flip", False)
                if type(rot) is not int or not 0 <= rot <= 3 or not isinstance(flip, bool):
                    raise ValueError
                out[fid] = (rot, flip)
            return out
        except (ValueError, AttributeError, _json.JSONDecodeError):
            abort(400, "Invalid transforms")

    def _stitch_to_buf():
        layout = _validate_layout()
        transforms = _parse_transforms()
        fit = request.form.get("fit", "false").lower() == "true"
        # Default "0" (no overlap) so callers that omit the field — anything
        # other than the bundled UI, which always sends it explicitly — get
        # plain concatenation rather than a silently auto-applied overlap.
        overlap_raw   = request.form.get("overlap", "0")
        overlap_blend = request.form.get("overlap_blend", "hard")
        if overlap_blend not in ("hard", "fade"):
            abort(400, "overlap_blend must be hard or fade")

        # Only uploads the layout actually references are read, and they stay
        # in memory. Each one is decoded once (in parallel); encoding params
        # and metadata come from the same bytes, so _stitch_grid never touches
        # the disk. Rotate/flip are applied here, to the original upload, so
        # the output keeps the source's encoding.
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
        opened = {fid: _apply_orientation(img, *transforms.get(fid, (0, False)))
                  for fid, img in zip(referenced, decoded)}
        opened_params = {fid: _jpeg_params_from_bytes(raw) for fid, raw in zip(referenced, raws)}
        opened_meta = {fid: _source_meta(raw) for fid, raw in zip(referenced, raws)}
        del raws, decoded

        if overlap_raw == "auto":
            direction = "horizontal" if len(layout) == 1 else "vertical"
            imgs = [opened[fid] for row in layout for fid in row if fid in opened]
            overlap = _suggest_overlap(imgs, direction) if len(imgs) >= 2 else 0
        else:
            try:
                overlap = max(0, int(overlap_raw))
            except (ValueError, TypeError):
                overlap = 0

        try:
            buf, overlap = _stitch_grid(
                layout, fit=fit, overlap=overlap, overlap_blend=overlap_blend,
                preopened=opened, preopened_params=opened_params, preopened_meta=opened_meta,
            )
        except CanvasTooLarge as e:
            abort(413, str(e))
        if buf is None:
            abort(400, "No images in layout")
        return buf, overlap, overlap_blend

    def _stitch_response(as_attachment):
        buf, overlap, overlap_blend = _stitch_to_buf()
        resp = send_file(buf, mimetype="image/jpeg", as_attachment=as_attachment,
                         download_name="stitched.jpg")
        resp.headers["X-Overlap-Px"]    = str(overlap)
        resp.headers["X-Overlap-Blend"] = overlap_blend
        return resp

    @app.route("/stitch", methods=["POST"])
    def stitch():
        return _stitch_response(as_attachment=True)

    @app.route("/preview", methods=["POST"])
    def preview():
        return _stitch_response(as_attachment=False)

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
    if not (_WEB_DIR / "index.html").is_file():
        print(
            f"Web UI files not found in {_WEB_DIR}\n"
            "The web/ folder must sit next to concat_jpeg.py."
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

    try:
        concat_images(args.images, args.output, direction, args.order, fit=args.fit, overlap=args.overlap, overlap_blend=args.overlap_blend)
    except CanvasTooLarge as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
