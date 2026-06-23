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
  background: #111;
  color: #e2e2e2;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  background: #1c1c1c;
  border-bottom: 1px solid #2a2a2a;
  flex-shrink: 0;
}
h1 { font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; }
.header-right { display: flex; align-items: center; gap: 12px; }
#status { font-size: 0.8rem; color: #666; }

#stitch-btn {
  padding: 7px 16px;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s;
}
#stitch-btn:hover:not(:disabled) { background: #1d4ed8; }
#stitch-btn:disabled { background: #2a2a2a; color: #555; cursor: default; }

main {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
  overflow: auto;
}

#drop-zone {
  width: 380px;
  height: 260px;
  border: 2px dashed #333;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #555;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
  user-select: none;
}
#drop-zone.drag-over {
  border-color: #2563eb;
  color: #3b82f6;
  background: rgba(37,99,235,0.06);
}
#drop-zone svg { flex-shrink: 0; }
#drop-zone .hint { font-size: 0.75rem; color: #444; }
#file-input { display: none; }

#page-drop-ring {
  position: fixed;
  inset: 4px;
  border: 2px solid #2563eb;
  border-radius: 8px;
  pointer-events: none;
  z-index: 200;
  opacity: 0;
  transition: opacity 0.12s;
}
body.drag-files-active #page-drop-ring { opacity: 1; }

#grid-outer {
  display: none;
  flex-direction: column;
  align-items: flex-start;
}
#grid-scroll {
  display: flex;
  align-items: flex-start;
}
#grid {
  display: inline-grid;
  gap: 3px;
  background: #1c1c1c;
  padding: 3px;
  border-radius: 8px;
  border: 1px solid #2a2a2a;
}

.cell {
  position: relative;
  overflow: hidden;
  border-radius: 4px;
  background: #1a1a1a;
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
  width: 72px;
  height: 72px;
  border: 2px dashed #2a2a2a;
  background: #181818;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #3a3a3a;
  font-size: 1.3rem;
  cursor: default;
  transition: border-color 0.1s, color 0.1s, background 0.1s;
}
.cell.empty.drag-over-empty {
  border-color: #2563eb;
  color: #3b82f6;
  background: rgba(37,99,235,0.08);
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
  background: rgba(37,99,235,0.45);
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
  background: rgba(0,0,0,0.65);
  border: none;
  border-radius: 4px;
  color: #aaa;
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
  background: rgba(0,0,0,0.72);
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
  color: #bbb;
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

.toolbar-sep {
  width: 1px;
  height: 14px;
  background: #444;
  margin: 0 2px;
  flex-shrink: 0;
}

#add-col-btn {
  width: 22px;
  align-self: stretch;
  margin-left: 4px;
  background: #1c1c1c;
  border: 1px dashed #2a2a2a;
  border-radius: 6px;
  color: #3a3a3a;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
  flex-shrink: 0;
}
#add-col-btn:hover { background: #2563eb; color: #fff; border-color: #2563eb; }

#add-row-btn {
  height: 22px;
  margin-top: 4px;
  background: #1c1c1c;
  border: 1px dashed #2a2a2a;
  border-radius: 6px;
  color: #3a3a3a;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
  align-self: stretch;
}
#add-row-btn:hover { background: #2563eb; color: #fff; border-color: #2563eb; }

#processing {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.65);
  z-index: 100;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 14px;
  font-size: 0.9rem;
  color: #ccc;
}
#processing.visible { display: flex; }

@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
  width: 28px;
  height: 28px;
  border: 2px solid #333;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
</style>
</head>
<body>

<header>
  <h1>Image Stitcher</h1>
  <div class="header-right">
    <span id="status">Drop images to start</span>
    <button id="stitch-btn" disabled>Stitch &amp; Download</button>
  </div>
</header>

<main>
  <div id="drop-zone">
    <svg width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
    </svg>
    <span>Drop images anywhere</span>
    <span class="hint">or click to browse &middot; JPEG, PNG, WebP</span>
    <input type="file" id="file-input" accept="image/*" multiple>
  </div>

  <div id="grid-outer">
    <div id="grid-scroll">
      <div id="grid"></div>
      <button id="add-col-btn" title="Add column">+</button>
    </div>
    <button id="add-row-btn" title="Add row">+</button>
  </div>
</main>

<div id="page-drop-ring"></div>

<div id="processing">
  <div class="spinner"></div>
  <span>Stitching&hellip;</span>
</div>

<script>
'use strict';

let grid = [];
let dragSrc = null;
let dragEnterCount = 0;

const dropZoneEl   = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const gridOuter    = document.getElementById('grid-outer');
const gridEl       = document.getElementById('grid');
const addColBtn    = document.getElementById('add-col-btn');
const addRowBtn    = document.getElementById('add-row-btn');
const stitchBtn    = document.getElementById('stitch-btn');
const statusEl     = document.getElementById('status');
const processingEl = document.getElementById('processing');

const uid = () => Math.random().toString(36).slice(2, 9);

function readDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
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

async function applyTransform(r, c, type) {
  const cell = grid[r][c];
  if (!cell) return;
  const img = new Image();
  img.src = cell.dataURL;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
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
  const newDataURL = canvas.toDataURL('image/jpeg', 0.92);
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
  const newFile = new File([blob], cell.name, { type: 'image/jpeg' });
  grid[r][c] = { ...cell, dataURL: newDataURL, file: newFile };
  render();
}

async function makeCells(files) {
  const imgs = [...files].filter(f => f.type.startsWith('image/'));
  return Promise.all(imgs.map(async f => ({
    id: uid(), name: f.name, file: f, dataURL: await readDataURL(f),
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
    '<button class="btn-transform" title="Rotate CCW">↺</button>' +
    '<button class="btn-transform" title="Rotate CW">↻</button>' +
    '<div class="toolbar-sep"></div>' +
    '<button class="btn-transform" title="Flip horizontal">↔</button>' +
    '<button class="btn-transform" title="Flip vertical">↕</button>';
  const [rotateCCW, rotateCW, , flipH, flipV] = toolbar.children;
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
    if (sr !== r || sc !== c) { swapCells(sr, sc, r, c); render(); }
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
  return el;
}

function zoneAt(clientX, clientY, el) {
  const rect = el.getBoundingClientRect();
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top)  / rect.height;
  const E = 0.28;
  if (x < E)     return 'left';
  if (x > 1 - E) return 'right';
  if (y < E)     return 'top';
  if (y > 1 - E) return 'bottom';
  return 'center';
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
  document.body.classList.add('drag-files-active');
});
document.addEventListener('dragleave', e => {
  if (!isFileDrag(e)) return;
  dragEnterCount--;
  if (dragEnterCount <= 0) {
    dragEnterCount = 0;
    document.body.classList.remove('drag-files-active');
    clearAllHighlights();
  }
});
document.addEventListener('dragover', e => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  updateDropHighlight(e.clientX, e.clientY);
});
document.addEventListener('drop', e => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  dragEnterCount = 0;
  document.body.classList.remove('drag-files-active');
  clearAllHighlights();
  const target = findNearestZone(e.clientX, e.clientY);
  if (target.type === 'empty') initFromFiles(e.dataTransfer.files);
  else dropFilesOnCell(target.r, target.c, target.zone, e.dataTransfer.files);
});
document.addEventListener('dragend', () => {
  dragEnterCount = 0;
  document.body.classList.remove('drag-files-active');
  clearAllHighlights();
});

dropZoneEl.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => {
  if (e.target.files.length) initFromFiles(e.target.files);
  fileInput.value = '';
});

addColBtn.addEventListener('click', () => { if (hasImages()) { insertCol(nCols() - 1); render(); } });
addRowBtn.addEventListener('click', () => { if (hasImages()) { insertRow(nRows() - 1); render(); } });

stitchBtn.addEventListener('click', async () => {
  if (!hasImages()) return;
  processingEl.classList.add('visible');
  stitchBtn.disabled = true;
  try {
    const formData = new FormData();
    const layoutIds = grid.map(row => row.map(cell => cell ? cell.id : null));
    formData.append('layout', JSON.stringify(layoutIds));
    const seen = new Set();
    for (const row of grid) {
      for (const cell of row) {
        if (cell && !seen.has(cell.id)) {
          formData.append(cell.id, cell.file, cell.name);
          seen.add(cell.id);
        }
      }
    }
    const res = await fetch('/stitch', { method: 'POST', body: formData });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(`Server error ${res.status}: ${msg}`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'stitched.jpg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    statusEl.textContent = 'Downloaded!';
    setTimeout(() => {
      const n = imgCount();
      statusEl.textContent = `${n} image${n !== 1 ? 's' : ''} · ${nRows()}×${nCols()} grid`;
    }, 2500);
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error — see console';
    alert('Stitch failed:\n' + err.message);
  } finally {
    processingEl.classList.remove('visible');
    stitchBtn.disabled = imgCount() < 2;
  }
});
</script>
</body>
</html>"""


def _stitch_grid(layout, paths, td):
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
    grid = []
    for row in layout:
        img_row = []
        for fid in row:
            if fid and fid in paths:
                img = Image.open(paths[fid]).convert("RGB")
                img_row.append(img)
                if first_path is None:
                    first_path = paths[fid]
            else:
                img_row.append(None)
        grid.append(img_row)

    if first_path is None:
        return None

    num_rows = len(grid)
    num_cols = max(len(r) for r in grid)
    for row in grid:
        while len(row) < num_cols:
            row.append(None)

    col_widths  = [0] * num_cols
    row_heights = [0] * num_rows
    for r, row in enumerate(grid):
        for c, img in enumerate(row):
            if img:
                col_widths[c]  = max(col_widths[c],  img.width)
                row_heights[r] = max(row_heights[r], img.height)

    total_w = sum(col_widths)
    total_h = sum(row_heights)
    if total_w == 0 or total_h == 0:
        return None

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    y_off = 0
    for r, row in enumerate(grid):
        x_off = 0
        for c, img in enumerate(row):
            if img:
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
    # Suppress Flask startup banner
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return _WEB_UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/stitch", methods=["POST"])
    def stitch():
        layout_raw = request.form.get("layout")
        if not layout_raw:
            abort(400, "Missing layout")

        layout = _json.loads(layout_raw)
        layout = [row for row in layout if any(c is not None for c in row)]
        if not layout:
            abort(400, "Empty layout")

        num_cols = max(len(r) for r in layout)
        for row in layout:
            while len(row) < num_cols:
                row.append(None)

        non_empty = [c for c in range(num_cols) if any(r[c] is not None for r in layout)]
        if not non_empty:
            abort(400, "No images in layout")
        layout = [[row[c] for c in non_empty] for row in layout]

        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for key, f in request.files.items():
                ext = os.path.splitext(f.filename)[1] or ".jpg"
                dest = os.path.join(td, key + ext)
                f.save(dest)
                paths[key] = dest

            output = _stitch_grid(layout, paths, td)
            if output is None:
                abort(500, "Nothing to stitch")

            buf = io.BytesIO()
            with open(output, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0)
            return send_file(
                buf, mimetype="image/jpeg",
                as_attachment=True, download_name="stitched.jpg",
            )

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
    concat_images(args.images, args.output, direction, args.order)


if __name__ == "__main__":
    main()
