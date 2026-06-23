#!/usr/bin/env python3
"""
server.py — Local web server for the drag-and-drop image stitching UI.

Run:  python3 server.py
Open: http://localhost:5001

Note: This path always re-encodes through Pillow (jpegtran lossless DCT path
is not used). Encoding params (quantization tables + subsampling) are read from
the first image in the grid, matching the CLI's Pillow fallback behavior.
"""

import io
import json
import os
import sys
import tempfile

try:
    from flask import Flask, request, send_file, abort, jsonify
except ImportError:
    print("Flask not installed. Run: pip3 install flask")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip3 install Pillow")
    sys.exit(1)

from concat_jpeg import _jpeg_params

app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/stitch', methods=['POST'])
def stitch():
    layout_raw = request.form.get('layout')
    if not layout_raw:
        abort(400, 'Missing layout')

    layout = json.loads(layout_raw)

    # Drop entirely-empty rows
    layout = [row for row in layout if any(c is not None for c in row)]
    if not layout:
        abort(400, 'Empty layout')

    # Pad all rows to the same length
    num_cols = max(len(r) for r in layout)
    for row in layout:
        while len(row) < num_cols:
            row.append(None)

    # Drop entirely-empty columns
    non_empty_cols = [c for c in range(num_cols)
                      if any(r[c] is not None for r in layout)]
    if not non_empty_cols:
        abort(400, 'No images in layout')
    layout = [[row[c] for c in non_empty_cols] for row in layout]

    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for key, f in request.files.items():
            ext = os.path.splitext(f.filename)[1] or '.jpg'
            dest = os.path.join(td, key + ext)
            f.save(dest)
            paths[key] = dest

        output = _stitch_grid(layout, paths, td)
        if output is None:
            abort(500, 'Nothing to stitch')

        buf = io.BytesIO()
        with open(output, 'rb') as fh:
            buf.write(fh.read())
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg',
                         as_attachment=True,
                         download_name='stitched.jpg')


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
                img = Image.open(paths[fid]).convert('RGB')
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

    canvas = Image.new('RGB', (total_w, total_h), (0, 0, 0))
    y_off = 0
    for r, row in enumerate(grid):
        x_off = 0
        for c, img in enumerate(row):
            if img:
                canvas.paste(img, (x_off, y_off))
            x_off += col_widths[c]
        y_off += row_heights[r]

    output_path = os.path.join(td, 'output.jpg')
    qtables, quality, subsampling, is_grayscale = _jpeg_params(first_path)

    if is_grayscale:
        canvas = canvas.convert('L')
        if qtables:
            canvas.save(output_path, 'JPEG', qtables=qtables)
        else:
            canvas.save(output_path, 'JPEG', quality=quality)
    elif qtables:
        canvas.save(output_path, 'JPEG', qtables=qtables, subsampling=subsampling)
    else:
        canvas.save(output_path, 'JPEG', quality=quality, subsampling=subsampling)

    return output_path


if __name__ == '__main__':
    print('Image Stitcher — http://localhost:5001')
    app.run(debug=False, port=5001)
