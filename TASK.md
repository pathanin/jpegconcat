# Task: Drag-and-Drop Web UI for Image Stitching

A local web tool for arranging images into a grid visually, then stitching them with the Python logic in `concat_jpeg.py`. It runs alongside the CLI from the same codebase.

This file is a brief for whoever works on the UI next. **Must hold** is the behaviour people rely on. **Current approach** is what exists today; treat it as a starting point and change it when you have a better idea, then update this file.

## Must hold

- **Preview matches the output.** Anything the preview shows, Download saves, and the reverse. That includes black fill, overlap, blend, rotation and flip.
- **Stitching follows `_stitch_grid`.** Each row is as tall as its tallest image and each column as wide as its widest. Size mismatches are filled with black, unless Fit is on. Overlap only applies to single-row or single-column layouts. If the UI wants different behaviour, change `_stitch_grid` and its tests first, then the UI.
- **Output keeps the source encoding.** The server re-encodes with the first image's quantization tables and subsampling. The browser never produces the final JPEG.
- **Local only.** The server binds to 127.0.0.1, and `/quit` keeps its same-origin check.

## Current approach

- **Architecture.** Flask in `concat_jpeg.py` serves the UI from `web/`:
  - `index.html` holds the markup.
  - `style.css` holds the styles.
  - `app.js` holds the behaviour.

  Run with `jpegconcat` (no arguments) or `python3 concat_jpeg.py`. The UI used to be embedded as a string so the Homebrew formula shipped one file. It was split out so it's cheaper to read and edit (for agents and people) and so standard tooling works on it. The cost is that the formula now installs `web/` too.
- **Frontend stack.** Plain HTML, CSS and JS: no npm, no build step, and one script with top-level globals. This keeps edits instant. Adopt a build step or modules if the UI outgrows that, but keep install simple for Homebrew users.
- **Building the grid.**
  - The canvas starts empty and the grid grows as images are dropped in.
  - Dropping on a cell's edge inserts beside it; dropping on its centre replaces it.
  - The "+" buttons on the grid's edges open the file picker for a new slot.
- **Editing.**
  - Drag one cell onto another to swap or move it.
  - × removes a cell (with undo), and the grid then compacts.
  - Each cell can be rotated and flipped from its hover toolbar, or with R, Shift+R, H and V when focused.
  - Rotate/flip never re-encode the file in the browser. Each cell keeps its original upload plus an orientation (`{rot, flip}`), sent as the `transforms` field; the server applies it with `_apply_orientation`. Thumbnails and the live preview draw the same orientation client-side.
- **Preview.**
  - Single-row and single-column layouts get an instant client-side composite.
  - Other grids get a debounced server render.
  - Download reuses the last server render when nothing has changed since.
  - Output is capped at 16384 px per side (`MAX_CANVAS_PX`, mirrored in `app.js`). The live preview warns when a layout is over it; the server answers 413 with a plain-text reason.
  - Server errors are plain text, shown as-is in the status line.
- **Visual style.** Minimal, with a neutral palette and one blue accent; the details are in the project memory. This is the current look, not a constraint. Colours are tokens in `:root` at the top of `style.css`.

## Open ideas

- Find a clearer way to fill grids whose image sizes differ, beyond suggesting Fit.
