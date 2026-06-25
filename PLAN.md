# Plan: Drag-and-Drop Web UI for Image Stitching

You're building a local web tool that lets users visually arrange images into a grid layout, then stitches them using the existing Python logic in `concat_jpeg.py`. It runs alongside the CLI — same codebase, new interface.

## Decision log

- **Architecture:** The web UI is served by Flask embedded in `concat_jpeg.py`. The HTML/CSS/JS is embedded as a string (`_WEB_UI_HTML`) and served via the built-in `_run_web_server()`. User runs `python3 concat_jpeg.py --web` (or the installed `jpegconcat --web` command). The frontend was originally a standalone `index.html` but is now embedded to keep the Homebrew formula shipping a single file.

- **Layout model:** Grid-snap with variable dimensions. Each row's height = the tallest image in that row; each column's width = the widest in that column. Size mismatches get black-fill padding, matching existing behavior.

- **Frontend stack:** Plain HTML/CSS/JS — embedded as a raw string literal in `concat_jpeg.py` (`_WEB_UI_HTML`), no npm, no build step, no external file dependency.

- **Output delivery:** Server streams the stitched JPEG bytes back; browser triggers a file download.

- **Grid emergence:** Drop-to-build. Canvas starts empty. The grid grows as images are dragged in.

- **Snap mechanism:** Edge drop zones. Dragging over the canvas reveals highlighted insertion zones on the edges of existing cells (left, right, above, below). Dropping there extends the grid in that direction.

- **Post-placement editing:** Drag-to-swap between cells + X button to remove a cell. Removing a cell collapses the grid to fill the gap.

## Open questions / assumptions

- `concat_jpeg.py` will be imported as a module. If its stitching logic isn't cleanly separable from the `if __name__ == "__main__"` block, a small refactor will be needed to expose a callable function — assumed to be minor.
- Grid collapse behavior on removal (e.g., removing the middle cell of a 1×3) will fill left-to-right; exact collapse rules to be decided during implementation.
