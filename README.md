# concat_jpeg

Joins 2+ JPEGs side-by-side or stacked, re-encoding at the original quality
so the output file size stays close to the sum of the inputs.
For 2 images, it auto-detects the correct order and orientation by matching
edge colors — no need to specify anything.

Also ships a **drag-and-drop browser UI** — run `jpegconcat --web` to open it.

---

## Install

**Via Homebrew (recommended):**

```
brew tap pathanin/jpegconcat https://github.com/pathanin/jpegconcat
brew trust pathanin/jpegconcat
brew install jpegconcat
jpegconcat photo1.jpg photo2.jpg
```

> **Note (Homebrew 6.0+):** Homebrew 6.0 requires taps to be trusted before installing.
> The `brew trust pathanin/jpegconcat` step above grants that trust. Without it the build
> fails silently inside the sandbox. Run `brew untrust pathanin/jpegconcat` to revoke.

**Manually (requires Python 3):**

```
pip3 install Pillow numpy flask
python3 concat_jpeg.py photo1.jpg photo2.jpg
```

---

## Browser UI

```
jpegconcat --web
```

Opens a drag-and-drop interface at `http://localhost:5001`. Drop images anywhere
on the page to build a grid, then click **Stitch & Download**.

**Grid building:**
- Drop an image onto the **left or right edge** of a cell → inserts a new column
- Drop onto the **top or bottom edge** → inserts a new row
- Drop onto the **center** of a cell → replaces that image
- Drag one cell onto another → **swap** them
- Click **×** on a cell → removes it (grid compacts automatically)
- Click **+** on the right or bottom border → adds an empty column/row

**Per-image transforms (hover any cell):**
- **↺ ↻** — rotate 90° counter-clockwise / clockwise
- **↔ ↕** — flip horizontal / vertical

**Custom port:**

```
jpegconcat --web --port 8080
```

---

## CLI usage

```
jpegconcat photo1.jpg photo2.jpg
```

Output is saved as `concat.jpg` in the same folder as your images.
If `concat.jpg` already exists, it saves as `concat_2.jpg`, `concat_3.jpg`, etc.

**Override layout or order if needed:**

```
jpegconcat a.jpg b.jpg --direction horizontal   # force side-by-side
jpegconcat a.jpg b.jpg --direction vertical     # force top/bottom
jpegconcat a.jpg b.jpg --order as-given         # keep the order you typed
jpegconcat a.jpg b.jpg --output my_name.jpg     # custom output path
```

**More than 2 images:**

```
jpegconcat a.jpg b.jpg c.jpg
```

---

## What it prints

The script shows edge seam scores (lower = better match) so you can
see why it chose the order/direction it did, plus a size report.
