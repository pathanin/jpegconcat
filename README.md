# concat_jpeg

Joins 2+ JPEGs side-by-side or stacked, re-encoding at the original quality
so the output file size stays close to the sum of the inputs.
For 2 images, it auto-detects the correct order and orientation by matching
edge colors — no need to specify anything.

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
pip3 install Pillow numpy
python3 concat_jpeg.py photo1.jpg photo2.jpg
```

---

## Usage

```
python3 concat_jpeg.py photo1.jpg photo2.jpg
```

Output is saved as `concat.jpg` in the same folder as your images.
If `concat.jpg` already exists, it saves as `concat_2.jpg`, `concat_3.jpg`, etc.

**Override layout or order if needed:**

```
python3 concat_jpeg.py a.jpg b.jpg --direction horizontal   # force side-by-side
python3 concat_jpeg.py a.jpg b.jpg --direction vertical     # force top/bottom
python3 concat_jpeg.py a.jpg b.jpg --order as-given         # keep the order you typed
python3 concat_jpeg.py a.jpg b.jpg --output my_name.jpg     # custom output path
```

**More than 2 images:**

```
python3 concat_jpeg.py a.jpg b.jpg c.jpg
```

---

## What it prints

The script shows edge seam scores (lower = better match) so you can
see why it chose the order/direction it did, plus a size report.
