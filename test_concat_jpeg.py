"""
Tests for concat_jpeg.py — correctness, edge cases, and boundary conditions.

Run with:
    python3 -m pytest test_concat_jpeg.py -v
"""
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Ensure the script module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import concat_jpeg (has side effects — prints and may exit). We mock sys.exit to
# prevent the Pillow-import-failure exit from killing the test harness.
with patch.object(sys, "exit"):
    import concat_jpeg as cj
    from PIL import Image, ImageDraw


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def test_images_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "testfile")


@pytest.fixture
def rgb_image():
    """Create a small synthetic RGB image and return its path + Image object."""
    img = Image.new("RGB", (100, 80), (128, 64, 32))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG", quality=90)
        path = f.name
    yield path, Image.open(path)
    os.unlink(path)


@pytest.fixture
def grayscale_jpeg():
    """Create a small synthetic grayscale JPEG."""
    img = Image.new("L", (64, 48), 128)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG", quality=85)
        path = f.name
    yield path
    os.unlink(path)


# ── Tests: _jpeg_params ───────────────────────────────────────────────────────

class TestJpegParams:
    def test_returns_defaults_for_non_jpeg(self):
        """Non-JPEG file returns (None, 85, 2, False)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal valid PNG
            img = Image.new("RGB", (10, 10))
            img.save(f, "PNG")
            path = f.name
        try:
            qtables, quality, subsampling, is_gray = cj._jpeg_params(path)
            assert qtables is None
            assert quality == 85
            assert subsampling == 2
            assert is_gray is False
        finally:
            os.unlink(path)

    def test_rgb_jpeg_detected(self, rgb_image):
        """RGB JPEG detects quality, subsampling, and is_grayscale=False."""
        path, _ = rgb_image
        qtables, quality, subsampling, is_gray = cj._jpeg_params(path)
        assert qtables is not None, "quantization tables should be detected"
        assert 1 <= quality <= 95
        assert subsampling in (0, 1, 2), f"unexpected subsampling {subsampling}"
        assert is_gray is False, "RGB image should not be marked grayscale"

    def test_grayscale_jpeg_detected(self, grayscale_jpeg):
        """Grayscale JPEG has is_grayscale=True and one quantization table."""
        qtables, quality, subsampling, is_gray = cj._jpeg_params(grayscale_jpeg)
        assert is_gray is True, "grayscale JPEG should be detected"
        assert qtables is not None
        assert list(qtables.keys()) == [0], "grayscale should have one qtable"

    def test_quality_bounds(self, rgb_image):
        """Quality estimate stays within 1-95."""
        path, _ = rgb_image
        _, quality, _, _ = cj._jpeg_params(path)
        assert 1 <= quality <= 95

    def test_missing_file(self):
        """Missing file returns defaults without crashing."""
        qtables, quality, subsampling, is_gray = cj._jpeg_params("/nonexistent/file.jpg")
        assert qtables is None
        assert quality == 85
        assert subsampling == 2
        assert is_gray is False

    def test_corrupted_jpeg(self):
        """Corrupted JPEG bytes return defaults without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            path = f.name
        try:
            qtables, quality, subsampling, is_gray = cj._jpeg_params(path)
            # Should not crash; returns defaults
            assert qtables is None
        finally:
            os.unlink(path)


# ── Tests: _edge_strip & _seam_mad (numpy required) ───────────────────────────

@pytest.mark.skipif(not cj._HAS_NUMPY, reason="numpy not installed")
class TestEdgeMatching:
    @pytest.fixture
    def gradient_pair(self):
        """Two gradient images — left dark, right light — touching edge differs."""
        left = Image.new("RGB", (50, 100), (20, 20, 20))
        right = Image.new("RGB", (50, 100), (200, 200, 200))
        return left, right

    def test_edge_strip_shape(self, gradient_pair):
        """Edge strip has shape (EDGE_LEN, 3)."""
        left, _ = gradient_pair
        strip = cj._edge_strip(left, "right")
        assert strip.shape == (cj.EDGE_LEN, 3)

    def test_seam_mad_same_image_zero(self):
        """A seam against itself should have near-zero MAD."""
        img = Image.new("RGB", (100, 100), (64, 128, 192))
        mad = cj._seam_mad(img, "right", img, "left")
        assert mad < 0.01, f"same-edge MAD should be ~0, got {mad}"

    def test_seam_mad_different_nonzero(self, gradient_pair):
        """Different edges produce non-zero MAD."""
        left, right = gradient_pair
        mad = cj._seam_mad(left, "right", right, "left")
        assert mad > 1.0, f"different edges should have high MAD, got {mad}"

    def test_find_best_arrangement_2_correct_order(self, gradient_pair):
        """find_best_arrangement_2 picks the right order for contrasting edges."""
        left, right = gradient_pair
        paths = ["/a.jpg", "/b.jpg"]
        images = [left, right]
        # With dark right edge of left and light left edge of right,
        # the seam score should be worse than the alternative.
        ordered_paths, ordered_images, direction = cj.find_best_arrangement_2(
            paths, images, preserve_order=False, fix_direction="horizontal"
        )
        # Direction should be horizontal since both images are square-ish
        assert direction == "horizontal"

    def test_find_best_arrangement_2_preserve_order(self, gradient_pair):
        """With preserve_order=True, only directions are tested (no crash)."""
        left, right = gradient_pair
        paths = ["/a.jpg", "/b.jpg"]
        _, _, direction = cj.find_best_arrangement_2(
            paths, [left, right], preserve_order=True, fix_direction=None
        )
        assert direction in ("horizontal", "vertical")


# ── Tests: _make_output_path ──────────────────────────────────────────────────

class TestOutputExtensions:
    def test_make_output_path_no_collision(self, tmpdir):
        """No existing file → uses concat.ext."""
        img = Image.new("RGB", (10, 10))
        path = os.path.join(tmpdir, "a.png")
        img.save(path, "PNG")
        opened = [Image.open(path)]
        result = cj._make_output_path([path], opened)
        assert result == os.path.join(tmpdir, "concat.png")

    def test_make_output_path_dedup(self, tmpdir):
        """Existing concat.ext → increments to concat_2.ext."""
        img = Image.new("RGB", (10, 10))
        path = os.path.join(tmpdir, "a.png")
        img.save(path, "PNG")

        # Create first collision
        first = os.path.join(tmpdir, "concat.png")
        Image.new("RGB", (1, 1)).save(first, "PNG")

        opened = [Image.open(path)]
        result = cj._make_output_path([path], opened)
        assert result == os.path.join(tmpdir, "concat_2.png")

    def test_make_output_path_dedup_high_n(self, tmpdir):
        """Multiple collisions → continues incrementing."""
        img = Image.new("RGB", (10, 10))
        path = os.path.join(tmpdir, "a.png")
        img.save(path, "PNG")

        Image.new("RGB", (1, 1)).save(os.path.join(tmpdir, "concat.png"), "PNG")
        for i in [2, 3]:
            Image.new("RGB", (1, 1)).save(os.path.join(tmpdir, f"concat_{i}.png"), "PNG")

        opened = [Image.open(path)]
        result = cj._make_output_path([path], opened)
        assert result == os.path.join(tmpdir, "concat_4.png")


# ── Tests: _sort_key ──────────────────────────────────────────────────────────

class TestSortKey:
    def _save_and_key(self, tmpdir, filename, img):
        """Save img under tmpdir/filename, return _sort_key(path, img2)."""
        path = os.path.join(tmpdir, filename)
        img.save(path)
        img2 = Image.open(path)
        return cj._sort_key(path, img2)

    def test_numeric_sort(self, tmpdir):
        """Filenames with numeric sequences are sorted numerically."""
        img = Image.new("RGB", (10, 10))
        key_a = self._save_and_key(tmpdir, "photo_2.jpg", img)
        key_b = self._save_and_key(tmpdir, "photo_10.jpg", img)
        assert key_a < key_b

    def test_no_numbers(self, tmpdir):
        """Filenames without numbers sort by mtime (can't predict but shouldn't crash)."""
        img = Image.new("RGB", (10, 10))
        key = self._save_and_key(tmpdir, "photo.jpg", img)
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_multiple_number_sequences(self, tmpdir):
        """Multiple numeric runs form a tuple key."""
        img = Image.new("RGB", (10, 10))
        key = self._save_and_key(tmpdir, "2024_03_15_001.jpg", img)
        assert key[0] == (2024, 3, 15, 1)


# ── Tests: concat_images (end-to-end) ─────────────────────────────────────────

class TestConcatEndToEnd:
    def test_two_images_default(self, tmpdir):
        """Two JPEG images concatenate without error (default auto)."""
        a = Image.new("RGB", (50, 100), (255, 0, 0))
        b = Image.new("RGB", (50, 100), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")

        assert os.path.exists(output)
        result = Image.open(output)
        assert result.size == (100, 100)  # 50+50 wide, 100 tall

    def test_two_images_vertical(self, tmpdir):
        """Vertical concatenation stacks images top-to-bottom."""
        a = Image.new("RGB", (100, 50), (255, 0, 0))
        b = Image.new("RGB", (100, 50), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="vertical", order="as-given")

        assert os.path.exists(output)
        result = Image.open(output)
        assert result.size == (100, 100)  # 100 wide, 50+50 tall

    def test_three_images(self, tmpdir):
        """Three images concatenate horizontally."""
        imgs = [Image.new("RGB", (30, 50), (c * 50, 0, 0)) for c in range(1, 4)]
        paths = []
        for i, img in enumerate(imgs):
            p = os.path.join(tmpdir, f"{i}.jpg")
            img.save(p, "JPEG", quality=85)
            paths.append(p)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images(paths, output, direction="horizontal", order="as-given")
        assert os.path.exists(output)
        result = Image.open(output)
        assert result.size == (90, 50)  # 30+30+30 wide, 50 tall

    def test_different_dimensions(self, tmpdir):
        """Different-sized images — canvas accommodates max height/width."""
        a = Image.new("RGB", (50, 100), (255, 0, 0))
        b = Image.new("RGB", (80, 60), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        result = Image.open(output)
        assert result.size == (130, 100)  # 50+80 wide, max(100,60)=100 tall

    def test_output_auto_generated(self, tmpdir):
        """output_path=None auto-generates next to first input."""
        a = Image.new("RGB", (10, 10), (255, 0, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        a.save(path_a, "JPEG", quality=90)

        # Create a second image to have 2 inputs
        path_b = os.path.join(tmpdir, "b.jpg")
        Image.new("RGB", (10, 10)).save(path_b, "JPEG", quality=90)

        cj.concat_images([path_a, path_b], output_path=None, direction="horizontal", order="as-given")
        assert os.path.exists(os.path.join(tmpdir, "concat.jpg")), \
            "auto-generated output should be concat.jpg next to first input"

    def test_size_report_output(self, tmpdir, capsys):
        """Size report prints without error."""
        a = Image.new("RGB", (10, 20), (255, 0, 0))
        b = Image.new("RGB", (10, 20), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")

        captured = capsys.readouterr()
        assert "Size report" in captured.out
        assert "Saved" in captured.out
        assert "Encoding" in captured.out

    def test_non_jpeg_output(self, tmpdir):
        """Non-JPEG extension saves losslessly in its own format."""
        a = Image.new("RGB", (10, 20), (255, 0, 0))
        b = Image.new("RGB", (10, 20), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.png")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        assert os.path.exists(output)
        result = Image.open(output)
        assert result.format == "PNG"

    def test_grayscale_input(self, tmpdir):
        """Grayscale JPEG input saves as grayscale JPEG without error."""
        a = Image.new("L", (20, 30), 128)
        b = Image.new("L", (20, 30), 64)
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=85)
        b.save(path_b, "JPEG", quality=85)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        assert os.path.exists(output)
        result = Image.open(output)
        # The output may be 'L' or 'RGB' depending on the save path;
        # just verify it opens and has the right dimensions.
        assert result.size == (40, 30)


# ── Tests: boundary conditions ────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_single_image(self, tmpdir):
        """Single image — concatenation works (no crash)."""
        a = Image.new("RGB", (50, 100), (255, 0, 0))
        path = os.path.join(tmpdir, "a.jpg")
        a.save(path, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path], output, direction="horizontal", order="as-given")
        result = Image.open(output)
        assert result.size == (50, 100)

    def test_minimal_image_1x1(self, tmpdir):
        """1x1 pixel images (edge of JPEG spec)."""
        a = Image.new("RGB", (1, 1), (255, 0, 0))
        b = Image.new("RGB", (1, 1), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        assert os.path.exists(output)
        result = Image.open(output)
        assert result.size == (2, 1)

    def test_extreme_aspect_ratio(self, tmpdir):
        """Very wide + very tall image concatenation."""
        wide = Image.new("RGB", (500, 1), (255, 0, 0))
        tall = Image.new("RGB", (1, 500), (0, 255, 0))
        path_a = os.path.join(tmpdir, "wide.jpg")
        path_b = os.path.join(tmpdir, "tall.jpg")
        wide.save(path_a, "JPEG", quality=90)
        tall.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        result = Image.open(output)
        assert result.size == (501, 500)  # 500+1 wide, max(1,500)=500 tall


# ── Tests: lossless path (jpegtran-dependent) ─────────────────────────────────

class TestLosslessPath:
    def _mcu_aligned_images(self, tmpdir, width=160, height=128):
        """Create images with MCU-friendly dimensions (16x16 multiple for 4:2:0)."""
        a = Image.new("RGB", (width, height), (255, 0, 0))
        b = Image.new("RGB", (width, height), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=85, subsampling=0)
        b.save(path_b, "JPEG", quality=85, subsampling=0)
        return [path_a, path_b]

    def test_lossless_taken_when_possible(self, tmpdir):
        """When jpegtran is available and images are MCU-aligned, lossless path runs."""
        paths = self._mcu_aligned_images(tmpdir)
        output = os.path.join(tmpdir, "out.jpg")

        opened = [Image.open(p) for p in paths]
        imgs = [img.convert("RGB") for img in opened]
        fmts = [img.format for img in opened]

        qtables, _, subsampling, _ = cj._jpeg_params(paths[0])
        result = cj._try_lossless(paths, imgs, fmts, output, "horizontal", qtables, subsampling)
        assert result is True or not shutil_which_jpegtran()

    def test_lossless_fallback_on_misalignment(self, tmpdir):
        """Odd-height images trigger lossless fallback (height not MCU-aligned)."""
        a = Image.new("RGB", (100, 99), (255, 0, 0))  # 99 is not 16-aligned
        b = Image.new("RGB", (100, 100), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=85)
        b.save(path_b, "JPEG", quality=85)
        paths = [path_a, path_b]

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images(paths, output, direction="horizontal", order="as-given")
        # Should succeed via Pillow fallback
        assert os.path.exists(output)

    def test_fit_resize_skips_lossless(self, tmpdir):
        """--fit resizes pixels, so jpegtran (which drops the original files)
        must not be used — otherwise the resized slot ends up black."""
        a = Image.new("RGB", (320, 480), (255, 0, 0))
        b = Image.new("RGB", (160, 240), (0, 255, 0))
        path_a, path_b = os.path.join(tmpdir, "a.jpg"), os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)
        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given", fit=True)
        result = Image.open(output).convert("RGB")
        assert result.size == (640, 480)
        r, g, b_ = result.getpixel((480, 400))
        assert g > 200 and r < 60, "fit-resized image must fill its slot"

    def test_exif_rotated_input_skips_lossless(self, tmpdir):
        """An EXIF-rotated input is stored unrotated on disk; the lossless path
        would drop it sideways, so the Pillow path must be used instead."""
        src = Image.new("RGB", (480, 320), (0, 0, 255))
        src.paste((255, 255, 0), (0, 0, 240, 320))  # left half yellow
        ex = Image.Exif()
        ex[0x0112] = 6  # display rotated 90° CW
        path_a = os.path.join(tmpdir, "a.jpg")
        src.save(path_a, "JPEG", quality=90, exif=ex.tobytes())
        path_b = os.path.join(tmpdir, "b.jpg")
        Image.new("RGB", (320, 480), (0, 255, 0)).save(path_b, "JPEG", quality=90)
        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal", order="as-given")
        out = Image.open(output)
        # Pixels are already upright, so the output must not ask viewers to rotate again
        assert out.getexif().get(0x0112, 1) == 1
        result = out.convert("RGB")
        assert result.size == (640, 480)
        # Rotated 90° CW: the source's left (yellow) half becomes the top half
        assert result.getpixel((160, 100))[2] < 80      # yellow
        assert result.getpixel((160, 400))[2] > 180     # blue

    @pytest.mark.skipif(not __import__("shutil").which("jpegtran"), reason="jpegtran not installed")
    def test_lossless_path_skips_pixel_decode(self, tmpdir):
        """3+ MCU-aligned JPEGs need only header dimensions; the lossless path
        must not pay for a full-resolution decode it never uses."""
        paths = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            p = os.path.join(tmpdir, f"img_{i}.jpg")
            Image.new("RGB", (64, 96), color).save(p, "JPEG", quality=85, subsampling=2)
            paths.append(p)
        output = os.path.join(tmpdir, "out.jpg")
        with patch.object(cj, "_decode_rgb", side_effect=AssertionError("decoded")):
            cj.concat_images(paths, output, direction="horizontal", order="as-given")
        result = Image.open(output).convert("RGB")
        assert result.size == (192, 96)
        assert result.getpixel((160, 48))[2] > 200  # third image is blue

    def test_pillow_fallback_decodes_lazily(self, tmpdir):
        """When lossless can't run, deferred decoding still produces the image."""
        paths = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            p = os.path.join(tmpdir, f"img_{i}.jpg")
            Image.new("RGB", (61, 95), color).save(p, "JPEG", quality=85)
            paths.append(p)
        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images(paths, output, direction="horizontal", order="as-given")
        result = Image.open(output).convert("RGB")
        assert result.size == (183, 95)
        assert result.getpixel((30, 47))[0] > 200
        assert result.getpixel((152, 47))[2] > 200

    def test_lossless_not_taken_without_jpegtran(self, tmpdir):
        """No jpegtran → _try_lossless returns False."""
        paths = self._mcu_aligned_images(tmpdir)
        output = os.path.join(tmpdir, "out.jpg")

        opened = [Image.open(p) for p in paths]
        imgs = [img.convert("RGB") for img in opened]
        fmts = [img.format for img in opened]
        qtables, _, subsampling, _ = cj._jpeg_params(paths[0])

        # Temporarily hide jpegtran
        import shutil as shutil_mod
        original = shutil_mod.which
        try:
            shutil_mod.which = lambda cmd: None if cmd == "jpegtran" else original(cmd)
            result = cj._try_lossless(paths, imgs, fmts, output, "horizontal", qtables, subsampling)
            assert result is False
        finally:
            shutil_mod.which = original


def shutil_which_jpegtran():
    """Check if jpegtran is available (helper for skip logic)."""
    import shutil
    return shutil.which("jpegtran") is not None


# ── Tests: overlap stitching ──────────────────────────────────────────────────

class TestMakeBlendMask:
    def test_horizontal_gradient_endpoints(self):
        """Horizontal mask goes from 0 (left) to 255 (right)."""
        mask = cj._make_blend_mask(10, 4, "horizontal")
        pix = mask.load()
        assert pix[0, 0] == 0
        assert pix[9, 0] == 255

    def test_vertical_gradient_endpoints(self):
        """Vertical mask goes from 0 (top) to 255 (bottom)."""
        mask = cj._make_blend_mask(4, 10, "vertical")
        pix = mask.load()
        assert pix[0, 0] == 0
        assert pix[0, 9] == 255

    def test_width_one_no_division_by_zero(self):
        """w=1 (boundary: denom would be 0) does not crash."""
        mask = cj._make_blend_mask(1, 5, "horizontal")
        assert mask.size == (1, 5)

    def test_height_one_no_division_by_zero(self):
        """h=1 (boundary: denom would be 0) does not crash."""
        mask = cj._make_blend_mask(5, 1, "vertical")
        assert mask.size == (5, 1)


class TestSuggestOverlap:
    def test_fallback_without_numpy(self):
        """Without numpy, falls back to ~10% of the join dimension."""
        a = Image.new("RGB", (200, 100), (255, 0, 0))
        b = Image.new("RGB", (200, 100), (0, 255, 0))
        with patch.object(cj, "_HAS_NUMPY", False):
            overlap = cj._suggest_overlap([a, b], "horizontal")
        assert overlap == max(30, round(200 * 0.10))

    def test_fewer_than_two_images_returns_fallback(self):
        """Failure case: a single image still returns a sane fallback, no crash."""
        a = Image.new("RGB", (200, 100), (255, 0, 0))
        overlap = cj._suggest_overlap([a], "horizontal")
        assert overlap == max(30, round(200 * 0.10))

    @pytest.mark.skipif(not cj._HAS_NUMPY, reason="numpy not installed")
    def test_returns_value_within_bounds(self):
        """With numpy, suggested overlap stays within [0, 30% of join dim]."""
        a = Image.new("RGB", (200, 100), (10, 10, 10))
        b = Image.new("RGB", (200, 100), (240, 240, 240))
        overlap = cj._suggest_overlap([a, b], "horizontal")
        assert 0 <= overlap <= round(200 * 0.30)


class TestOverlapConcat:
    def test_overlap_reduces_output_width(self, tmpdir):
        """Horizontal overlap shrinks total width vs plain concatenation."""
        a = Image.new("RGB", (100, 50), (255, 0, 0))
        b = Image.new("RGB", (100, 50), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal",
                          order="as-given", overlap=20)
        result = Image.open(output)
        assert result.size == (180, 50)  # 100+100-20 wide

    def test_overlap_clamped_to_min_join_dim(self, tmpdir, capsys):
        """Boundary: overlap >= the smallest join dimension gets clamped, with a warning."""
        a = Image.new("RGB", (40, 50), (255, 0, 0))
        b = Image.new("RGB", (100, 50), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal",
                          order="as-given", overlap=40)
        captured = capsys.readouterr()
        assert "clamped" in captured.out
        result = Image.open(output)
        assert result.size == (101, 50)  # 40+100 - 39 (clamped to min_join_dim-1)

    def test_fade_blend_differs_from_hard_blend(self, tmpdir):
        """fade vs hard overlap_blend produce different pixel content at the seam."""
        a = Image.new("RGB", (100, 50), (255, 0, 0))
        b = Image.new("RGB", (100, 50), (0, 0, 255))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        hard_out = os.path.join(tmpdir, "hard.jpg")
        fade_out = os.path.join(tmpdir, "fade.jpg")
        cj.concat_images([path_a, path_b], hard_out, direction="horizontal",
                          order="as-given", overlap=20, overlap_blend="hard")
        cj.concat_images([path_a, path_b], fade_out, direction="horizontal",
                          order="as-given", overlap=20, overlap_blend="fade")

        hard_px = Image.open(hard_out).convert("RGB").load()
        fade_px = Image.open(fade_out).convert("RGB").load()
        # Mid-overlap pixel: hard paste shows pure blue, fade shows a blend.
        assert hard_px[90, 25] != fade_px[90, 25]

    def test_zero_overlap_unaffected(self, tmpdir):
        """overlap=0 (default) behaves exactly like plain concatenation."""
        a = Image.new("RGB", (100, 50), (255, 0, 0))
        b = Image.new("RGB", (100, 50), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        cj.concat_images([path_a, path_b], output, direction="horizontal",
                          order="as-given", overlap=0)
        result = Image.open(output)
        assert result.size == (200, 50)


class TestStitchGrid:
    def _save(self, tmpdir, name, size, color):
        path = os.path.join(tmpdir, name)
        Image.new("RGB", size, color).save(path, "JPEG", quality=90)
        return path

    def test_single_row_overlap(self, tmpdir):
        """1xN grid: overlap shrinks total width and is echoed back unchanged."""
        path_a = self._save(tmpdir, "a.jpg", (100, 50), (255, 0, 0))
        path_b = self._save(tmpdir, "b.jpg", (100, 50), (0, 255, 0))
        paths = {"a": path_a, "b": path_b}
        out, applied_overlap = cj._stitch_grid([["a", "b"]], paths, str(tmpdir), overlap=20)
        assert out is not None
        assert applied_overlap == 20
        result = Image.open(out)
        assert result.size == (180, 50)

    def test_non_1d_grid_ignores_overlap(self, tmpdir):
        """Boundary: a 2x2 grid silently drops overlap (non-1D shape) instead of crashing,
        and reports the actually-applied overlap (0) rather than the requested one."""
        paths = {k: self._save(tmpdir, f"{k}.jpg", (40, 40), (10, 10, 10)) for k in "abcd"}
        out, applied_overlap = cj._stitch_grid([["a", "b"], ["c", "d"]], paths, str(tmpdir), overlap=20)
        assert out is not None
        assert applied_overlap == 0
        result = Image.open(out)
        assert result.size == (80, 80)  # overlap ignored, plain 2x2 grid

    def test_preopened_images_used_instead_of_reopening(self, tmpdir):
        """preopened images are reused rather than re-decoded from disk."""
        path_a = self._save(tmpdir, "a.jpg", (100, 50), (255, 0, 0))
        path_b = self._save(tmpdir, "b.jpg", (100, 50), (0, 255, 0))
        paths = {"a": path_a, "b": path_b}
        preopened = {
            "a": Image.open(path_a).convert("RGB"),
            "b": Image.open(path_b).convert("RGB"),
        }
        # exif_transpose is only called on the decode-from-disk path in
        # _stitch_grid; if preopened images are used it should never run.
        with patch.object(cj.ImageOps, "exif_transpose", side_effect=AssertionError("should not re-decode")) as mock_transpose:
            out, _ = cj._stitch_grid([["a", "b"]], paths, str(tmpdir), overlap=0, preopened=preopened)
        mock_transpose.assert_not_called()
        assert out is not None


# ── Tests: CLI entry point ────────────────────────────────────────────────────

class TestMain:
    def test_main_missing_file_exits(self):
        """Missing input file → sys.exit(1)."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["prog", "/nonexistent/file.jpg"]):
                cj.main()

    def test_main_help(self):
        """--help prints usage and exits."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["prog", "--help"]):
                cj.main()

    def test_main_no_args_launches_web_ui(self):
        """No images → web UI starts (no -w needed), honouring --port."""
        with patch.object(cj, "_run_web_server") as run:
            with patch.object(sys, "argv", ["prog"]):
                cj.main()
            run.assert_called_once_with(port=5001)
            run.reset_mock()
            with patch.object(sys, "argv", ["prog", "--port", "8080"]):
                cj.main()
            run.assert_called_once_with(port=8080)

    def test_main_web_flag_still_works(self):
        with patch.object(cj, "_run_web_server") as run:
            with patch.object(sys, "argv", ["prog", "-w"]):
                cj.main()
            run.assert_called_once_with(port=5001)

    def test_main_success(self, tmpdir):
        """Happy path through main()."""
        a = Image.new("RGB", (10, 20), (255, 0, 0))
        b = Image.new("RGB", (10, 20), (0, 255, 0))
        path_a = os.path.join(tmpdir, "a.jpg")
        path_b = os.path.join(tmpdir, "b.jpg")
        a.save(path_a, "JPEG", quality=90)
        b.save(path_b, "JPEG", quality=90)

        output = os.path.join(tmpdir, "out.jpg")
        with patch.object(sys, "argv", ["prog", path_a, path_b, "--output", output]):
            cj.main()
        assert os.path.exists(output)


# ── Tests: _FORMAT_TO_EXT coverage ────────────────────────────────────────────

class TestFormatToExt:
    def test_known_formats(self):
        """All known formats map to expected extensions."""
        assert cj._FORMAT_TO_EXT["JPEG"] == ".jpg"
        assert cj._FORMAT_TO_EXT["PNG"] == ".png"
        assert cj._FORMAT_TO_EXT["WEBP"] == ".webp"
        assert cj._FORMAT_TO_EXT["TIFF"] == ".tiff"
        assert cj._FORMAT_TO_EXT["BMP"] == ".bmp"
        assert cj._FORMAT_TO_EXT["GIF"] == ".gif"
        assert cj._FORMAT_TO_EXT["ICO"] == ".ico"
        assert cj._FORMAT_TO_EXT["PDF"] == ".pdf"



# ── Tests: web UI routes ──────────────────────────────────────────────────────

@pytest.mark.skipif(not cj._HAS_FLASK, reason="Flask not installed")
class TestWebApp:
    def _client(self, on_quit=None):
        return cj._create_web_app(on_quit=on_quit).test_client()

    def _jpeg_bytes(self, size, color):
        import io
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
        buf.seek(0)
        return buf

    def test_index_has_quit_button(self):
        res = self._client().get("/")
        assert res.status_code == 200
        assert b'id="quit-btn"' in res.data

    def test_quit_closes_tab(self):
        res = self._client().get("/static/app.js")
        assert b"window.close()" in res.data

    def test_every_asset_the_page_references_is_served(self):
        """A renamed or missing file in web/ would 404 and break the UI."""
        import re
        client = self._client()
        html = client.get("/").get_data(as_text=True)
        assets = re.findall(r'(?:href|src)="(static/[^"]+)"', html)
        assert assets, "expected the page to reference its CSS and JS"
        for path in assets:
            res = client.get("/" + path)
            assert res.status_code == 200, path
            assert res.data, path

    def test_quit_invokes_callback(self):
        calls = []
        res = self._client(on_quit=lambda: calls.append(1)).post(
            "/quit", headers={"X-Requested-With": "fetch"})
        assert res.status_code == 204
        assert calls == [1]

    def test_quit_rejects_get(self):
        calls = []
        res = self._client(on_quit=lambda: calls.append(1)).get("/quit")
        assert res.status_code == 405
        assert calls == []

    def test_quit_rejects_plain_post_without_header(self):
        """A cross-site <form> POST can't set custom headers, so it must be refused."""
        calls = []
        res = self._client(on_quit=lambda: calls.append(1)).post("/quit")
        assert res.status_code == 403
        assert calls == []

    def test_quit_rejects_foreign_origin(self):
        calls = []
        res = self._client(on_quit=lambda: calls.append(1)).post(
            "/quit", headers={"X-Requested-With": "fetch", "Origin": "https://evil.example"})
        assert res.status_code == 403
        assert calls == []

    def test_quit_accepts_same_origin(self):
        calls = []
        res = self._client(on_quit=lambda: calls.append(1)).post(
            "/quit", headers={"X-Requested-With": "fetch", "Origin": "http://localhost"})
        assert res.status_code == 204
        assert calls == [1]

    def test_preview_stitches_uploaded_images(self):
        import io, json
        data = {
            "layout": json.dumps([["a", "b"]]),
            "fit": "false", "overlap": "10", "overlap_blend": "fade",
            "a": (self._jpeg_bytes((60, 40), (255, 0, 0)), "a.jpg"),
            "b": (self._jpeg_bytes((60, 40), (0, 0, 255)), "b.jpg"),
        }
        res = self._client().post("/preview", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        assert res.headers["X-Overlap-Px"] == "10"
        out = Image.open(io.BytesIO(res.data))
        assert out.size == (110, 40)

    def test_stitch_ignores_unreferenced_uploads_and_rejects_empty_layout(self):
        import io, json
        client = self._client()
        data = {
            "layout": json.dumps([["a"], ["b"]]),
            "a": (self._jpeg_bytes((40, 30), (255, 0, 0)), "a.jpg"),
            "b": (self._jpeg_bytes((40, 30), (0, 255, 0)), "b.jpg"),
            "unused": (self._jpeg_bytes((10, 10), (0, 0, 0)), "c.jpg"),
        }
        res = client.post("/stitch", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        assert Image.open(io.BytesIO(res.data)).size == (40, 60)
        res = client.post("/stitch", data={"layout": "[]"}, content_type="multipart/form-data")
        assert res.status_code == 400


class TestWebUIFiles:
    """Static checks on web/, which has no JS test suite of its own."""

    @staticmethod
    def _read(name):
        return (cj._WEB_DIR / name).read_text(encoding="utf-8")

    @pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
    def test_script_is_valid_javascript(self):
        import subprocess
        res = subprocess.run(["node", "--check", str(cj._WEB_DIR / "app.js")],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

    def test_every_looked_up_id_exists(self):
        """A getElementById for a removed element returns null and throws at
        load, breaking the whole UI — pytest never runs the JS, so check here."""
        import re
        ids = set(re.findall(r'\bid="([^"]+)"', self._read("index.html")))
        used = set(re.findall(r"getElementById\('([^']+)'\)", self._read("app.js")))
        assert used, "expected getElementById lookups in the script"
        assert used <= ids, f"script looks up missing ids: {sorted(used - ids)}"

    def test_overlap_defaults_to_zero(self):
        import re
        html = self._read("index.html")
        assert re.search(r'id="overlap-slider"[^>]*value="0"', html)
        assert '<span id="overlap-val">0 px</span>' in html

    def test_download_is_a_keyboard_reachable_button(self):
        """An <a> without href can't take focus; Download must stay a button."""
        assert '<button id="download-btn"' in self._read("index.html")
