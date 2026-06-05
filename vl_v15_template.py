from paddleformers.datasets.template.template import *
from paddleformers.datasets.template.mm_plugin import *
from paddleformers.datasets.template.augment_utils import *
import inspect
import io
import os
import cv2
import numpy as np
import random
from typing import Optional, Tuple, List, Dict
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from loguru import logger

_LOG_FILE = os.environ.get("AUG_LOG_FILE", "images_transform.log")
_LOG_LEVEL = os.environ.get("AUG_LOG_LEVEL", "DEBUG")
logger.add(_LOG_FILE, rotation="10 MB", level=_LOG_LEVEL)

try:
    from scipy import ndimage as _ndimage
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ============================================================
# Probability is built into __call__ so these can be used directly
# or wrapped in CustomRandomApply (pass p=1.0 then).
# ============================================================
class GaussianNoise:
    """Random Gaussian noise — simulates low-quality scan."""
    def __init__(self, p: float = 0.3, mean: float = 0.0, std: float = 25.0):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: GaussianNoise (std={})", self.std)
        arr = np.array(img.convert("RGB")).astype(np.float32)
        arr = np.clip(arr + np.random.normal(self.mean, self.std, arr.shape), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class GaussianBlur:
    """Random Gaussian blur — simulates defocus."""
    def __init__(self, p: float = 0.3, radius_range: Tuple[float, float] = (1.0, 3.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        radius = random.uniform(*self.radius_range)
        logger.info("apply: GaussianBlur (radius={:.2f})", radius)
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


class JpegCompression:
    """Random JPEG compression — simulates compression artefacts."""
    def __init__(self, p: float = 0.3, quality_range: Tuple[int, int] = (40, 85)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        quality = random.randint(*self.quality_range)
        logger.info("apply: JpegCompression (quality={})", quality)
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


# PIL resampling filter names → constants  (shared by RandomScale / RandomRotation)
_INTERP_MAP: Dict[str, int] = {
    "bicubic":  Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest":  Image.NEAREST,
    "lanczos":  Image.LANCZOS,
}


class ColorJitter:
    """PIL-based colour jitter (brightness / contrast / saturation / hue)."""
    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.0,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: ColorJitter (b={}, c={}, s={}, h={})",
                     self.brightness, self.contrast, self.saturation, self.hue)
        img = img.convert("RGB")
        for enhancer_cls, strength in [
            (ImageEnhance.Brightness, self.brightness),
            (ImageEnhance.Contrast,   self.contrast),
            (ImageEnhance.Color,      self.saturation),
        ]:
            factor = random.uniform(1 - strength, 1 + strength)
            img = enhancer_cls(img).enhance(factor)
        if self.hue > 0:
            arr = np.array(img)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
            shift = random.uniform(-self.hue * 180, self.hue * 180)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            img = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        return img


# ============================================================
# Effects adapted from image_effects/ (text_renderer)
# ============================================================
class Curve:
    """Sine-wave curve distortion (adapted from image_effects/curve.py)."""
    def __init__(self, p: float = 0.5, period: float = 180, amplitude: Tuple[float, float] = (1, 5)):
        assert amplitude[0] < amplitude[1], "amplitude must be (min, max) with min < max"
        self.p = p
        self.period = period
        self.amplitude = amplitude

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: Curve (period={}, amplitude={})", self.period, self.amplitude)
        img = img.convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        max_val = np.random.uniform(*self.amplitude)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        img_x, img_y_base = np.meshgrid(xs, ys)
        offset = max_val * np.sin(2 * np.pi * xs / self.period)
        img_y = (img_y_base + offset).astype(np.float32)
        dst = cv2.remap(arr, img_x, img_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(dst)


class DropoutHorizontal:
    """Horizontal line dropout (adapted from image_effects/dropout_horizontal.py)."""
    def __init__(self, p: float = 0.5, num_line: int = 3, thickness: int = 3):
        self.p = p
        self.num_line = num_line
        self.thickness = thickness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: DropoutHorizontal (num_line={}, thickness={})", self.num_line, self.thickness)
        img = img.convert("RGB")
        if img.height <= self.thickness + 1:
            return img
        arr = np.array(img)
        for _ in range(self.num_line):
            row = random.randint(1, img.height - self.thickness - 1)
            vals = np.random.randint(0, 21, (self.thickness, img.width, 1), dtype=np.uint8)
            arr[row : row + self.thickness] = np.broadcast_to(vals, (self.thickness, img.width, 3)).copy()
        return Image.fromarray(arr)


class DropoutVertical:
    """Vertical line dropout (adapted from image_effects/dropout_vertical.py)."""
    def __init__(self, p: float = 0.5, num_line: int = 8, thickness: int = 3):
        self.p = p
        self.num_line = num_line
        self.thickness = thickness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: DropoutVertical (num_line={}, thickness={})", self.num_line, self.thickness)
        img = img.convert("RGB")
        if img.width <= self.thickness + 1:
            return img
        arr = np.array(img)
        for _ in range(self.num_line):
            col = random.randint(1, img.width - self.thickness - 1)
            # Generate all noise values at once; broadcast single channel to RGB
            vals = np.random.randint(0, 21, (img.height, self.thickness, 1), dtype=np.uint8)
            arr[:, col : col + self.thickness] = np.broadcast_to(vals, (img.height, self.thickness, 3)).copy()
        return Image.fromarray(arr)


class DropoutRand:
    """Random pixel dropout (adapted from image_effects/dropout_rand.py)."""
    def __init__(self, p: float = 0.5, dropout_p: Tuple[float, float] = (0.2, 0.4)):
        self.p = p
        self.dropout_p = dropout_p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: DropoutRand (dropout_p={})", self.dropout_p)
        img = img.convert("RGB")
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        total = h * w
        drop_count = random.randint(int(total * self.dropout_p[0]), int(total * self.dropout_p[1]))
        ys = np.random.randint(0, h, drop_count)
        xs = np.random.randint(0, w, drop_count)
        # Darken dropped pixels: scale each channel by a random fraction in [0, 1],
        # matching the original intent of random.randint(0, pixel_value) per channel.
        orig = arr[ys, xs].astype(np.float32)                    # (N, 3)
        fracs = np.random.rand(drop_count, 3).astype(np.float32) # (N, 3) in [0, 1]
        arr[ys, xs] = (orig * fracs).astype(np.uint8)
        return Image.fromarray(arr)


class LineOverlay:
    """Draw a line at a random position (adapted from image_effects/line.py)."""
    def __init__(self, p: float = 0.5, thickness: Tuple[int, int] = (1, 3)):
        self.p = p
        self.thickness = thickness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: LineOverlay (thickness={})", self.thickness)
        img = img.convert("RGB")   # convert() returns a new image; .copy() would be redundant
        draw = ImageDraw.Draw(img)
        # random.randint is inclusive on both ends; np.random.randint would exclude the upper bound
        thickness = random.randint(*self.thickness)
        color = tuple(int(v) for v in np.random.randint(0, 170, 3))
        positions = [
            lambda: draw.line((0, thickness, img.width, thickness), fill=color, width=thickness),
            lambda: draw.line((0, img.height - thickness, img.width, img.height - thickness), fill=color, width=thickness),
            lambda: draw.line((thickness, 0, thickness, img.height), fill=color, width=thickness),
            lambda: draw.line((img.width - thickness, 0, img.width - thickness, img.height), fill=color, width=thickness),
            lambda: draw.line((0, img.height // 2, img.width, img.height // 2), fill=color, width=thickness),
            lambda: draw.line((img.width // 2, 0, img.width // 2, img.height), fill=color, width=thickness),
        ]
        random.choice(positions)()
        return img


class ImagePadding:
    """Add random padding around image (adapted from image_effects/padding.py)."""
    def __init__(self, p: float = 0.5, w_ratio: Tuple[float, float] = (0.0, 0.05), h_ratio: Tuple[float, float] = (0.0, 0.3), center: bool = False):
        self.p = p
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.center = center

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: ImagePadding (w_ratio={}, h_ratio={}, center={})",
                     self.w_ratio, self.h_ratio, self.center)
        img = img.convert("RGB")   # strip alpha so paste onto RGB canvas is correct
        w_ratio = random.uniform(*self.w_ratio)
        h_ratio = random.uniform(*self.h_ratio)
        new_w = int(img.width + img.width * w_ratio)
        new_h = int(img.height + img.height * h_ratio)
        new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        if self.center:
            xy = (int((new_w - img.width) / 2), int((new_h - img.height) / 2))
        else:
            x = random.randint(0, max(0, new_w - img.width))
            y = random.randint(0, max(0, new_h - img.height))
            xy = (x, y)
        new_img.paste(img, xy)
        return new_img


class TextBorder:
    """Add a border around dark text regions (adapted from image_effects/text_border.py)."""
    def __init__(self, p: float = 0.5, border_width: Tuple[int, int] = (1, 3)):
        self.p = p
        self.border_width = border_width

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: TextBorder (border_width={})", self.border_width)
        if not _SCIPY_AVAILABLE:
            return img
        img_rgba = img.convert("RGBA")
        gray_arr = np.array(img.convert("L"))
        # Keep as bool — binary_dilation is designed for boolean input; casting to uint8
        # first would work but creates an unnecessary intermediate array.
        text_mask = gray_arr < 128              # bool
        if not np.any(text_mask):
            return img
        # random.randint is inclusive on both ends; np.random.randint would exclude the upper bound
        bw = random.randint(*self.border_width)
        border_mask = _ndimage.binary_dilation(text_mask, iterations=bw).astype(np.uint8) * 255
        text_mask_u8 = text_mask.astype(np.uint8) * 255
        border_only = np.clip(border_mask - text_mask_u8, 0, 255).astype(np.uint8)
        border_img = Image.new("RGBA", img.size, (0, 0, 0, 255))
        border_img.putalpha(Image.fromarray(border_only, mode="L"))
        result = Image.alpha_composite(img_rgba, border_img)
        return result.convert("RGB")


# ============================================================
# Albumentations-style effects (standalone, no dependency)
# ============================================================
class AlbumentationsMotionBlur:
    """Directional motion blur approximating A.MotionBlur."""
    def __init__(self, p: float = 0.5, blur_limit: int = 3):
        self.p = p
        self.blur_limit = blur_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: AlbumentationsMotionBlur (blur_limit={})", self.blur_limit)
        img = img.convert("RGB")
        arr = np.array(img)
        # Pick an odd kernel size uniformly in [3, 2*blur_limit+1].
        # max() ensures range(3, ...) is non-empty even when blur_limit < 2.
        max_size = max(3, self.blur_limit * 2 + 1)
        size = random.choice(range(3, max_size + 1, 2))
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        angle_rad = random.uniform(0, np.pi)
        dx = int(round(center * np.cos(angle_rad)))
        dy = int(round(center * np.sin(angle_rad)))
        # cv2.line guarantees a non-empty line, preventing the zero-kernel black-image bug
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0  # identity fallback (should never trigger)
        blurred = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(blurred)


class AlbumentationsGridDistortion:
    """Grid distortion approximating A.GridDistortion."""
    def __init__(self, p: float = 0.5, num_steps: int = 4, distort_limit: float = 0.15):
        self.p = p
        self.num_steps = num_steps
        self.distort_limit = distort_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: AlbumentationsGridDistortion (steps={}, limit={})",
                     self.num_steps, self.distort_limit)
        img = img.convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        x_steps = [0] + sorted([random.uniform(0, 1) for _ in range(self.num_steps - 1)]) + [1]
        y_steps = [0] + sorted([random.uniform(0, 1) for _ in range(self.num_steps - 1)]) + [1]
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        # Loop over num_steps cells (not num_steps+1 — the +1 iteration always produced
        # an empty slice because x_steps[-1]==1.0 maps x_start==w with no pixels left)
        for i in range(self.num_steps):
            for j in range(self.num_steps):
                x_start = int(x_steps[j] * w)
                x_end   = int(x_steps[j + 1] * w)
                y_start = int(y_steps[i] * h)
                y_end   = int(y_steps[i + 1] * h)
                dx = random.uniform(-self.distort_limit, self.distort_limit) * w
                dy = random.uniform(-self.distort_limit, self.distort_limit) * h
                map_x[y_start:y_end, x_start:x_end] = np.clip(
                    np.arange(x_start, x_end)[None, :] + dx, 0, w - 1
                )
                map_y[y_start:y_end, x_start:x_end] = np.clip(
                    np.arange(y_start, y_end)[:, None] + dy, 0, h - 1
                )
        dst = cv2.remap(arr, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(dst)


class AlbumentationsOpticalDistortion:
    """Optical / barrel distortion approximating A.OpticalDistortion."""
    def __init__(self, p: float = 0.5, distort_limit: float = 0.03):
        self.p = p
        self.distort_limit = distort_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: AlbumentationsOpticalDistortion (distort_limit={})", self.distort_limit)
        img = img.convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        # Use independent k1/k2 so the distortion variety is not artificially constrained
        k1 = random.uniform(-self.distort_limit, self.distort_limit)
        k2 = random.uniform(-self.distort_limit * 0.5, self.distort_limit * 0.5)
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = w * 0.5
        cam[1, 2] = h * 0.5
        cam[0, 0] = w * 0.5
        cam[1, 1] = h * 0.5
        dist = np.array([[k1, k2, 0, 0, 0]], dtype=np.float32)
        dst = cv2.undistort(arr, cam, dist)
        return Image.fromarray(dst)


class AlbumentationsISONoise:
    """Camera-sensor / Poisson noise approximating A.ISONoise."""
    def __init__(self, p: float = 0.5, intensity: Tuple[float, float] = (0.05, 0.2)):
        self.p = p
        self.intensity = intensity

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        logger.info("apply: AlbumentationsISONoise (intensity={})", self.intensity)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        intensity = random.uniform(*self.intensity)
        noise = np.random.normal(1.0, intensity, arr.shape)
        arr = np.clip(arr * noise, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)


# ============================================================
# Mask augmentation classes (PIL-compatible, for CustomRandomApply)
# ============================================================

def _random_mask_coords(
    w: int, h: int, region_ratio: float, num_regions: int = 1,
) -> List[Tuple[int, int, int, int]]:
    """Return *num_regions* random bounding boxes, each sized
    ``region_ratio × image dimensions``, placed at uniformly random positions.

    When *num_regions* > 1 the boxes are independently drawn (they may overlap).
    """
    boxes = []
    box_w = max(1, int(w * region_ratio))
    box_h = max(1, int(h * region_ratio))
    for _ in range(num_regions):
        x1 = random.randint(0, max(0, w - box_w))
        y1 = random.randint(0, max(0, h - box_h))
        boxes.append((x1, y1, x1 + box_w, y1 + box_h))
    return boxes


def _contour_mask_coords(
    img_bgr: np.ndarray,
    region_ratio: float,
    num_regions: int = 1,
    min_area_fraction: float = 0.001,
) -> List[Tuple[int, int, int, int]]:
    """Find text-like contours via thresholding and return their bounding boxes.

    Steps:
      1. Convert to grayscale, Otsu threshold (dark text on light bg).
      2. Optional morphological close to merge fragmented strokes.
      3. Find external contours, filter by ``min_area_fraction × image_area``.
      4. Sort by area (descending), pick top-*num_regions*.
      5. Expand each box by ``region_ratio`` (padding around the contour).

    Falls back to :func:`_random_mask_coords` when no contours survive.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(1, int(h * w * min_area_fraction))
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        logger.debug("contour: no contours found ({}x{}), falling back to random", w, h)
        return _random_mask_coords(w, h, region_ratio, num_regions)
    contours.sort(key=cv2.contourArea, reverse=True)
    boxes = []
    pad_x = max(1, int(w * region_ratio * 0.5))
    pad_y = max(1, int(h * region_ratio * 0.5))
    for c in contours[:num_regions]:
        cx, cy, cw, ch_c = cv2.boundingRect(c)
        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + cw + pad_x)
        y2 = min(h, cy + ch_c + pad_y)
        boxes.append((x1, y1, x2, y2))
    logger.debug("contour: found {} regions from {} contours ({}x{})", len(boxes), len(contours), w, h)
    return boxes


def _generate_mask_coords(
    img_bgr_or_size,
    coord_mode: str = "random",
    region_ratio: float = 0.3,
    num_regions: int = 1,
) -> List[Tuple[int, int, int, int]]:
    """Dispatch to the appropriate coordinate generation strategy.

    Parameters
    ----------
    img_bgr_or_size :
        Either a numpy BGR image (for contour mode) or a ``(width, height)`` tuple.
    coord_mode :
        ``"random"`` — random sub-regions (default).
        ``"contour"`` — text-like contour bounding boxes.
    region_ratio :
        Size of each random box as a fraction of image dimensions.
        For contour mode this is the padding around each detected contour.
    num_regions :
        How many boxes to return.
    """
    if isinstance(img_bgr_or_size, tuple):
        w, h = img_bgr_or_size
        img_bgr = None
    else:
        img_bgr = img_bgr_or_size
        h, w = img_bgr.shape[:2]

    if coord_mode == "contour" and img_bgr is not None:
        coords = _contour_mask_coords(img_bgr, region_ratio, num_regions)
        logger.debug("generate_mask_coords: mode=contour, {} regions, size={}x{}", len(coords), w, h)
        return coords

    coords = _random_mask_coords(w, h, region_ratio, num_regions)
    logger.debug("generate_mask_coords: mode=random, {} regions, size={}x{}", len(coords), w, h)
    return coords


class MaskRandomBlock:
    """Random rectangular patch mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskRandomBlock (coord_mode={}, coords={})", self.coord_mode if self.coords is None else "explicit", coords)
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "random_block")
        return Image.fromarray(masked[:, :, ::-1])


class MaskRandomPixel:
    """Per-pixel random noise mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskRandomPixel (coord_mode={}, coords={})", self.coord_mode if self.coords is None else "explicit", coords)
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "random_pixel")
        return Image.fromarray(masked[:, :, ::-1])


class MaskGaussianNoise:
    """Additive Gaussian noise mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, std: float = 25.0, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.std = std
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskGaussianNoise (coord_mode={}, std={})", self.coord_mode if self.coords is None else "explicit", self.std)
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "gaussian_noise", gaussian_std=self.std)
        return Image.fromarray(masked[:, :, ::-1])


class MaskSaltPepper:
    """Salt-and-pepper noise mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskSaltPepper (coord_mode={}, coords={})", self.coord_mode if self.coords is None else "explicit", coords)
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "salt_pepper")
        return Image.fromarray(masked[:, :, ::-1])


class MaskBlur:
    """Gaussian blur mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, radius: int = None, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.radius = radius
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskBlur (coord_mode={}, radius={})", self.coord_mode if self.coords is None else "explicit", self.radius)
        kwargs = {}
        if self.radius is not None:
            kwargs["blur_radius"] = self.radius
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "blur", **kwargs)
        return Image.fromarray(masked[:, :, ::-1])


class MaskMosaic:
    """Mosaic / pixelation mask over a randomly generated image region."""
    def __init__(self, mask_ratio: float = 0.15, size: int = None, coords: Optional[List[Tuple[int,int,int,int]]] = None,
                 coord_mode: str = "random", region_ratio: Optional[float] = None,
                 num_regions: int = 1, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.size = size
        self.coords = coords
        self.coord_mode = coord_mode
        self.region_ratio = region_ratio
        self.num_regions = num_regions
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        img_bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.coords is not None:
            coords = self.coords
        else:
            effective_ratio = self.region_ratio if self.region_ratio is not None else self.mask_ratio
            coords = _generate_mask_coords(img_bgr, self.coord_mode, effective_ratio, self.num_regions)
        logger.info("apply: MaskMosaic (coord_mode={}, size={})", self.coord_mode if self.coords is None else "explicit", self.size)
        kwargs = {}
        if self.size is not None:
            kwargs["mosaic_size"] = self.size
        masked = apply_text_mask_to_region(img_bgr, coords, self.mask_ratio, "mosaic", **kwargs)
        return Image.fromarray(masked[:, :, ::-1])


# ============================================================
# Augmentation registry + config-driven pipeline builder
# ============================================================

#: Maps aug_config names → transform class.  All entries are treated
#: uniformly by ``_build_aug_transform`` — no separate dispatch path.
_AUG_REGISTRY: dict = {
    # region masks
    "mask_random_block":   MaskRandomBlock,
    "mask_random_pixel":   MaskRandomPixel,
    "mask_gaussian_noise": MaskGaussianNoise,
    "mask_salt_pepper":    MaskSaltPepper,
    "mask_blur":           MaskBlur,
    "mask_mosaic":         MaskMosaic,
    # text_renderer effects
    "curve":               Curve,
    "dropout_horizontal":  DropoutHorizontal,
    "dropout_vertical":    DropoutVertical,
    "dropout_rand":        DropoutRand,
    "image_padding":       ImagePadding,
    "line_overlay":        LineOverlay,
    "text_border":         TextBorder,
    # albumentations-style
    "motion_blur":         AlbumentationsMotionBlur,
    "grid_distortion":     AlbumentationsGridDistortion,
    "optical_distortion":  AlbumentationsOpticalDistortion,
    "iso_noise":           AlbumentationsISONoise,
    # whole-image transforms
    "gaussian_noise":      GaussianNoise,
    "gaussian_blur":       GaussianBlur,
    "jpeg_compression":    JpegCompression,
    "color_jitter":        ColorJitter,
    "random_scale":        RandomScale,
    "padding":             RandomSingleSidePadding,
    "random_rotation":     RandomDiscreteRotation,
    "rotation":            RandomDiscreteRotation,
}


class CustomRandomApply:
    """Randomly apply a list of transforms with probability *p*.

    This is a near-exact replica of ``paddleformers.datasets.template.augment_utils.RandomApply``,
    extended with shuffle order and error handling.

    Args:
        transforms: List of callables. When ``p`` fires, every transform in the
                   list is applied in sequence (matching the original behaviour).
        p:          Probability [0, 1] that the entire transform list is applied.
                    When the gate fires, all transforms run. When it misses, the
                    image is returned unchanged.
        shuffle:    Randomise the application order. Default ``False`` (same order
                    as the list, which matches the original ``RandomApply``).
        onerror:    ``"raise"`` (default) — re-raise exceptions. ``"skip"`` —
                    swallow exceptions and return the unchanged input.
        seed:       Optional int seed for a private ``random.Random`` instance.
                    Useful for reproducible tests.

    Examples
    --------
    # match paddleformers RandomApply exactly
    CustomRandomApply([blur, noise], p=0.5)

    # same, with shuffled order
    CustomRandomApply([blur, noise, jitter], p=0.5, shuffle=True)

    # skip on error instead of raising
    CustomRandomApply([unreliable_transform], p=0.3, onerror="skip")
    """

    def __init__(
        self,
        transforms: list,
        p: float = 0.5,
        shuffle: bool = False,
        onerror: str = "raise",
        seed: Optional[int] = None,
    ):
        if not transforms:
            raise ValueError("transforms must not be empty")
        if onerror not in ("raise", "skip"):
            raise ValueError(f"onerror must be 'raise' or 'skip', got {onerror!r}")

        self._fns: List = list(transforms)
        self.p: float = float(p)
        self._shuffle: bool = bool(shuffle)
        self._onerror: str = onerror
        self._rng = random.Random(seed)

    def __call__(self, img):
        if self._rng.random() >= self.p:
            return img

        fns = list(self._fns)
        if self._shuffle:
            self._rng.shuffle(fns)

        fn_names = [getattr(fn, "__name__", type(fn).__name__) for fn in fns]
        logger.info("CustomRandomApply: p={:.2f} fired, applying {}", self.p, fn_names)

        for fn in fns:
            try:
                img = fn(img)
            except Exception:
                if self._onerror == "raise":
                    raise
        return img

    def __repr__(self) -> str:
        names = [getattr(f, "__name__", type(f).__name__) for f in self._fns]
        return (
            f"CustomRandomApply(p={self.p}, shuffle={self._shuffle}, "
            f"transforms=[{', '.join(names)}])"
        )


def _build_aug_transform(name: str, params: dict):
    """Instantiate one augmentation transform from aug_config params.

    All parameters come directly from the YAML aug_config entry's ``params``
    block — no hard-coded defaults are merged in here.

    ``p`` is injected via :func:`_setdefault_p` for any class whose
    ``__init__`` accepts it, unless the caller already set it.

    Mask transforms (``mask_*``) generate their own bounding-box region at
    call time via ``_generate_mask_coords``.  ``coord_mode``, ``region_ratio``
    and ``num_regions`` can be set in the YAML ``params`` block to control
    region placement; they default to ``random`` / ``mask_ratio`` / ``1``.
    """
    key = name.lower()
    merged = dict(params)
    cls = _AUG_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown augmentation {name!r}. "
            f"Known names: {sorted(_AUG_REGISTRY)}"
        )
    _setdefault_p(cls, merged)
    logger.debug("build_aug: {} → {} (params={})", name, cls.__name__, merged)
    return cls(**merged)


def _setdefault_p(cls, params: dict, p: float = 1.0) -> None:
    """Inject ``params['p'] = p`` if the class accepts ``p`` and it is not already set.

    Defaults to ``1.0`` so that ``CustomRandomApply`` is the sole probability
    gate in classic mode.  In policy mode the YAML ``p`` is used as the
    eligibility gate and should be set explicitly in the config.
    """
    try:
        if "p" in inspect.signature(cls.__init__).parameters:
            params.setdefault("p", p)
    except (TypeError, ValueError):
        pass


def load_aug_config(config):
    """Normalise an augmentation config to a flat list of entry dicts.

    Accepts three forms:

    * **File path** (``str`` / ``pathlib.Path``) pointing to a YAML file whose
      top-level key is ``augmentations``::

          augmentations:
            - name: curve
              p: 0.3
              params:
                period: 180
                amplitude: [1, 5]
            - name: color_jitter
              p: 0.5
              params:
                brightness: 0.3
                contrast:   0.3
                saturation: 0.2
                hue:        0.1

    * **Dict** with an ``"augmentations"`` key (same structure as YAML root).
    * **List** of entry dicts directly (useful for programmatic construction).

    Each entry must have at minimum ``name`` (str) and ``p`` (float 0–1).
    ``params`` is an optional mapping of constructor keyword arguments.
    """
    if isinstance(config, (str, os.PathLike)):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load augmentation configs from a file. "
                "Install it with:  pip install pyyaml"
            ) from exc
        with open(config) as fh:
            logger.info("load_aug_config: loaded from file ({})", fh.name)
            config = yaml.safe_load(fh)
    if isinstance(config, dict):
        config = config.get("augmentations", [])
    if not isinstance(config, list):
        raise TypeError(
            f"Augmentation config must be a file path, a dict with key "
            f"'augmentations', or a list of entries; got {type(config).__name__}"
        )
    logger.debug("load_aug_config: {} entries loaded", len(config))
    return config


# ============================================================
# Policy-driven augmentation pipeline
# ============================================================

def load_aug_policy(config) -> Optional[dict]:
    """Extract the ``augmentation_policy`` section from a config source.

    Returns ``None`` when no policy is defined (falls back to the default
    per-entry ``p`` behaviour).
    """
    raw = None
    if isinstance(config, (str, os.PathLike)):
        try:
            import yaml
        except ImportError:
            return None
        with open(config) as fh:
            raw = yaml.safe_load(fh)
    elif isinstance(config, dict):
        raw = config

    if isinstance(raw, dict):
        policy = raw.get("augmentation_policy")
        if policy:
            logger.info("load_aug_policy: policy found (strategy={}, num_transforms={})",
                        policy.get("strategy", "uniform"), policy.get("num_transforms"))
        return policy
    logger.debug("load_aug_policy: no policy section found, using classic mode")
    return None


def _resolve_num_transforms(policy: dict) -> int:
    """Draw the number of transforms to apply from the policy."""
    nt = policy.get("num_transforms", 0)
    if isinstance(nt, (list, tuple)):
        lo, hi = int(nt[0]), int(nt[1])
        return random.randint(lo, hi)
    return int(nt)


def _build_category_index(entries: list) -> Dict[str, List[int]]:
    """Build a mapping ``category_name → [entry indices]``."""
    idx: Dict[str, List[int]] = {}
    for i, entry in enumerate(entries):
        cat = entry.get("category", "default")
        idx.setdefault(cat, []).append(i)
    return idx


def _sample_indices_weighted(
    entries: list,
    n: int,
    category_weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    """Sample *n* entry indices without replacement, respecting category weights.

    Two-phase sampling:
      1.  Allocate slots to categories proportional to their weights
          (multinomial draw).
      2.  Within each category, sample without replacement up to the
          allocated slot count.

    Entries whose eligibility gate ``p`` does not fire are excluded from
    the candidate pool *before* sampling.
    """
    cat_index = _build_category_index(entries)

    # Pre-filter: eligibility gate
    eligible: Dict[str, List[int]] = {}
    for cat, indices in cat_index.items():
        eligible[cat] = [i for i in indices if random.random() < float(entries[i].get("p", 0.5))]
    total_eligible = sum(len(v) for v in eligible.values())
    if total_eligible == 0:
        return []

    n = min(n, total_eligible)

    # No weights → uniform sample from all eligible
    if not category_weights:
        all_eligible = [i for indices in eligible.values() for i in indices]
        return random.sample(all_eligible, n)

    # Build per-entry sampling weights: category weight normalised by
    # the number of eligible entries in that category, so each *category*
    # hits its target proportion regardless of how many entries it has.
    entry_weights = []
    eligible_indices = []
    for cat, indices in eligible.items():
        if not indices:
            continue
        w = category_weights.get(cat, 0.0)
        per_entry_w = w / len(indices)
        for i in indices:
            eligible_indices.append(i)
            entry_weights.append(per_entry_w)
    if not eligible_indices:
        return []
    total_w = sum(entry_weights)
    if total_w == 0:
        return []
    entry_weights = [w / total_w for w in entry_weights]
    k = min(n, len(eligible_indices))
    chosen = [int(x) for x in np.random.choice(eligible_indices, size=k, replace=False, p=entry_weights)]
    random.shuffle(chosen)
    return chosen


class PolicyAugmentationPipeline:
    """Two-layer augmentation controller.

    **Layer 1 — policy gate**: decides *how many* transforms to apply
    and *which categories* get priority.  Controlled by the
    ``augmentation_policy`` section in the YAML config.

    **Layer 2 — individual ``p`` gate**: each entry still carries its
    own ``p`` field (the *eligibility* probability).  Only entries that
    pass their ``p`` gate become candidates for the policy sampler.

    When no ``augmentation_policy`` section exists in the config the
    pipeline degrades to the classic behaviour (each entry independently
    rolls its own ``p`` via ``CustomRandomApply``).

    YAML schema
    -----------

    .. code-block:: yaml

        augmentation_policy:
          num_transforms: [2, 4]          # exact int  or [min, max]
          strategy: weighted              # weighted | uniform
          category_weights:               # only when strategy == weighted
            region_mask: 0.15
            noise: 0.25
            distortion: 0.20
            degradation: 0.20
            geometric: 0.10
            padding: 0.10

        augmentations:
          - name: mask_random_block
            p: 0.5                       # eligibility gate — must pass to be candidate
            category: region_mask
            params:
              mask_ratio: 0.15
          - name: gaussian_noise
            p: 0.5
            category: noise
            params:
              std: 30.0
    """

    def __init__(
        self,
        entries: list,
        policy: dict,
    ):
        self._entries = entries
        self._policy = policy
        self._strategy = policy.get("strategy", "uniform")
        self._cat_weights = policy.get("category_weights") if self._strategy == "weighted" else None
        self._transform_cache: Dict[int, object] = {}
        for i, entry in enumerate(entries):
            name = entry.get("name", "")
            params = dict(entry.get("params") or {})
            try:
                self._transform_cache[i] = _build_aug_transform(name, params)
            except Exception as exc:
                logger.warning("policy_pipeline: could not build transform {}: {}", name, exc)

    def __call__(self, img: Image.Image) -> Image.Image:
        n = _resolve_num_transforms(self._policy)
        if n <= 0:
            return img

        indices = _sample_indices_weighted(
            self._entries, n, self._cat_weights
        )
        if not indices:
            logger.debug("policy_pipeline: no eligible transforms this call")
            return img

        names = [self._entries[i]["name"] for i in indices]
        logger.info("policy_pipeline: applying {} transforms: {}", len(indices), names)

        for i in indices:
            transform = self._transform_cache.get(i)
            if transform is None:
                continue
            try:
                img = transform(img)
            except Exception as exc:
                logger.warning("policy_pipeline: {} failed: {}", self._entries[i]["name"], exc)
        return img

    def __repr__(self) -> str:
        return (
            f"PolicyAugmentationPipeline("
            f"entries={len(self._entries)}, "
            f"strategy={self._strategy}, "
            f"num_transforms={self._policy.get('num_transforms')})"
        )


@dataclass
class PaddleOCRVLV15Plugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"
    aug_config: object = "configs/aug_config.yaml"

    def __init__(
        self,
        image_token,
        video_token,
        audio_token,
        aug_config="configs/aug_config.yaml",
        **kwargs,
    ):
        super().__init__(image_token, video_token, audio_token, **kwargs)
        self.aug_config = aug_config

        self._aug_entries = load_aug_config(aug_config) if aug_config is not None else None

        self._aug_policy = load_aug_policy(aug_config) if aug_config is not None else None

        if self._aug_policy is not None:
            self.image_augmentation = PolicyAugmentationPipeline(
                entries=self._aug_entries,
                policy=self._aug_policy,
            )
            logger.info("PaddleOCRVLV15Plugin: policy mode ({} entries, strategy={})",
                        len(self._aug_entries or []), self._aug_policy.get("strategy", "uniform"))
        else:
            self.image_augmentation = self.get_ocr_augmentations()
            logger.info("PaddleOCRVLV15Plugin: classic mode ({} entries)",
                        len(self._aug_entries or []))

    def get_ocr_augmentations(self):
        """Build a ``transforms.Compose`` pipeline from ``self._aug_entries``.

        Mask transforms generate their own random region from the image
        dimensions at call time — no external bounding boxes needed.
        """
        augmentations = []
        for entry in (self._aug_entries or []):
            name   = entry["name"]
            p      = float(entry.get("p", 0.5))
            params = dict(entry.get("params") or {})
            transform = _build_aug_transform(name, params)
            augmentations.append(CustomRandomApply([transform], p=p))
        return transforms.Compose(augmentations)

    @override
    def _preprocess_image(self, image, **kwargs):
        width, height = image.size
        image_max_pixels = kwargs["image_max_pixels"]
        image_min_pixels = kwargs["image_min_pixels"]
        image_processor = kwargs["image_processor"]

        resized_height, resized_width = image_processor.get_smarted_resize(
            height,
            width,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )[0]

        image = image.resize((resized_width, resized_height))

        logger.debug("preprocess_image: {}x{} → {}x{}, aug={}",
                     width, height, resized_width, resized_height,
                     type(self.image_augmentation).__name__ if self.image_augmentation else None)

        if self.image_augmentation is not None:
            image = self.image_augmentation(image)

        return image

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(image_processor, "max_pixels", 1003520),
                image_min_pixels=getattr(image_processor, "min_pixels", 112896),
                image_processor=image_processor,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)

        # here, we replace the IMAGE_PLACEHOLDER with the corresponding image tokens
        # you can customize the way of inserting image tokens as you like
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.image_bos_token}{self.image_token * image_seqlen}{self.image_eos_token}",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages

register_mm_plugin(
    name = "paddleocr_vl_v15",
    plugin_class = PaddleOCRVLV15Plugin,
)

# ==========================================
# Template
# ==========================================

register_template(
    name="paddleocr_vl_v15",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant:\n"]), # "/n" after "Assistant:"
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    mm_plugin=get_mm_plugin(name="paddleocr_vl_v15", image_token="<|IMAGE_PLACEHOLDER|>"),
)

# ============================================================
# Region masking (inlined from masked_region_aug.py)
# ============================================================
def _ensure_numpy(image):
    """Ensure image is a numpy ndarray in BGR format."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))[:, :, ::-1]
    if isinstance(image, np.ndarray):
        return image.copy()
    raise TypeError(f"Unsupported image type: {type(image)}")



def apply_text_mask_to_region(
    image,
    coords: List[Tuple[int, int, int, int]],
    mask_ratio: float = 0.15,
    mask_type: str = "random_block",
    **kwargs,
):
    """Apply a mask inside each bounding-box region."""
    img = _ensure_numpy(image)

    for coord in coords:
        x1, y1, x2, y2 = map(int, coord)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            continue

        roi = img[y1:y2, x1:x2]

        if mask_type == "random_block":
            roi = _mask_random_block(roi, mask_ratio)
        elif mask_type == "random_pixel":
            roi = _mask_random_pixel(roi, mask_ratio)
        elif mask_type == "gaussian_noise":
            std = kwargs.get("gaussian_std", 25.0)
            roi = _mask_gaussian_noise(roi, mask_ratio, std)
        elif mask_type == "salt_pepper":
            roi = _mask_salt_pepper(roi, mask_ratio)
        elif mask_type == "blur":
            radius = kwargs.get("blur_radius", max(1, int(min(h, w) * mask_ratio)))
            roi = _mask_blur(roi, radius)
        elif mask_type == "mosaic":
            size = kwargs.get("mosaic_size", max(2, int(min(h, w) * mask_ratio)))
            roi = _mask_mosaic(roi, size)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        img[y1:y2, x1:x2] = roi

    return img


def _mask_random_block(roi: np.ndarray, mask_ratio: float) -> np.ndarray:
    """Fill the ROI entirely with random per-pixel noise.

    The calling mask class already positioned and sized the ROI via
    ``_random_mask_coords`` — the ROI *is* the block, so there is no
    further sub-sampling needed here.  ``mask_ratio`` is accepted but
    unused (kept for a uniform function signature across all ``_mask_*``
    helpers).
    """
    h, w = roi.shape[:2]
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mask_random_pixel(roi: np.ndarray, mask_ratio: float) -> np.ndarray:
    """Randomly replace individual pixels with noise inside the ROI."""
    out = roi.copy()
    mask = np.random.rand(*roi.shape[:2]) < mask_ratio
    noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
    out[mask] = noise[mask]
    return out


def _mask_gaussian_noise(roi: np.ndarray, mask_ratio: float, std: float) -> np.ndarray:
    """Add Gaussian noise to the ROI."""
    if mask_ratio >= 1.0:
        noise = np.random.normal(0, std, roi.shape).astype(np.int16)
        return np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out = roi.copy()
    mask = np.random.rand(*roi.shape[:2]) < mask_ratio
    noise = np.random.normal(0, std, roi.shape).astype(np.int16)
    noisy = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out[mask] = noisy[mask]
    return out


def _mask_salt_pepper(roi: np.ndarray, mask_ratio: float) -> np.ndarray:
    """Apply salt-and-pepper noise inside the ROI."""
    out = roi.copy()
    h, w = roi.shape[:2]
    num_salt = int(np.ceil(mask_ratio * h * w * 0.5))
    num_pepper = int(np.ceil(mask_ratio * h * w * 0.5))
    if num_salt > 0:
        coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
        out[coords[0], coords[1], :] = 255
    if num_pepper > 0:
        coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
        out[coords[0], coords[1], :] = 0
    return out


def _mask_blur(roi: np.ndarray, radius: int) -> np.ndarray:
    """Apply Gaussian blur inside the ROI."""
    ksize = max(1, radius) * 2 + 1
    return cv2.GaussianBlur(roi, (ksize, ksize), 0)


def _mask_mosaic(roi: np.ndarray, size: int) -> np.ndarray:
    """Apply mosaic effect by downscaling then upscaling."""
    h, w = roi.shape[:2]
    size = max(1, min(size, h, w))
    small = cv2.resize(roi, (w // size, h // size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

