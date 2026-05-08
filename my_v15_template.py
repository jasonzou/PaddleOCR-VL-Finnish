# paddleocr_vl_v15_template.py
# ============================================================
# PaddleOCR-VL-1.5 custom template plugin with noise augmentation
# ============================================================
# Place this file in the same directory as your training configs.
# The config must contain: custom_register_path: ./paddleocr_vl_v15_template.py
#
# Noise augmentations (applied in addition to the built‑in ones):
#   - Gaussian noise      (NOISE_GAUSS_PROB, NOISE_GAUSS_STD)
#   - Salt-and-pepper     (NOISE_SP_PROB, NOISE_SP_AMOUNT, NOISE_SP_SALT_VS_PEPPER)
#   - Speckle noise       (NOISE_SPECKLE_PROB, NOISE_SPECKLE_STD)
#
# Set any probability to 0.0 to disable that noise type.
# Example env: export NOISE_GAUSS_PROB=0.4 NOISE_GAUSS_STD=8.0
# ============================================================

import os
import numpy as np
import random
from dataclasses import dataclass
from copy import deepcopy

import torchvision.transforms as transforms
from PIL import Image

from paddleformers.datasets.template.template import *
from paddleformers.datasets.template.mm_plugin import *
from paddleformers.datasets.template.augment_utils import *

# -------------------------------
# Noise configuration from ENV
# -------------------------------
def _env_float(key, default):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

# Gaussian
GAUSS_PROB = _env_float("NOISE_GAUSS_PROB", 0.3)
GAUSS_STD  = _env_float("NOISE_GAUSS_STD", 5.0)   # in pixel intensity [0,255]

# Salt & Pepper
SP_PROB           = _env_float("NOISE_SP_PROB", 0.2)
SP_AMOUNT         = _env_float("NOISE_SP_AMOUNT", 0.01)  # fraction of corrupted pixels
SP_SALT_VS_PEPPER = _env_float("NOISE_SP_SALT_VS_PEPPER", 0.5)

# Gaussian blur
BLUR_PROB         = _env_float("NOISE_BLUR_PROB", 0.3)
BLUR_RADIUS_MIN   = _env_float("NOISE_BLUR_RADIUS_MIN", 1.0)
BLUR_RADIUS_MAX   = _env_float("NOISE_BLUR_RADIUS_MAX", 3.0)

# Speckle
SPECKLE_PROB = _env_float("NOISE_SPECKLE_PROB", 0.2)
SPECKLE_STD  = _env_float("NOISE_SPECKLE_STD", 0.05)  # multiplicative std, 0-1 range

# -------------------------------
# Noise functions (PIL → numpy → PIL)
# -------------------------------
def _add_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def _add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def _add_salt_pepper(img: Image.Image, amount: float, salt_vs_pepper: float) -> Image.Image:
    arr = np.array(img.convert("RGB")).copy()
    # Salt
    num_salt = int(np.ceil(amount * arr.size * salt_vs_pepper))
    coords = [np.random.randint(0, i, num_salt) for i in arr.shape[:2]]
    arr[coords[0], coords[1], :] = 255
    # Pepper
    num_pepper = int(np.ceil(amount * arr.size * (1.0 - salt_vs_pepper)))
    coords = [np.random.randint(0, i, num_pepper) for i in arr.shape[:2]]
    arr[coords[0], coords[1], :] = 0
    return Image.fromarray(arr)

def _add_speckle_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    gauss = np.random.normal(1.0, std, arr.shape)
    arr = np.clip(arr * gauss, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

@dataclass
class PaddleOCRVLV15Plugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"

    def __init__(self, image_token, video_token, audio_token, **kwargs):
        super().__init__(image_token, video_token, audio_token, **kwargs)

        # Original augmentations (you may re‑enable them by setting probs)
        self.image_augmentation = self.get_ocr_augmentations(
            rotation_p=0.0,
            jpeg_p=0.0,
            scale_p=0.0,
            padding_p=0.0,
            color_jitter_p=0.0,
        )

    def get_ocr_augmentations(
        self,
        scale_range=(0.8, 1.2),
        scale_p=0.5,
        padding_range=(0, 15),
        padding_p=0.5,
        rotation_degrees=[0],
        rotation_p=0.5,
        color_jitter_p=0.5,
        jpeg_quality_range=(40, 90),
        jpeg_p=0.5,
    ):
        augmentations = []

        if scale_p > 0:
            scale_transform = RandomScale(scale_range=scale_range)
            augmentations.append(RandomApply([scale_transform], p=scale_p))

        if padding_p > 0:
            padding_transform = RandomSingleSidePadding(padding_range=padding_range, fill="white")
            augmentations.append(RandomApply([padding_transform], p=padding_p))

        if rotation_p > 0 and rotation_degrees:
            rotation_transform = RandomDiscreteRotation(
                degrees=rotation_degrees, interpolation="nearest", expand=True
            )
            augmentations.append(RandomApply([rotation_transform], p=rotation_p))

        if color_jitter_p > 0:
            color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            )
            augmentations.append(RandomApply([color_jitter], p=color_jitter_p))

        if jpeg_p > 0:
            jpeg_transform = JpegCompression(quality_range=jpeg_quality_range)
            augmentations.append(RandomApply([jpeg_transform], p=jpeg_p))

        return transforms.Compose(augmentations)

    # Override _preprocess_image to add noise after the standard augmentation
    @override
    def _preprocess_image(self, image, **kwargs):
        width, height = image.size
        image_max_pixels = kwargs["image_max_pixels"]
        image_min_pixels = kwargs["image_min_pixels"]
        image_processor = kwargs["image_processor"]

        # pre-resize before augmentation
        resized_height, resized_width = image_processor.get_smarted_resize(
            height,
            width,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )[0]
        image = image.resize((resized_width, resized_height))

        # 1. Standard augmentations (if any)
        if hasattr(self, "image_augmentation") and self.image_augmentation:
            image = self.image_augmentation(image)

        # 2. Custom noise augmentations
        image = self._apply_noise_augmentations(image)

        return image

    def _apply_noise_augmentations(self, image: Image.Image) -> Image.Image:
        if random.random() < GAUSS_PROB:
            image = _add_gaussian_noise(image, GAUSS_STD)

        if random.random() < BLUR_PROB:
            radius = random.uniform(BLUR_RADIUS_MIN, BLUR_RADIUS_MAX)
            image = _add_gaussian_blur(image, radius)

        if random.random() < SP_PROB:
            image = _add_salt_pepper(image, SP_AMOUNT, SP_SALT_VS_PEPPER)

        if random.random() < SPECKLE_PROB:
            image = _add_speckle_noise(image, SPECKLE_STD)

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

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length
                    if self.expand_mm_tokens
                    else 1
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
    name="paddleocr_vl_v15",
    plugin_class=PaddleOCRVLV15Plugin,
)

register_template(
    name="paddleocr_vl_v15",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant:\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    mm_plugin=get_mm_plugin(name="paddleocr_vl_v15", image_token="<|IMAGE_PLACEHOLDER|>"),
)

def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


_train_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
])


def process_fn(examples: dict, image_root: str) -> dict:
    """Dataset map function: load images, apply augmentations, return pixel_values."""
    images = []
    for img_path in examples["images"]:
        full_path = os.path.join(image_root, img_path) if not os.path.isabs(img_path) else img_path
        image = _load_image(full_path)
        image = _train_augment(image)
        plugin = PaddleOCRVLV15Plugin.__new__(PaddleOCRVLV15Plugin)
        image = plugin._apply_noise_augmentations(image)
        images.append(image)
    return {"pixel_values": images}


"""
export NOISE_GAUSS_PROB=0.4 NOISE_GAUSS_STD=8.0
export NOISE_BLUR_PROB=0.3 NOISE_BLUR_RADIUS_MIN=1.0 NOISE_BLUR_RADIUS_MAX=3.0
export NOISE_SP_PROB=0.2 NOISE_SP_AMOUNT=0.05
export NOISE_SPECKLE_PROB=0.3 NOISE_SPECKLE_STD=0.05
paddleformers-cli train ...
"""
