"""
EcoSort AI - Image Preprocessing Pipeline
Handles image validation, color space normalization, aspect-ratio handling,
resizing to (224, 224), and MobileNetV2 tensor preprocessing.
"""

import io
from pathlib import Path
from PIL import Image
import numpy as np

# Standard input dimensions for MobileNetV2
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_FORMATS = {'JPEG', 'JPG', 'PNG', 'WEBP'}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def is_valid_image(image_source) -> tuple[bool, str]:
    """
    Validates if the provided image source is a readable and non-corrupted image file.
    
    :param image_source: File path (str/Path), bytes, or file-like object.
    :return: (is_valid: bool, error_message: str or "")
    """
    try:
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                return False, f"Image file not found: {image_source}"
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                return False, "Image file exceeds the 5MB size limit."
            with Image.open(path) as img:
                img.verify()
                fmt = (img.format or '').upper()
                if fmt not in ALLOWED_FORMATS:
                    return False, f"Unsupported image format: {fmt}. Allowed: JPEG, PNG, WEBP."

        elif hasattr(image_source, 'read'):
            # File-like object (e.g., Flask request.files['image'])
            image_source.seek(0, io.SEEK_END)
            size = image_source.tell()
            image_source.seek(0)

            if size > MAX_FILE_SIZE_BYTES:
                return False, "Uploaded image exceeds the 5MB size limit."
            if size == 0:
                return False, "Uploaded file is empty."

            data = image_source.read()
            image_source.seek(0)
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
                fmt = (img.format or '').upper()
                if fmt not in ALLOWED_FORMATS:
                    return False, f"Unsupported image format: {fmt}. Allowed: JPEG, PNG, WEBP."

        elif isinstance(image_source, bytes):
            if len(image_source) > MAX_FILE_SIZE_BYTES:
                return False, "Image bytes exceed 5MB limit."
            if len(image_source) == 0:
                return False, "Image bytes are empty."
            with Image.open(io.BytesIO(image_source)) as img:
                img.verify()
                fmt = (img.format or '').upper()
                if fmt not in ALLOWED_FORMATS:
                    return False, f"Unsupported image format: {fmt}. Allowed: JPEG, PNG, WEBP."

        elif isinstance(image_source, Image.Image):
            return True, ""

        else:
            return False, "Unsupported image source type."

        return True, ""

    except Exception as e:
        return False, f"Invalid or corrupted image: {str(e)}"


def preprocess_image(image_source, target_size=TARGET_IMAGE_SIZE) -> tuple[np.ndarray | None, str | None]:
    """
    Loads, cleans, resizes, and normalizes an image for MobileNetV2 inference.
    
    Processing Steps:
    1. Loads image from path, stream, bytes, or PIL Image.
    2. Converts color mode to standard RGB (converts RGBA transparency and Grayscale).
    3. Resizes to target dimensions (224, 224) using high-quality resampling.
    4. Converts to float32 NumPy array.
    5. Normalizes pixel values to [-1.0, 1.0] using MobileNetV2 standard: (x / 127.5) - 1.0.
    6. Expands batch dimension to output shape: (1, 224, 224, 3).
    
    :param image_source: File path, file-like object, bytes, or PIL Image.
    :param target_size: Tuple (width, height), defaults to (224, 224).
    :return: (tensor_array: np.ndarray, None) on success, or (None, error_message: str) on failure.
    """
    try:
        # Load into PIL Image
        if isinstance(image_source, Image.Image):
            img = image_source
        elif isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif hasattr(image_source, 'read'):
            # Read from stream
            image_source.seek(0)
            data = image_source.read()
            image_source.seek(0)
            img = Image.open(io.BytesIO(data))
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        else:
            return None, "Invalid image source provided."

        # Convert to RGB mode (safely handles RGBA transparency, palettes, grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to target size with Lanczos resampling
        resample_filter = getattr(Image.Resampling, 'LANCZOS', getattr(Image, 'LANCZOS', None))
        if resample_filter is not None:
            img = img.resize(target_size, resample=resample_filter)
        else:
            img = img.resize(target_size)

        # Convert to NumPy float32 array
        img_array = np.asarray(img, dtype=np.float32)

        # MobileNetV2 normalization: scales [0, 255] -> [-1.0, 1.0]
        # Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input
        normalized_array = (img_array / 127.5) - 1.0

        # Expand batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        batch_tensor = np.expand_dims(normalized_array, axis=0)

        return batch_tensor, None

    except Exception as e:
        return None, f"Image preprocessing failed: {str(e)}"
