"""Video preprocessing: ndarray → PIL → PNG/JPEG → base64."""

import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("backend.llm.video")


def encode_image_to_base64(image_data: np.ndarray | bytes | Image.Image, image_format: str = "jpeg") -> str:
    """Convert image to base64-encoded string.

    Mirrors MiniCpmModel._encode_image_to_base64() in model_call.py.
    Supports numpy arrays, raw bytes, and PIL Images.
    """
    format_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}
    pil_format = format_map.get(image_format.lower(), image_format.upper())

    if isinstance(image_data, np.ndarray):
        if image_data.dtype != np.uint8:
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
        if len(image_data.shape) == 3:
            image = Image.fromarray(image_data)
        else:
            image = Image.fromarray(image_data, mode="L")
        img_buf = io.BytesIO()
        image.save(img_buf, format=pil_format)
        image_bytes = img_buf.getvalue()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif hasattr(image_data, "save"):
        img_buf = io.BytesIO()
        image_data.save(img_buf, format=pil_format)
        image_bytes = img_buf.getvalue()
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")

    return base64.b64encode(image_bytes).decode("utf-8")
