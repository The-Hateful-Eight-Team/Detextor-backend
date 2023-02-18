import base64

import cv2
import numpy as np


def decode_base64_to_image(image: str) -> np.ndarray:
    np_arr = np.frombuffer(base64.b64decode(image), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def encode_image_to_base64(image: np.ndarray) -> str:
    _, im_arr = cv2.imencode(".jpg", image)
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode("utf-8")
