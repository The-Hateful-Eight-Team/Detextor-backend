import logging
import config

from app.core.base64image import decode_base64_to_image
from app.core.recognition import recognise_text

logging.basicConfig(level=config.LOGLEVEL)
log = logging.getLogger(__name__)


def convert_image_to_text(payload: dict) -> dict:
    log.info(f"Image size: payload['image_size']")
    image = decode_base64_to_image(payload["image"])
    text = recognise_text(image)
    return text
