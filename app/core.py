import logging
import config

from app.image_utils import decode_base64_to_image

logging.basicConfig(level=config.LOGLEVEL)
logger = logging.getLogger(__name__)


def convert_image_to_text(payload: dict) -> dict:
    image = decode_base64_to_image(payload["image"])
    return "Hello API!"
