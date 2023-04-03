# -*- coding: utf-8 -*-
import logging

import pytesseract
import cv2
import imutils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.core.image_transform import preprocess, get_skew_angle, rotate_image

logger = logging.getLogger(__name__)


def recognise_text(image):
    preprocessed_image = preprocess(image)
    recognized_handwritten_text = recognise_microsost(preprocessed_image)
    recognized_printed_text = recognise_tesseract(preprocessed_image)
    logger.info(recognized_printed_text)
    logger.info(recognized_handwritten_text)
    if len(recognized_printed_text) > len(recognized_handwritten_text):
        return recognized_printed_text
    return recognized_handwritten_text


def recognise_microsost(image_cv2):
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    pixel_values = processor(images=image_cv2, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def recognise_tesseract(image_cv2):
    image = image_cv2
    angle = get_skew_angle(image)
    if angle != -90:
        image = rotate_image(image, -1.0 * angle)
    try:
        results = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        image = imutils.rotate_bound(image, angle=results["rotate"])
    except Exception as e:
        print("too few characters", e)

    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, lang='eng', config=custom_config)


