import cv2
import numpy as np


def preprocess(img):
    gray_image = to_greyscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    no_noise = remove_noise(im_bw)
    no_borders = remove_borders(no_noise)
    color = [255, 255, 255]
    top, bottom, left, right = [150] * 4
    image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_with_border


def to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    """
    Dilation с последующим расширением+ медианный фильтр
    Закрытие небольших отверстий внутри объектов
    переднего плана или небольших черных точек на объекте.
    :param image: входное изображение
    :return:
    """
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def remove_borders(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    largest_contour = contours_sorted[-1]
    x, y, w, h = cv2.boundingRect(largest_contour)
    crop = image[y:y + h, x:x + w]
    return crop


def get_skew_angle(cvImage) -> float:
    newImage = cvImage.copy()
    blur = cv2.GaussianBlur(cvImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    largest_contour = contours[0]
    min_area_rectangle = cv2.minAreaRect(largest_contour)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = min_area_rectangle[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotate_image(cv_image, angle: float):
    new_image = cv_image.copy()
    (h, w) = new_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return new_image
