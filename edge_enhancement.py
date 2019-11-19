import cv2

def enhance_edge(image) :
    image = cv2.bilateralFilter(image, 2, 100, 100)
    image = cv2.convertScaleAbs(image, alpha=3, beta=0)
    image = cv2.Canny(image, 0, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(image, kernel)
    return cv2.erode(dilated,kernel)