import cv2

def nothing(x):
    pass

def enhance_edge(image) :
    image = cv2.bilateralFilter(image, 2, 100, 100)
    image = cv2.convertScaleAbs(image, alpha=3, beta=0)
    image = cv2.Canny(image, 0, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(image, kernel)
    return cv2.erode(dilated,kernel)

'''def enhance_edge_var(image, threshhold1, threshhold2, d, sigmaColor, sigmaSpace):
    image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    image = cv2.convertScaleAbs(image, alpha=3, beta=0)
    image = cv2.Canny(image, threshhold1, threshhold2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(image, kernel)
    return cv2.erode(dilated,kernel)
    return image

if __name__ == '__main__':
    cv2.namedWindow('image')
    cv2.createTrackbar('threshhold1', 'image', 0, 700, nothing)
    cv2.createTrackbar('threshhold2', 'image', 0, 700, nothing)

    cv2.createTrackbar('d', 'image', 1, 30, nothing)
    cv2.createTrackbar('sigmaColor', 'image', 0, 400, nothing)
    cv2.createTrackbar('sigmaSpace', 'image', 0, 400, nothing)

    img = cv2.imread('samples/t1.jpg')
    while(1):
        threshhold1 = cv2.getTrackbarPos('threshhold1', 'image')
        threshhold2 = cv2.getTrackbarPos('threshhold2', 'image')

        d = cv2.getTrackbarPos('d', 'image')
        sigmaColor = cv2.getTrackbarPos('sigmaColor', 'image')
        sigmaSpace = cv2.getTrackbarPos('sigmaSpace', 'image')

        cv2.imshow('image', enhance_edge_var(img, threshhold1, threshhold2, d, sigmaColor, sigmaSpace))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()'''