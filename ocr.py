from PIL import Image
import pytesseract
import cv2
import numpy as np

class OCR:
    def __init__(self):        
        print("----- Tesseract Version -----")
        print(pytesseract.get_tesseract_version())
        print("-----------------------------")
        self.img = cv2.imread('image.jpg')

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image,5)
     
    #thresholding
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
        
    #erosion
    def erode(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    def run(self):
        img = cv2.imread('testocr.png')

        gray = self.get_grayscale(img)
        gray = self.remove_noise(gray)

        custom_config = r'--oem 3 --psm 6'        
        return pytesseract.image_to_string(gray, config=custom_config)