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

    def run(self, file_name: str, oem: int, psm: int):
        img = cv2.imread(f'{file_name}')

        gray = self.get_grayscale(img)
        gray = self.remove_noise(gray)

        # oem
        # 0    Legacy engine only.
        # 1    Neural nets LSTM engine only.
        # 2    Legacy + LSTM engines.
        # 3    Default, based on what is available.

        # psm
        # 0    Orientation and script detection (OSD) only.s
        # 1    Automatic page segmentation with OSD.
        # 2    Automatic page segmentation, but no OSD, or OCR.
        # 3    Fully automatic page segmentation, but no OSD. (Default)
        # 4    Assume a single column of text of variable sizes.
        # 5    Assume a single uniform block of vertically aligned text.
        # 6    Assume a single uniform block of text.
        # 7    Treat the image as a single text line.
        # 8    Treat the image as a single word.
        # 9    Treat the image as a single word in a circle.
        # 10    Treat the image as a single character.
        # 11    Sparse text. Find as much text as possible in no particular order.
        # 12    Sparse text with OSD.
        # 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

        custom_config = r'--oem {} --psm {}'.format(oem,psm)    

        return pytesseract.image_to_string(gray, config=custom_config)
