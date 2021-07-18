import cv2
import numpy as np

class LightDetector():
    def __init__(self):
        self.center = None
        self.box = None
    
    
    def set_img(self, frame):
        self.image = frame.copy()
        self.get_mask()
        

    
    def get_mask(self):
        b, g, r = cv2.split(self.image)
#         single = 2 * g - b - r
        ret, thresh = cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        erode = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel=(3,3), iterations=1)
        self.mask = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel=(3,3), iterations=2)
        cv2.imshow('mask', self.mask)
        
        
    def get_center(self):
        max_area = 0
        flag = 0
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not area < 20:
                light_rect = cv2.minAreaRect(contour)
                ratio = light_rect[1][0]/light_rect[1][1]
                if (ratio <= 2) and (ratio >= 0.5):
                    if area > max_area:
                        max_area = area
                        self.center = light_rect[0][0], light_rect[0][1]
                        self.box = np.int0(cv2.boxPoints(light_rect))
                        flag = 1
        return flag