import cv2
import numpy as np
import math
from .base import BaseDetector


class ShapeDetector(BaseDetector):

    contours = None

    def set_source(self, src):
        super().set_source(src)
        blur = cv2.GaussianBlur(self.dst, (5, 5), 0)
        self.img = self.remove_lines_from_image(src)
        cv2.imshow('cleared image', self.img)

    def find_shapes(self, src):
        self.set_source(src)
        img, self.contours, hierarchy = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def remove_lines_from_image(self, src):
        gray = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        # invert gray image
        dst = (255 - gray)
        img = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15, dst=dst)

        (rows, cols) = img.shape
        vertical_size = int(rows / 30)

        structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        cv2.morphologyEx(img, cv2.MORPH_OPEN, structure, dst=img, iterations=1)

        return img

    def draw(self, img=None):
        # draw bounding boxes of found shapes
        for contour in self.contours:
            area = cv2.contourArea(contour)
            if 5 < area < 1000:
                [x, y, w, h] = cv2.boundingRect(contour)
                if h > 5:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        return img
