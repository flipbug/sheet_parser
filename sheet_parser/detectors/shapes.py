import cv2
import numpy as np
import math
import os

from .base import BaseDetector


class ShapeDetector(BaseDetector):

    contours = []
    img = None
    shapes = []

    def find_shapes(self, src):
        contours = self.find_contours(src)

        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        model = self.setup_model()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresh[y: y + h, x: x + w]

            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)

            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)

            self.shapes.append({
                'label': chr(results[0][0]),
                'contour': contour,
                'x': x,
                'y': y
            })

        return self.shapes

    def find_contours(self, src):
        self.set_source(src)
        img, contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area:
                x, y, w, h = cv2.boundingRect(contour)
                if h > 5 and w > 5:
                    self.contours.append(contour)

        return self.contours

    def setup_model(self):
        dir = os.path.dirname(__file__)

        # load training set
        samples = np.loadtxt(os.path.join(dir, '../../training/data/samples.data'), np.float32)
        responses = np.loadtxt(os.path.join(dir, '../../training/data/responses.data'), np.float32)
        responses = responses.reshape((responses.size, 1))

        # train model on set
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)

        return model

    def set_source(self, src):
        super().set_source(src)
        self.img = self.remove_lines_from_image(self.gray)
        cv2.imshow('cleared image', self.img)

    def remove_lines_from_image(self, src):
        # first attempt using morphology
        # img = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)
        # vertical_size = 3
        # structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        # cv2.morphologyEx(img, cv2.MORPH_OPEN, structure, dst=img, iterations=1)

        # second attempt using blur
        blur = cv2.GaussianBlur(src, (3, 3), 3)
        ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return img

    def draw(self, img=None):
        # draw bounding boxes of found shapes
        for shape in self.shapes:
            [x, y, w, h] = cv2.boundingRect(shape['contour'])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(img, shape['label'],  (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return img
