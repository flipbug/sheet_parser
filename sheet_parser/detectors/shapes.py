import cv2
import numpy as np
import os
import random

from .base import BaseDetector


class ShapeDetector(BaseDetector):

    contours = []
    img = None
    shapes = []
    staff_lines = None

    symbols = {
        'quarter': ['../images/note_02.png'],
        'half': ['../images/note_03.png'],
        'full': ['../images/note_04.png'],
    }

    def find_shapes(self, src, staff_lines):
        self.staff_lines = staff_lines
        self.set_source(src)

        img_rgb = self.src
        clean_img = self.remove_lines_from_image(self.gray)

        for key, templates in self.symbols.items():
            result = self.match_templates(clean_img, templates)

            color = (0, random.randint(0, 255), random.randint(0, 255))
            for pt in zip(*result[0][0][::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + result[0][1], pt[1] + result[0][2]), color, 1)

        cv2.imshow('cleared imageerer', self.src)

        return self.shapes

    def match_templates(self, src, template_imgs):
        dir = os.path.dirname(__file__)
        result = []

        for template_img in template_imgs:
            template = cv2.imread(os.path.join(dir, template_img), 0)

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.75
            loc = np.where(res >= threshold)

            result.append([loc, w, h])

        # TODO clean up result!!
        return result

    def set_source(self, src):
        super().set_source(src)
        self.img = self.remove_lines_from_image(self.gray)
        cv2.imshow('cleared image', self.img)

    def remove_lines_from_image(self, src):
        img = src
        for line in self.staff_lines:
            x = line[0][0]
            y = line[0][1]

            while x < line[0][2]:
                color = self.get_color(img, x, y - 1)
                cv2.rectangle(img, (x, y), (x, y), color, thickness=cv2.FILLED)
                x += 1

        ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img

    def get_color(self, img, x, y):
        intensity = int(img[y, x])  # yes, x and y coordinates are the other way around
        if intensity > 180:
            return 255, 255, 255
        return intensity, intensity, intensity

    def draw(self, img=None):
        # draw bounding boxes of found shapes
        for shape in self.shapes:
            [x, y, w, h] = cv2.boundingRect(shape['contour'])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(img, shape['label'],  (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return img
