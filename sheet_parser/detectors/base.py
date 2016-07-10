import cv2


class BaseDetector(object):

    def __init__(self, src):
        self.src = src

        ret, self.dst = cv2.threshold(src, 180, 255, cv2.THRESH_BINARY_INV)
        self.output = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
