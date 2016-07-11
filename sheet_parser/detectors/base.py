import cv2


class BaseDetector(object):
    dst = None
    output = None
    src = None
    gray = None

    def set_source(self, src):
        self.src = src
        self.gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, self.dst = cv2.threshold(self.gray, 180, 255, cv2.THRESH_BINARY_INV)
