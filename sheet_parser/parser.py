import cv2
from sheet_parser.detectors.lines import LineDetector
from sheet_parser.detectors.shapes import ShapeDetector


class Parser(object):

    def __init__(self):
        self.line_detector = LineDetector()
        self.shape_detector = ShapeDetector()

    def process(self, source_image):
        img = cv2.imread(source_image)
        result = img

        staff_lines, bar_lines = self.line_detector.find_lines(img)
        shapes = self.shape_detector.find_shapes(img)

        result = self.line_detector.draw(staff_lines, result)
        result = self.line_detector.draw(bar_lines, result)
        result = self.shape_detector.draw(result)

        cv2.imshow('Resulting image', result)
