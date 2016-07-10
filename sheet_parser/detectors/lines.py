import cv2
import numpy as np
import math
from .base import BaseDetector


class LineDetector(BaseDetector):
    line_color = (0, 100, 255)
    lines = list()

    X = 0
    Y = 1

    def __init__(self, threshold=50, min_line_width=100):
        self.threshold = threshold
        self.min_line_width = min_line_width

    def find_lines(self, src):
        self.set_source(src)
        h_lines, v_lines = self.detect()
        # return staff_lines, bar_lines
        return self.remove_duplicates(h_lines, self.Y), self.remove_duplicates(v_lines, self.X)

    def detect(self):
        horizontal_lines = list()
        vertical_lines = list()

        # find lines
        lines = cv2.HoughLinesP(self.dst, 1, math.pi / 180.0, self.threshold, np.array([]), 20, 10)
        a, b, c = lines.shape
        for i in range(a):
            start = (lines[i][0][0], lines[i][0][1])
            end = (lines[i][0][2], lines[i][0][3])

            # check if horizontal and not too short
            if -0.2 < self.get_angle(start, end) < 0.2 and self.calc_min_width(start[0], end[0]):
                horizontal_lines.append(lines[i])

            # check if vertical and not too short
            if 89.8 < self.get_angle(start, end) < 90.2 and self.calc_min_width(start[0], end[0]):
                vertical_lines.append(lines[i])

        # sort horizontal lines by y-axis
        horizontal_lines.sort(key=lambda x: x[0][self.Y])

        # sort vertical lines by x-axis
        vertical_lines.sort(key=lambda x: x[0][self.X])

        return horizontal_lines, vertical_lines

    def remove_duplicates(self, lines, axis):
        result = list()
        for index, line in enumerate(lines):
            if index > 0:
                if line[0][axis] - lines[index - 1][0][axis] > 1:
                    result.append(line)
            else:
                result.append(line)
        return result

    def draw(self, lines, img=None):
        if img is None:
            img = cv2.cvtColor(self.src, cv2.COLOR_GRAY2BGR)
        for line in lines:
            start = (line[0][0], line[0][1])
            end = (line[0][2], line[0][3])
            cv2.line(img, start, end, self.line_color, 1, cv2.LINE_AA)
        return img

    def get_angle(self, pt1, pt2):
        return np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180. / np.pi

    def calc_min_width(self, x1, x2):
        return x2 - x1 > self.min_line_width

    def get_y_diff(self):
        if len(self.lines) > 1:
            return self.lines[1][0][1] - self.lines[0][0][1]
        return 0
