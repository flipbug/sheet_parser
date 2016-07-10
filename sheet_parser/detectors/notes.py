import cv2
import numpy as np
import math


class NoteDetector(object):
    line_color = (0, 100, 255)
    lines = list()

    def __init__(self, src, threshold=50, min_line_width=100):
        self.src = src
        self.threshold = threshold
        self.min_line_width = min_line_width

        ret, self.dst = cv2.threshold(src, 180, 255, cv2.THRESH_BINARY_INV)
        self.output = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    def process(self):
        self.lines = self.detect_lines()
        print('lines: {}'.format(len(self.lines)))
        image = self.draw_lines()
        return image

    def detect_lines(self):
        result = list()
        tmp = list()

        # find lines
        lines = cv2.HoughLinesP(self.dst, 1, math.pi / 180.0, self.threshold, np.array([]), 20, 10)
        a, b, c = lines.shape
        for i in range(a):
            start = (lines[i][0][0], lines[i][0][1])
            end = (lines[i][0][2], lines[i][0][3])
            # check if horizontal and not too short
            if -0.2 < self.get_angle(start, end) < 0.2 and self.calc_min_width(start[0], end[0]):
                tmp.append(lines[i])

        # sort lines by y-axis
        tmp.sort(key=lambda x: x[0][1])

        # add lines which are not too close to each other
        for index, line in enumerate(tmp):
            if index > 0:
                if line[0][1] - tmp[index - 1][0][1] > 1:
                    result.append(line)
            else:
                result.append(line)

        return result

    def draw_lines(self):
        img = cv2.cvtColor(self.src, cv2.COLOR_GRAY2BGR)
        for line in self.lines:
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
