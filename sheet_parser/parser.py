#!/usr/bin/env python3

import cv2

from sheet_parser.detectors.lines import LineDetector

test_image = cv2.imread('../../tests/images/score01.png', 0)
detector = LineDetector(test_image)

result_image = detector.process()
print('y diff: {}'.format(detector.get_y_diff()))

cv2.imshow("dst", detector.dst)
cv2.imshow("detected lines", result_image)
cv2.waitKey(0)