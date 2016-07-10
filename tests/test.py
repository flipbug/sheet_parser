#!/usr/bin/env python3

import cv2

from sheet_parser.detectors.lines import LineDetector

test_image = cv2.imread('/Users/flipbug/Development/Python/sheet_parser/tests/images/score01.png', 0)
cv2.imshow("dst", test_image)

result_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
detector = LineDetector(test_image)

y_diff = detector.get_y_diff()
result_image = detector.draw(result_image)
print('y diff: {}'.format(detector.get_y_diff()))

cv2.imshow("detected lines", result_image)
cv2.waitKey(0)