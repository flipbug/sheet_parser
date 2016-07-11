#!/usr/bin/env python3

import cv2
import os

from sheet_parser.parser import Parser
test_image = os.path.abspath('tests/images/score01.png')
# test_image = os.path.abspath('training/images/training_set_001.png')

parser = Parser()
parser.process(test_image)

cv2.waitKey(0)