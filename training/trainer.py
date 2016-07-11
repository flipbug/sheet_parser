#!/usr/bin/python3

import cv2
import numpy as np
import os

img_path = os.path.abspath('training/images/training_set_001.png')
img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thres = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

im, contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100))
responses = []

for cnt in contours:
    if cv2.contourArea(cnt) > 10:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thres[y: y + h, x: x + w]
            roismall = cv2.resize(roi, (10, 10))

            cv2.imshow('norm', img)
            print('Enter the symbol name (leave empty to skip symbol):')

            name = ''
            while True:
                key = cv2.waitKey(0)
                print(chr(key), end="")
                if key == 13:  # Enter Key
                    break
                else:
                    name += chr(key)

            print('')
            print('----')

            if name is not '':
                responses.append(name)
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)

# responses = np.array(responses)
# responses = responses.reshape((responses.size, 1))
# np.savetxt(os.path.abspath('training/data/responses.data'), responses)

np.savetxt(os.path.abspath('training/data/samples.data'), samples)
with open(os.path.abspath('training/data/responses.data'), "w") as text_file:
    print(";".join(responses), file=text_file)

print('Training complete')