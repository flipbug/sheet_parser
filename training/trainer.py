#!/usr/bin/python3

import cv2
import numpy as np
import os


class Trainer(object):

    responses = []
    samples = []

    def __init__(self):
        self.samples = np.empty((0, 100))

    def train(self, img_paths):
        for img_path in img_paths:
            self.process_image(img_path)
        self.save()

    def process_image(self, img_path):
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if 10 < cv2.contourArea(cnt):
                [x, y, w, h] = cv2.boundingRect(cnt)

                if h > 10:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi = thresh[y: y + h, x: x + w]
                    roismall = cv2.resize(roi, (10, 10))

                    self.read_samples(img, roismall)

    def read_samples(self, img, roismall):
        cv2.imshow('norm', img)
        print('Enter the symbol name (leave empty to skip symbol):')

        name = ''
        while True:
            key = cv2.waitKey(0)
            print(chr(key), end="")
            if key == 13:  # Enter Key
                break
            else:
                name += str(key)

        print('')
        print('----')

        if name is not '':
            self.responses.append(int(name))
            sample = roismall.reshape((1, 100))
            self.samples = np.append(self.samples, sample, 0)

    def save(self):
        self.responses = np.array(self.responses)
        self.responses = self.responses.reshape((self.responses.size, 1))

        np.savetxt(os.path.abspath('training/data/responses.data'), self.responses)
        np.savetxt(os.path.abspath('training/data/samples.data'), self.samples)


images = [
    os.path.abspath('training/images/test2.jpg'),
    os.path.abspath('training/images/test3.jpg'),
    os.path.abspath('training/images/test4.jpg'),
    os.path.abspath('training/images/training_set_001.png'),
]

trainer = Trainer()
trainer.train(images)

print('Training complete')
