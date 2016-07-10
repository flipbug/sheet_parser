import numpy as np
import cv2

img = cv2.imread('../../tests/score01.png', 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
template = cv2.imread('images/note.png', 0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF',
           'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

# find notes
method = cv2.TM_CCOEFF_NORMED
res = cv2.matchTemplate(img, template, method)

# find and draw lines
lines = cv2.HoughLines(edges, 1, np.pi/180, 100, 10, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 10, 100), 1)

# draw rectangles for found notes
threshold = 0.80
location = np.where(res >= threshold)
for pt in zip(*location[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 125, 1)

#binary = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
#gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
