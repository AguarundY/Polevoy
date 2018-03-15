import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import time

img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(img.shape[0]):
   for j in range(img.shape[1]):
       img[i, j, 0] = 64 * np.sin(i / 12) + 128
       img[j, i, 2] = 64 * np.cos(i / 6) + 128
 #img[i, j, 2] = 64 * np.sin(i/4) + 128
cv2.imwrite('img.jpg', img)
cv2.imshow('window_name', img)
cv2.waitKey(0)
cv2.destroyWindow('window_name')