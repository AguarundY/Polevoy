import cv2
import numpy as np


img = cv2.imread(r"gray.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.array([[-0.5, 0.5]])

dst = cv2.filter2D(img, -1, kernel)
vis = np.concatenate((img, dst), axis=1)
cv2.imshow('window_name', vis)
cv2.waitKey(0)
cv2.destroyWindow('window_name')

