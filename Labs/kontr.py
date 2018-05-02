import numpy as np
import cv2 as cv2
from itertools import permutations
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

colors = [0, 1, 2]

image = np.zeros((300, 600), dtype=np.int8)

ind = 0
for figure in ["square", "triangle", "circle"]:

    for pair in permutations(colors, 2):

        img = np.ones((100, 100), dtype=np.int8) * pair[0]

        if figure == "circle":
            cv2.circle(img, (50, 50), 40, pair[1], -1)

        if figure == "triangle":
            pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], pair[1])

        if figure == "square":
            pts = np.array([[25, 25], [25, 75], [75, 75], [75, 25]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], pair[1])

        np.copyto(image[(ind % 3) * 100: (ind % 3) * 100 + 100,
                  (ind // 3) * 100: (ind // 3) * 100 + 100], img)

        ind += 1

plt.figure(figsize=[21, 7])
plt.imshow(image)
plt.show()
cv2.imwrite('img_icx.png', image)

img = image * 255 / 2

kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
image_filtered_x = (cv2.filter2D(img, -1, kernel))
image_filtered_x = (image_filtered_x/(2*image_filtered_x.max()) + 0.5)*255
image_filtered_y = (cv2.filter2D(img, -1, kernel.T))
image_filtered_y = (image_filtered_y/(2*image_filtered_y.max()) + 0.5)*255

plt.figure(figsize=[16, 11.5])
plt.subplot(211)
plt.imshow(image_filtered_x, cmap = 'gray')
plt.axis("off")

plt.subplot(212)
plt.imshow(image_filtered_y, cmap = 'gray')
plt.axis("off")

plt.show()

image_filtered = (image_filtered_x**2 + image_filtered_y**2)**0.5
image_filtered = (image_filtered - image_filtered.min())/(image_filtered.max() - image_filtered.min())*255
plt.figure(figsize=[10,6])
plt.imshow(image_filtered, cmap = 'gray')
plt.axis("off")
plt.show()

image_result = np.zeros((*img.shape, 3), dtype=np.uint8)
image_result[:,:,0] = image_filtered
image_result[:,:,1] = image_filtered_x
image_result[:,:,2] = image_filtered_y
image_result = cv2.cvtColor(image_result, cv2.COLOR_LAB2RGB)

cv2.imwrite('img_result.jpg', image_result)

plt.figure(figsize=[10, 6])
plt.imshow(image_result)
plt.axis("off")
plt.show()