import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"app.png")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist(imggray, [0],None,[256],[0,256])
per = imggray.flatten()
histo,bins = np.histogram(imggray.flatten(),256,[0,256])
plt.plot(histo)
#plt.show()

num_pix = imggray.size

quantile_black = 0.05
required_sum_black = int(num_pix * quantile_black)
summ = 0
index = -1
for i,v in enumerate(histo):

    summ+=v
    if summ > required_sum_black:
        index = i
        break
index_black = index

quantile_white = 0.05
required_sum_white = int(num_pix * quantile_white)
summ = 0
index = -1
for i,v in enumerate(reversed(histo)):
    summ+=v
    if summ > required_sum_white:
        index = i
        break
index_white = len(histo) - index

f=0

def relu(x):
    if x < index_black:
        return 0
    if x >= index_white:
        return 255
    return (x - index_white) * 255 / (index_white - index_black) + 255

lookup_table = np.zeros(256, dtype=np.uint8)
for i in range(256):
    lookup_table[i] = int(relu(i))


copslow = imggray.copy()
for i in range(imggray.shape[0]):
    for j in range(imggray.shape[1]):
        copslow[i,j] = relu(imggray[i,j])


cop = imggray.copy()
cop = cv2.LUT(cop, lookup_table)

hist = cv2.calcHist([copslow],[0],None,[256],[0,256])
plt.plot(hist)

cv2.imshow('gray', copslow)
cv2.imshow('gray_pic', imggray)
cv2.waitKey(0)
#plt.imshow(copslow, cmap='gray')

# cv2.imshow('image1', hist)
# cv2.waitKey(0)
# cv2.destroyWindow('image1')
