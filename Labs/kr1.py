import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

wh = (255, 255, 255)
bl = (0, 0, 0)
gr = (128, 128, 128)


img = np.zeros((300, 600), dtype=np.int16)
#cv2.floodFill(img, (100, 255,0), (0, 0), 20)

cv2.rectangle(img, (0, 0), (100, 100), bl, -1)
cv2.circle(img, (50, 50), 30, wh, -1)

cv2.rectangle(img, (100, 0), (200, 100), gr, -1)
cv2.rectangle(img, (125, 20), (175, 70), bl, -1)
cv2.rectangle(img, (200, 0), (300, 100), wh, -1)
cv2.circle(img, (250, 50), 30, bl, -1)

cv2.rectangle(img, (0, 100), (100, 200), gr, -1)
cv2.rectangle(img, (100, 100), (200, 200), wh, -1)

cv2.rectangle(img, (200, 100), (300, 200), bl, -1)

cv2.rectangle(img, (0, 200), (100, 300), wh, -1)
cv2.rectangle(img, (200, 100), (300, 200), bl, -1)
#cv2.circle(img, (150, 250), 30, gr, -1)
cv2.rectangle(img, (200, 200), (300, 300), gr, -1)

cv2.rectangle(img, (300, 0), (400, 100), bl, -1)
cv2.rectangle(img, (400, 0), (500, 100), gr, -1)
cv2.rectangle(img, (500, 0), (600, 100), wh, -1)

cv2.rectangle(img, (300, 100), (400, 200), gr, -1)
cv2.rectangle(img, (400, 100), (500, 200), wh, -1)
cv2.circle(img, (450, 50), 30, wh, -1)
cv2.rectangle(img, (500, 100), (600, 200), bl, -1)

cv2.rectangle(img, (300, 200), (400, 300), bl, -1)
cv2.circle(img, (350, 250), 30, gr, -1)
cv2.rectangle(img, (400, 200), (500, 300), gr, -1)
cv2.rectangle(img, (500, 200), (600, 300), wh, -1)

cv2.circle(img, (450, 250), 30, bl, -1)
cv2.circle(img, (50, 250), 30, gr, -1)

cv2.rectangle(img, (25, 125), (75, 170), wh, -1)
cv2.rectangle(img, (325, 25), (375, 75), gr, -1)

cv2.rectangle(img, (525, 225), (575, 275), gr, -1)
cv2.rectangle(img, (525, 25), (575, 75), bl, -1)
cv2.rectangle(img, (225, 125), (275, 170), wh, -1)


a3 = np.array( [[[125,125],[180,125],[155,170]]], dtype=np.int32 )
cv2.fillPoly(img, a3,bl)

a3 = np.array( [[[430,125],[480,125],[450,175]]], dtype=np.int32 )
cv2.fillPoly(img, a3,gr)

a3 = np.array( [[[540,125],[580,125],[550,175]]], dtype=np.int32 )
cv2.fillPoly(img, a3,wh)

a3 = np.array( [[[130,225],[170,225],[150,270]]], dtype=np.int32 )
cv2.fillPoly(img, a3,wh)

a3 = np.array( [[[225,225],[280,225],[250,270]]], dtype=np.int32 )
cv2.fillPoly(img, a3,wh)

a3 = np.array( [[[330,125],[380,125],[350,175]]], dtype=np.int32 )
cv2.fillPoly(img, a3,bl)


cv2.imshow('window_name', np.uint8(img))
cv2.waitKey(0)
cv2.imwrite('img_icx.jpg', img)

#image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)

arrimage1 = np.asarray(sobelx,np.int32)
arrimage2 = np.asarray(sobely,np.int32)

for i in range (arrimage1.__len__()):
    for j in range (arrimage1[i].__len__()):
        arrimage1[i,j]=arrimage1[i,j]/2+128
        arrimage2[i,j]=arrimage2[i,j]/2+128
arrim1=arrimage1.astype(np.uint8)
arrim2=arrimage2.astype(np.uint8)
cv2.imshow("1grad",arrim1)
cv2.waitKey(0)
cv2.imshow("2grad",arrim2)
cv2.waitKey(0)

newim = np.zeros((300,600),np.uint8)
finalimage= np.zeros((300,600,3),np.uint8)
for i in range (arrimage1.__len__()):
    for j in range (arrimage1[i].__len__()):
        tmp1 = np.int32(arrimage1[i,j])*np.int32(arrimage1[i,j])
        tmp2 = np.int32(arrimage2[i, j]) * np.int32(arrimage2[i, j])
        tmp3 = tmp1+tmp2
        tmp4 = np.sqrt(tmp3)
        newim[i, j] = np.uint8(tmp4)
        finalimage[i,j,0]=newim[i,j]
        finalimage[i,j,1]=arrimage1[i,j]
        finalimage[i,j,2]=arrimage2[i,j]
cv2.imshow("modul_grad",newim.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow("finalimg",finalimage.astype(np.uint8))



cv2.waitKey(0)
cv2.imwrite('img_result.jpg', finalimage.astype(np.uint8))