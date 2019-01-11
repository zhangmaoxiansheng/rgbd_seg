import cv2 as cv
import numpy as np


img = cv.imread("./1_depth/fid_629.png",0)

ret,thresh1 = cv.threshold(img,200,255,cv.THRESH_TOZERO_INV)#larger than 200 = 0 
ret2,thresh2 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)#smaller that 50 = 0
thresh3 = cv.resize(thresh2,(768,636))
dilated_res = cv.dilate(thresh3,(5,5))

test = np.ones((636,768),dtype = np.uint8)

result = np.multiply(test,dilated_res)

cv.imshow("threshold3",thresh3)
cv.imshow("dilated_res",dilated_res)
cv.imshow("res",result)

cv.waitKey(0)

print("finished")