import cv2 as cv
import numpy as np


img = cv.imread("./1_depth/fid_244.png",0)
img_ = cv.imread("./1_depth/fid_1.png")
img_res = cv.imread("./example_result/1477_seg_read.png",0)
img_ = cv.resize(img_,(768,636))
ret,thresh1 = cv.threshold(img,179,255,cv.THRESH_TOZERO_INV)#larger than 200 = 0 
ret2,thresh2 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)#smaller that 50 = 0
thresh3 = cv.resize(thresh2,(768,636))
dilated_res = cv.dilate(thresh3,(5,5))
cv.imshow("z",dilated_res)
cv.waitKey(0)


contours, hierarchy = cv.findContours(img_res,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  

max_area = 0
index = 0
for i in range(0,len(contours)):
    area = cv.contourArea(contours[i])
    if(area > max_area):
        max_area = area
        index = i
final_contours = contours[index]
cv.drawContours(img_,final_contours,-1,(0,0,255),3)  
    
x, y, w, h = cv.boundingRect(final_contours) 
#cv.rectangle(img_res, (x-20,y), (x+w+20,y+h), (153,153,0), 5) 

img_roi = dilated_res[y:y+h-80,x-20:x+w+20]
img_roi2 = img_res[y:y+h-80,x-20:x+w+20]

a = img_roi != img_roi2
r = a.sum()

r = float(r)/img_roi.size 
#numel = w*h
img_res[y:y+h-80,x-20:x+w+20] = img_roi + img_res[y:y+h-80,x-20:x+w+20] 
img_res[img_res > 0] = 255
img_res = cv.erode(img_res,(5,5))
img_res=cv.dilate(img_res,(5,5))
cv.imshow("roi",img_roi)
#cv.imshow("threshold3",thresh3)
cv.imshow("dilated_res",dilated_res)
#cv.imshow("res",result)
cv.imshow("contours",img_res)
cv.waitKey(0)
print("fin")
print(len(contours))