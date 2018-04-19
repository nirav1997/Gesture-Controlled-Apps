# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:03:40 2018

@author: i
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    crop_img = frame[50:300, 50:300]
    fgmask = fgbg.apply(crop_img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('frame1',fgmask)
    im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        largest_area = sorted(contours, key=cv2.contourArea)[-1]
        
        x,y,w,h = cv2.boundingRect(largest_area)
    
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.imshow('frame',frame)
        # This displays the frame, mask 
        # and res which we created in 3 separate windows.
    except:
        pass
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

