# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:30:17 2018

@author: i
"""


import cv2
import numpy as np
from numpy import linalg

cap= cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret == True:
        
        
        
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        
        cv2.rectangle(img, (300,300), (50,50), (0,255,0),0)
        crop_img = img[50:300, 50:300]
        
        #b_img = np.copy(crop_img)
        #b_img += 10
        
        crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        
        #crop_img = cv2.GaussianBlur(crop_img,(5,5),0)
    
        #rett, th = cv2.threshold(crop_img,127,255,cv2.THRESH_BINARY)
        th = cv2.adaptiveThreshold(crop_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        
        th = cv2.GaussianBlur(th,(25, 25), 1)
        
        #th = cv2.blur(th,(5,5))
        
        #th = cv2.medianBlur(th,50)
        
        #cv2.imshow('th',th)
        #cv2.waitKey(0)
        
        image, contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_area = sorted(contours, key=cv2.contourArea)[-1]
        
        mask = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(mask, [largest_area], 0, (255,255,255,255), -1)
        dst = cv2.bitwise_and(crop_img, mask)
        mask = 255 - mask
        roi = cv2.add(dst, mask)
    
        #rett, roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
        #roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    
        cv2.imshow('image', th)
        cv2.imshow('roi', roi)
        #cv2.imshow('b', b_img)
        
        '''
        contours = sorted(contours, key=cv2.contourArea)
         
        for i, contour in enumerate(contours):
            mask = np.zeros(crop_img.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, (255,255,255,255), -1)
            dst = cv2.bitwise_and(crop_img, mask)
            mask = 255 - mask
            roi = cv2.add(dst, mask)
            cv2.imshow('roi' + str(i), roi)
        
        cv2.imshow('image', img)
        '''
        
        if cv2.waitKey(1) == 27:
            break
    
    #cv2.destroyAllWindows()
    
cap.release()
cv2.destroyAllWindows()
    	
    	