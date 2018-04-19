# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:03:09 2018

@author: i
"""

import cv2
import numpy as np 
from pywinauto.application import Application


# Webcamera no 0 is used to capture the frames
cap = cv2.VideoCapture(0) 

# This drives the program into an infinite loop.
while(1):		
    # Captures the live stream frame-by-frame
    _, frame = cap.read() 
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame,(1366,768))
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([75,215,130])
    upper_red = np.array([95,235,210])

# Here we are defining range of bluecolor in HSV
# This creates a mask of blue coloured 
# objects found in the frame.
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # The bitwise and of the frame and mask is done so 
    # that only the blue coloured objects are highlighted 
    # and stored in res
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        largest_area = sorted(contours, key=cv2.contourArea)[-1]
        
        x,y,w,h = cv2.boundingRect(largest_area)
    
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.imshow('frame',frame)
        # This displays the frame, mask 
        # and res which we created in 3 separate windows.
    except:
        pass
    k = cv2.waitKey(5)
    if k == 27:
        break
    
    # Destroys all of the HighGUI windows.
cv2.destroyAllWindows()

# release the captured frame
cap.release()