# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:08:20 2018

@author: i
"""


import cv2
import numpy as np
from opens import *
image_hsv = None 
pixel = (20,60,80) 

a=0
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)
        cv2.destroyAllWindows()
        global a
        a=1
        openpaint(lower,upper)

def main():
    import sys
    global image_hsv, pixel
    
    cap= cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()
        image_src = img
        
        cv2.imshow("bgr",image_src)

        image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv",image_hsv)
        cv2.setMouseCallback('bgr', pick_color)
        if cv2.waitKey(1) == 27:
                break
        if(a==1):
            break

    cv2.destroyAllWindows()
if __name__=='__main__':
    main()