# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:59:54 2018

@author: jaydeep thik
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from PIL import Image
from utility import encode_one_hot, load_dataset
from opens import *


new_saver = tf.train.import_meta_graph('new-model.meta')
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name('y_pred:0')
predict_op = graph.get_tensor_by_name("predict_op:0")
X = graph.get_tensor_by_name('X:0')

    

    

# instantiate the camera object and haar cascade
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

l = [0,0,0,0,0]

with tf.Session() as sess:
   while True:
        ret, frame = cam.read()
        #frame = cv2.flip(frame, 3)
        if ret == True:
            cv2.rectangle(frame, (300,300), (50,50), (0,255,0),0)
            crop_img = frame[50:300, 50:300]
            
            X_data = cv2.resize(crop_img, (64, 64))
            #X_data = cv2.cvtColor(X_data, cv2.COLOR_BGR2RGB)

            X_data = X_data.reshape((1, 64, 64, 3))
            
            X_data = X_data/255.
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            text = sess.run(y_pred,feed_dict={X:X_data})
            print(np.max(text))
            if np.max(text)>0.90:
                value = sess.run(predict_op, feed_dict ={X:X_data})
                print("value : ",value)
                value = int(value)
                l[value-1] = l[value-1] + 1
                if(l[value-1] == 5):
                    openpaint()
                for i in range(len(l)):
                    if(i == value-1):
                        continue
                    l[i]=0
                
                cv2.putText(frame, str(value),(50,50), font, 1,(0,255,0), 2)       
            cv2.imshow('op', frame)
            #cv2.imshow('cap', crop)
            if cv2.waitKey(1) == 27:
                break
        else:
            print ('Error')
            break

cam.release()
cv2.destroyAllWindows()

"""
images= cv2.imread('F:/machine learning/code/hand gesture/test_hi/ccam_50.jpg')




images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

"""
"""
value = (33, 33)
hsv = cv2.cvtColor(images,cv2.COLOR_BGR2HSV)
blur = cv2.GaussianBlur(hsv,value,0)
    # blurred = cv2.GaussianBlur(grey, value, 0)
lower_green = np.array([80,50,30])
upper_green = np.array([255,255,255])
mask = cv2.inRange(hsv, lower_green, upper_green)
gaussian = cv2.GaussianBlur(mask, (11,11), 0)

erosion = cv2.erode(mask, None, iterations = 1)
dilated = cv2.dilate(erosion,None,iterations = 1)
median = cv2.medianBlur(dilated, 7)




images = median

"""
"""

plt.imshow(images)
plt.show()

import os

#images=[]

path = "F:/machine learning/code/hand gesture/test_hi"


"""
"""
for img in os.listdir(path):
    image = plt.imread('F:/machine learning/code/hand gesture/test_hi/'+img)
    image = cv2.resize(image, (64, 64))
    images.append(image)
"""   
"""

 
images = cv2.resize(images,(64,64))
#i=1000

#images = X_train[i]
#print(y_train[0][i])
#image = np.array(image, dtype=np.uint8)

#image = X_test[55]
#plt.imshow(image)

#images = np.array(images)

#images = images.astype('float32')

images = images/225.
#images  =  X_test[11]
plt.imshow(images)
#print(images[0])
images = images.reshape((1,64,64,3))


with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    text= sess.run(predict_op,feed_dict={X:images})
    print("value : ",text)



"""
    
 