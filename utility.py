# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:46:16 2018

@author: jaydeep thik
"""

import h5py
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

def load_dataset():
    """
    train_dataset = h5py.File('dataset/train_signs.h5','r')
    X_train_orig = np.array(train_dataset['train_set_x'][:])
    y_train_orig = np.array(train_dataset['train_set_y'][:])
    y_train_orig = y_train_orig.reshape((1,y_train_orig.shape[0]))
    
    test_dataset = h5py.File('dataset/test_signs.h5','r')
    X_test_orig = np.array(test_dataset['test_set_x'][:])
    y_test_orig = np.array(test_dataset['test_set_y'][:])
    y_test_orig = y_test_orig.reshape((1,y_test_orig.shape[0]))
    
    classes = np.array(test_dataset['list_classes'][:])
    """
    dataset = h5py.File('./dataset.h5', 'r')
    X_data = np.array(dataset['X_data'][:])
    y_data = np.array(dataset['y_data'][:])
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_data, y_data, test_size=0.3, random_state=3)
    classes = np.array(np.unique(y_train_orig))
    #print(classes)
    y_train_orig = y_train_orig.reshape((1, y_train_orig.shape[0]))
    y_test_orig = y_test_orig.reshape((1, y_test_orig.shape[0]))
    
    return X_train_orig, X_test_orig, y_train_orig, y_test_orig, classes

def encode_one_hot(labels, c):
    c = tf.constant(c, name='c')
    
    #print(labels.shape)
    one_hot = tf.one_hot(labels, c, axis=-1)
    
    with tf.Session() as sess:
        encode = sess.run(one_hot)
        
    #print(encode.shape)    
    return encode.reshape(encode.shape[1], encode.shape[2])
    