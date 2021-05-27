# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:10:14 2020

@author: Sanjeet
"""


import cv2
import cannyEdgeDetector as cd

img=cv2.imread('D:\ML\Edge_Detector\Photo1.jpeg',0)
#cv2.imshow('photo',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

detector=cd.cannysEdgeDetector(img,sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
img=detector.detect()

img1=img[0]
cv2.imshow('photo',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()