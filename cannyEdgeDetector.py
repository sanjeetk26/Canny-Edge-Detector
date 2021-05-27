# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:40:36 2020

@author: Sanjeet
"""

from scipy import ndimage
from scipy.ndimage import convolve


import numpy as np
import cv2



img=cv2.imread('D:\ML\Edge_Detector\Photo1.jpeg',0)
#cv2.imshow('photo',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

class cannysEdgeDetector:
    
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs=imgs
        self.imgs_final=[]
        self.imgs_smoothed= None
        self.gradientMat= None
        self.thetaMat=None
        self.nonMaxImg=None
        self.thresholdImg=None
        self.weak_pixel=weak_pixel
        self.strong_pixel=strong_pixel
        self.sigma=sigma
        self.kernel_size=kernel_size
        self.lowThreshold=lowthreshold
        self.highThreshold=highthreshold
        return
        

    def gaussian_kernel(size, sigma=1):
        x,y=np.mgrid[-size:size+1, -size:size+1]
        g=1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
        return g
        
    
    
#    img_2=convolve(img,gaussian_kernel(5))
    #cv2.imshow('photo',img_2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    def sobel_filters(img):
        Kx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],np.float32)
        Ky=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],np.float32)
        
        Ix=ndimage.filters.convolve(img,Kx)
        Iy=ndimage.filters.convolve(img,Ky)
        
        G=np.hypot(Ix,Iy)
        G=G/G.max()*255
        
        theta=np.arctan2(Iy,Ix)
        
        return G,theta
    
        
    def non_max_suppression(img, A):
        
        m,n=img.shape
        Z=np.zeros((m,n), dtype=np.int32)
        A=A*180.0/np.pi
        A[A<0]+=180
        
        for i in range (1,m-1):
            for j in range (1,n-1):
                
                a=255
                b=255
                
                if(0<=A[i,j]<22.5) or (157.5<=A[i,j]<=180):
                    a=img[i,j+1]
                    b=img[i,j-1]
                    
                elif (22.5<=A[i,j]<67.5):
                    a=img[i+1,j-1]
                    b=img[i-1,j+1]
                
                elif (67.5<=A[i,j]<112.5):
                    a=img[i+1,j]
                    b=img[i-1,j]
                elif (112.5<=A[i,j]<157.5):
                    a=img[i-1,j-1]
                    b=img[i+1,j+1]
                    
                if(img[i,j]>=a)and(img[i,j]>=b):
                    Z[i,j]=img[i,j]
                    
                else:
                    Z[i,j]=0
         
        
        return Z
    
#    G,theta=sobel_filters(img_2)
#    img4=non_max_suppression(G,theta)
    
    #G = Image.fromarray(np.uint8(G * 255) , 'L')
    #G.show()
    #
    #img4 = Image.fromarray(np.uint8(img4 * 255) , 'L')
    #img4.show()
    
    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        
        highThreshold=img.max()*highThresholdRatio
        lowThreshold=highThreshold*lowThresholdRatio
        
        m,n=img.shape
        thresh=np.zeros((m,n),dtype=np.int32)
        
        strong=np.int32(255)
        weak=np.int32(25)
        
        strong_i,strong_j=np.where(img>=highThreshold)
        
        weak_i,weak_j=np.where((img>=lowThreshold) & (img<highThreshold))
        
        zero_i,zero_j=np.where(img<lowThreshold)
        
        thresh[strong_i,strong_j]=strong
        thresh[weak_i,weak_j]=weak
        
        
        return (thresh,strong,weak)
        
        
        
    def hysterisis(img,strong=255):
        m,n=img.shape
        
        for i in range(1,m-1):
            for j in range(1,n-1):
                if ((img[i,j-1]==strong) or (img[i-1,j-1]==strong) or (img[i-1,j]==strong) or (img[i-1,j+1]==strong) or (img[i,j+1]==strong) or (img[i+1,j+1]==strong) or (img[i+1,j]==strong) or (img[i+1,j-1]==strong)):
                    img[i,j]=strong
                    
                else:
                    img[i,j]=0
                
        return img
    
    def detect(self):
        for i,img in enumerate(self.imgs):
            self.img_smoothed=convolve(img, self.gaussian_kernel(self.kernel_size))
            self.gradientMat, self.thetaMat=self.sobel_filters(self.imgs_smoothed)
            self.nonMaxImg=self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg=self.threshold(self.nonMaxImg)
            img_final=self.hysterisis(self.thresholdImg)
            self.imgs_final.append(img_final)
            
            return self.imgs_final
        

    
    

    
    
    
    