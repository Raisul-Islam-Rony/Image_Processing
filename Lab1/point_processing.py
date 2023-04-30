# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:12:56 2023

@author: Raisul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)

image_bordered=cv2.copyMakeBorder(src=img,top=25,bottom=25,left=25,right=25,borderType=cv2.BORDER_WRAP)
#cv2.imshow("Input",image_bordered)
cv2.waitKey(0)
cv2.destroyAllWindows()


def gamma_correction(a):
    
    #a=float(input("Enter Gamma "))
    
    out=np.ones((image_bordered.shape[0],image_bordered.shape[1]),dtype=np.uint8)
    for i in range (image_bordered.shape[0]):
        for j in range (image_bordered.shape[1]):
            c=image_bordered[i][j]
            d=pow(c,a)
            out.itemset((i,j),d)
    cv2.imshow("output",out)

gamma_correction(0.75)

def sobel_kernel(n):
    c=n//2
    kernel=[]
    for i in range (-c,c+1):
        a=[]
        for j in range (-c,c+1):
            if(i==0):
                kernel.append(j*2)
            else:
                kernel.append()
        kernel.append(a)
    kernel=np.array(kernel)
    return kernel

def Sobel_Kernel(n):
    mid=n//2
    kernel=[]
    for i in range(-mid,mid+1):
        a=[]
        for j in range(-mid,mid+1):
            if(i==0):
                a.append(j*2)
            else:
                a.append(j)
        kernel.append(a)
    kernel=np.array(kernel)
    return kernel


cv2.waitKey(0)
cv2.destroyAllWindows()

