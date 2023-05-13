# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:29:10 2023

@author: Raisul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def Find_Val(s,erl_eq):
    if(erl_eq[0]>s):
        return 0
    for i in range(256):
        if(s==erl_eq[i]):
            return i
        if(erl_eq[i]>s):
            return i-1
    return 255



def Erlang(k,u):
   erlang_pdf=np.zeros(256,np.float64)
   for i in range (255):
       val= i**(k-1)
       val=val*math.exp(-(i)/u);
       val=val/(u**k)
       val=val/(math.factorial(k-1))
       erlang_pdf[i]=val
       
   return erlang_pdf
   
    
image=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)


histogram=np.zeros(256)
height=image.shape[0]
width=image.shape[1]
total_pixel=height*width


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
          histogram[image[i][j]]+=1
          

         
pdf=histogram/total_pixel

print(pdf)
# plt.hist(image.ravel(),256,[0,256])

cv2.imshow("Input",image)
l=255
sum=0
s=np.zeros(256)
cdf=np.zeros(256,np.float64)

for i in range (256):
    sum=sum+pdf[i]
    cdf[i]=sum
    s[i]=round(l*cdf[i])
    
'''
out=np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        out[i][j]=s[image[i][j]]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("input Image Histogram")
plt.hist(image.ravel(),256,[0,256])
plt.show

plt.subplot(1,2,2)
plt.title("input Image Histogram")
plt.hist(out.ravel(),256,[0,256])
plt.show
'''

'''
histogram1=np.zeros(256)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
          histogram1[out[i][j]]+=1

pdf1=histogram1/total_pixel
cdf1=pdf1
sum=0

for i in range(256):
    sum=sum+pdf1[i]
    cdf1[i]=sum
    
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("input Image CDF")
plt.plot(cdf)
plt.show

plt.subplot(1,2,2)
plt.title("output image CDF")
plt.plot(Erlang(2,2))
plt.show

cv2.imshow("Output",out)
'''

Erlang_Pdf=[]
Erlang_Pdf=Erlang(2,2)

Erlang_Cdf=np.zeros(256,np.float64)
Erlang_Equalized=np.zeros(256)

sum=0

for i in range(256):
    sum=sum+ Erlang_Pdf[i]
    Erlang_Cdf[i]=sum
    Erlang_Equalized[i]=round(Erlang_Cdf[i]*255)


match=np.zeros(256)


for i in range (256):
    match[i]=Find_Val(s[i], Erlang_Equalized)
    

print("Map")
print(Erlang_Equalized)
print(s)
print(match)


out=np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        out[i][j]=match[image[i][j]]


cv2.imshow("Output",out)






plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("pdf")
plt.plot(Erlang_Pdf)
plt.show()

plt.subplot(1,2,2)
plt.title("Lena PDF")
plt.plot(s)
plt.show()




        