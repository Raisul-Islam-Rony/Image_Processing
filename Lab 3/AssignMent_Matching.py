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



def Erlang():
   erlang_pdf=np.zeros(256,np.float64)
   k=int(input())
   u=int(input())
   for i in range (255):
       val= i**(k-1)
       val=val*math.exp(-(i)/u);
       val=val/(u**k)
       val=val/(math.factorial(k-1))
       erlang_pdf[i]=val
       
   return erlang_pdf
   
    
image=cv2.imread("eye.png",cv2.IMREAD_GRAYSCALE)


histogram=np.zeros(256)
height=image.shape[0]
width=image.shape[1]
total_pixel=height*width


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
          histogram[image[i][j]]+=1
          

         
pdf=histogram/total_pixel

print(pdf)


cv2.imshow("Input",image)
l=255
sum=0
s=np.zeros(256)
cdf=np.zeros(256,np.float64)

for i in range (256):
    sum=sum+pdf[i]
    cdf[i]=sum
    s[i]=round(l*cdf[i])
    




Erlang_Pdf=[]
Erlang_Pdf=Erlang()

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
plt.title("Erlang PDF")
plt.plot(Erlang_Pdf)
plt.show()









        