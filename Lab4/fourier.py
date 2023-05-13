# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:24:06 2022

@author: NLP Lab
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

uk = []
vk = []
#print(uk)

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        uk.append(x)
        vk.append(y)
        print(uk)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
 

   
img = cv2.imread('period_input.jpg', 0)
imgg = cv2.imread('period_input.jpg', 0)

cv2.imshow('image', img)   
cv2.setMouseCallback('image', click_event)  
cv2.waitKey(0) 
cv2.destroyAllWindows()

#img = cv2.imread('period_input.jpg', 0)

plt.imshow(imgg, 'gray')
plt.show()



img_h, img_w = imgg.shape

notch = np.zeros((img_h, img_w), np.float32)

print(notch)

D0 = 25.0
n = 2 

#Dk = math.sqrt((u-img_h//2-uk)**2 + (v-img_w//2-vk)**2)
#Dkk = math.sqrt((u-img_h//2+uk)**2 + (v-img_w//2+vk)**2)

for u in range(img_h):
   for v in range(img_w):
       for k in range(0,4):
           Dk = math.sqrt((u- img_h/2.0 -uk[k])**2 + (v - img_w/2.0 - vk[k])**2)
           Dkk = math.sqrt((u - img_w/2.0 + uk[k])**2 + (v - img_w/2.0 + vk[k])**2)
           #print(Dkk)
           if(Dkk > D0 or Dk > D0):   
               if Dk ==0:
                   Dk = 1
               if Dkk ==0:
                   Dkk = 1
               notch[u][v] += (1.0/(1.0+(D0/Dk)**(2*n))) * (1.0/(1.0+(D0/Dkk)**(2*n)))
           
           
print(notch)
    


#img = np.log1p(img)

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
d = np.fft.fft2(imgg)
dft_shift = np.fft.fftshift(d)
magnitude = np.log1p(np.abs(dft_shift))
#magnitude = np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

plt.imshow(magnitude, cmap= 'gray')
plt.show()

#phase = cv2.phase(dft_shift[:,:,0], dft_shift[:,:,1])

plt.imshow(notch, cmap= 'gray')
plt.show()

s = dft_shift * notch
#s = d * notch

img_ba = np.log1p(np.abs(s))
plt.imshow(img_ba, cmap= 'gray')
plt.show()

img_inv_shift = np.fft.ifftshift(s)

#img_inv = cv2.idft(img_inv_shift)
img_inv = np.real(np.fft.ifft2(img_inv_shift))

#img_back = np.log(cv2.magnitude(img_inv[:,:,0],img_inv[:,:,1]))

#img_out = np.exp(img_inv)-1
plt.imshow(img_inv, cmap= 'gray')
plt.show()










