# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 02:53:45 2022

@author: ASUS
"""

import cv2
import  matplotlib.pyplot as plt
import numpy as np
import math

img = cv2.imread('lena.jpg', 0)

plt.imshow(img, 'gray')
plt.show()

img_h, img_w = img.shape

homo_filter = np.zeros((img_h, img_w), np.float32)

print(homo_filter)

sigma = 50.0
GH = 1.2
GL = 0.5
c = 0.1
D = 2 * math.pi * sigma**2

'''         Homomorphic Filter H(u,v)       '''

for u in range(img_h):
    for v in range(img_w):
        homo_filter[u][v] = (GH - GL) * (1 - np.exp(-(math.sqrt(((u - img_h//2)**2 + (v - img_w//2)**2)**2/D**2))*c)) +GL

plt.imshow(homo_filter, 'gray')
plt.show()


''''          Log of img + 1    '''
img = np.log1p(img)

'''              fourier transform of img    '''
f = np.fft.fft2(img)
'''             Shift intensity to center       '''
f_shift = np.fft.fftshift(f)

mag = np.abs(f_shift)   #magnitude of img

magg = np.log(mag)  # plotting magnitude

plt.imshow(magg, 'gray')
plt.show()

angle = np.angle(f_shift)   # angle of img

plt.imshow(angle, 'gray')
plt.show()

s = homo_filter * f_shift   #  filtering s = h * f

'''             Inverse shift '''
inv_shift = np.fft.ifftshift(s)

'''             Inverse fourier transfrom     '''
inv_f = np.real(np.fft.ifft2(inv_shift)) 


'''           exp of inverse img - 1        '''
inv = np.exp(inv_f)-1

plt.imshow(inv, 'gray')
plt.show()



