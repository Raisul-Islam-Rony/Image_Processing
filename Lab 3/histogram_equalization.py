# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:25:05 2022

@author: ASUS
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lena.jpg',0)

plt.imshow(img, 'gray')
plt.title("Input Image")
plt.show()


# plt.hist(img.ravel(), 255, [0,255])
# plt.title("Input Image Histogram")
# plt.show()

equalized_img =  np.zeros_like(img)

img_h, img_w = img.shape

total_pixel = img_h * img_w


''' Caalculating pdf(normal_hist) '''

histogram = np.zeros(256)

for i in range(img_h):
    for j in range(img_w):
        histogram[img[i][j]] += 1

plt.plot(histogram)
print(histogram)
plt.show()

pdf = histogram / total_pixel

plt.plot(pdf)
plt.show()

print(histogram)
# print(pdf)

''''  Calculatin cdf && s = (L-1)*sum(pdf[i]) '''

cdf = pdf

s = pdf

sum= 0.0
L = 256

for i in range (256):
    sum += pdf[i]
    cdf[i] = sum
    s[i] = round((L-1) * cdf[i])


# print(cdf)

plt.plot(cdf)
plt.show()


# plt.plot(s)
# plt.show()

# print(s)

'''  update value img[i][j] with s[img[i][j]] if img[2][5] = 10 then img[2][5] = s[10] '''

for i in range(img_h):
    for j in range(img_w):
        equalized_img[i][j] = s[img[i][j]]
        
plt.imshow(equalized_img, 'gray')
plt.title("Histogram Equalized Image")
plt.show()

# plt.hist(equalized_img.ravel(), 255, [0, 255])
# plt.title("Histogram Equalized Image Histogram")
# plt.show()


