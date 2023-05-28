# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dpc
import math



def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('input.jpg', 0)
print(img_input.shape)
img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = 1 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

x=int(input())
y=int(input())
r=int(input())

kernel=np.ones((img_input.shape[0],img_input.shape[1]))

for u in range (img_input.shape[0]):
    for v in range(img_input.shape[1]):
        d=pow(u-x,2) + pow(v-x,2)
        d=math.sqrt(d)
        if(d<=r):
            kernel[u][v]=0


ang = np.angle(ft_shift)
ans_magnitude=np.multiply(magnitude_spectrum_ac,kernel)
ans_mag_log=np.log(np.abs(ans_magnitude)+1)
ans_mag_scale=min_max_normalize(ans_mag_log)

## phase add
final_result = np.multiply(ans_magnitude, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("ans_mag_scale",ans_mag_scale)

cv2.imshow("Phase",ang)
cv2.imshow("Inverse transform",img_back_scaled)
cv2.imshow("Filter",kernel)



cv2.waitKey(0)
cv2.destroyAllWindows() 
