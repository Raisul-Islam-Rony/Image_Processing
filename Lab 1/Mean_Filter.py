import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from skimage import io, img_as_float
from skimage.filters import gaussian
from math import ceil,pi

#%%

# def gaussianKernel(sigma):
#     k=15
#     # k=ceil(sigma*5)
#     # if(k%2==0):
#     #     k+=1
#     a=k//2
#     sigma2=sigma*sigma
#     normalizer= 2*3.1416*sigma2
#     normalizer= 1/normalizer
#     krnl=np.full((k,k),normalizer)
#     for i in range(-a,a+1):
#         for j in range(-a,a+1):
#             pwr= ((i*i)+(j*j))
#             pwr=pwr/(2*sigma2)
#             val= np.exp(-pwr)
#             krnl[i+a][j+a]=krnl[i+a][j+a]*val
#     return krnl
# #%%


# spatial_kernel=gaussianKernel(25)


#%%
def showImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


img= cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

X=img.shape[0]
Y=img.shape[1]

def Bilateral_filter(img,kernel,sigma):
    #%%
    n=5
    a=n//2

    inpt= cv2.copyMakeBorder(src=img,top=a,bottom=a,left=a,right=a, borderType=cv2.BORDER_CONSTANT)
    #%%
    inpt=np.float64(inpt)
    out= np.zeros(img.shape, dtype=np.float64)
    


    for i in range (img.shape[0]):
          for j in range(img.shape[1]):
            x=i+a
            y=j+a
            intensity= inpt[x][y]
            neighbour= inpt[i:(i+n),j:(j+n)]
            temp=intensity-neighbour
            temp= np.square(temp)
            temp= temp/ (2*(sigma**2))
            range_kernel= np.exp(-temp)
            bilateral_kernel= kernel * range_kernel
            upr= bilateral_kernel*neighbour
            sumU=np.sum(upr) 
            suml=np.sum(bilateral_kernel) 
            res= sumU/suml
            out[i][j]=res

    #%%      
    result = cv2.normalize(out, None, 0, 1.0,
    cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    showImage(img, 'title')
    showImage(result, 'filter')





mean_kernel=np.array([[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25]])
Bilateral_filter(img, mean_kernel, 5)