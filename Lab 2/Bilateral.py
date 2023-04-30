import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from skimage import io, img_as_float
from skimage.filters import gaussian
from math import ceil,pi

def Convolution(img,reverse_kernel):
    
    image_bordered= cv2.copyMakeBorder(src=img, top=reverse_kernel.shape[0]//2, bottom=reverse_kernel.shape[0]//2, left=reverse_kernel.shape[0]//2, right=reverse_kernel.shape[0]//2,borderType= cv2.BORDER_WRAP)
    cv2.imshow("Input",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    center=reverse_kernel.shape[0]//2
    
    out = np.zeros((image_bordered.shape[0],image_bordered.shape[1]),dtype=np.uint8)
    
    l=reverse_kernel.shape[0]
    
    for i in range(image_bordered.shape[0]-reverse_kernel.shape[0]):
        for j in range(image_bordered.shape[1]-reverse_kernel.shape[1]):
            sum=0
            for k in range(reverse_kernel.shape[0]):
                for p in range(reverse_kernel.shape[1]):
                    a=image_bordered.item((i+k,p+j))*reverse_kernel[l-k-1][l-p-1]
                    sum=sum+a
                    
            out.itemset((i+center,j+center),sum)
    cv2.imshow("output",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def Gaussian_kernel(a):
    sigma=a
    #print(sigma)
    cons=2*3.1416*sigma*sigma
    cons=1/cons
   
    kernel=[]
    
    height=sigma*5
    center=height//2
    for i in range (-center,center+1):
        a=[]
        for j in range (-center,center+1):
            val=i*i+j*j
            val=-(val/(2*sigma*sigma))
            val=np.exp(val)
            val=val*cons
            a.append(val)
        kernel.append(a)

    
        
    kernel=np.array(kernel)
    return kernel


#%%
def showImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


img= cv2.imread('taj.jpg', cv2.IMREAD_GRAYSCALE)

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
    


    for row in range (img.shape[0]):
          for col in range(img.shape[1]):
            val_row=row+a
            val_col=col+a
            intensity= inpt[val_row][val_col]
            neighbour= inpt[row:(row+n),col:(col+n)]
            temp=intensity-neighbour
            temp= np.square(temp)
            temp= temp/ (2*(sigma**2))
            range_kernel= np.exp(-temp)
            bilateral_kernel= kernel * range_kernel
            upr= bilateral_kernel*neighbour
            sumU=np.sum(upr) 
            suml=np.sum(bilateral_kernel) 
            res= sumU/suml
            out[row][col]=res

    #%%      
    result = cv2.normalize(out, None, 0, 1.0,
    cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    showImage(img, 'title')
    showImage(result, 'filter')





mean_kernel=np.array([[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25],[1/25,1/25,1/25,1/25,1/25]])
Bilateral_filter(img, mean_kernel, 25)

# a=Gaussian_kernel(2) 
# Convolution(img, a)