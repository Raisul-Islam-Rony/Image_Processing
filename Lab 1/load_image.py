import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


img = cv2.imread('noise.png',cv2.IMREAD_GRAYSCALE)

#out=img.copy()

# #out = np.zeros((512,512), dtype=np.uint8)
# print(img.shape[0])
# print(img.shape[1])
image_bordered = cv2.copyMakeBorder(src=img, top=25, bottom=25, left=25, right=25,borderType= cv2.BORDER_WRAP)#BORDER_WRAP, cv.BORDER_REFLECT  
cv2.imshow("input ",image_bordered)

def Gaussian_kernel():
    sigma=int(input("Enter Value Of Sigma : "))
    print("Value Of Sigma : ",sigma)
    
    cons=2*3.1416*sigma*sigma;
    cons=1/cons
    height=5*sigma
    width=5*sigma
    center=height//2
    
    kernel=[]
    sum=0.0
    for i in range(-center,center+1):
        a=[]
        for j in range(-center,center+1):
            val=(i*i)+(j*j)
            val=-val
            val=val//(2*sigma*sigma)
            val=np.exp(val)
            val=val*cons
            sum=sum+val
            a.append(val)
            
        kernel.append(a)
        
        
    kernel=np.array(kernel)
    reverse_kernel=[]
        
    N=kernel.shape[0]
    row=N-1
    print("Sum is ",sum)
    print()
    while(row>=0):
         col=N-1
         a=[]
         while(col>=0):
             a.append(kernel[row][col])
             col=col-1
         reverse_kernel.append(a)
         row=row-1

    

    
    reverse_kernel=np.array(reverse_kernel)
   
    i=0
    while(i<height):
        j=0
        while(j<width):
            print(reverse_kernel[i][j], end=" ")
            j=j+1
        print()
        i=i+1
    
    print()
    print()
    
    
    
    
    out = np.ones((image_bordered.shape[0],image_bordered.shape[1]),dtype=np.uint8)
    
    
    
    for i in range(image_bordered.shape[0]-reverse_kernel.shape[0]):
        for j in range(image_bordered.shape[1]-reverse_kernel.shape[1]):
            sum=0
            for k in range(reverse_kernel.shape[0]):
                for p in range(reverse_kernel.shape[1]):
                    a=image_bordered.item((i+k,p+j))*reverse_kernel[k][p]
                    sum=sum+a
                    out.itemset((i,j),sum)
    cv2.imshow("output",out)
                



Gaussian_kernel()

cv2.waitKey(0)
cv2.destroyAllWindows()


        #out.itemset((i+1,j+1),sum)
 




 


  







