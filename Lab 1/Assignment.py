import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import statistics



img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)

#image_bordered=cv2.copyMakeBorder(src=img,top=25,bottom=25,left=25,right=25, borderType=cv2.BORDER_WRAP)
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
    


# def Reverse_Kernel(kernel):
#     reverse_kernel=[]
        
#     N=kernel.shape[0]
#     row=N-1
    
#     print()
#     while(row>=0):
#          col=N-1
#          a=[]
#          while(col>=0):
#              a.append(kernel[row][col])
#              col=col-1
#          reverse_kernel.append(a)
#          row=row-1
         
#     reverse_kernel=np.array(reverse_kernel)
#     return reverse_kernel
    
    
    

    



def Mean_Filter(image_bordered,size):
    
    print("Kernel Size : ",size)
    cv2.imshow("Input",image_bordered)
    out = np.ones((image_bordered.shape[0],image_bordered.shape[1]),dtype=np.uint8)



    for i in range((image_bordered.shape[0]-size+1)):
        for j in range((image_bordered.shape[1]-size+1)):
            sum=0
            for k in range(size):
                for p in range(size):
                    sum=sum+image_bordered.item(i+k,j+p)
            out.itemset((i+size//2,j+size//2),sum/(size*size))
    cv2.imshow('Output',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def Median_Filter(image_bordered,size):
    
    print("Kernel Size : ",size)
    cv2.imshow("Input",image_bordered)
    
    out = np.ones((image_bordered.shape[0],image_bordered.shape[1]),dtype=np.uint8)
    
    for i in range((image_bordered.shape[0]-size+1)):
        for j in range((image_bordered.shape[1]-size+1)):
            a=[]
            for k in range(size):
                for p in range(size):
                    b=image_bordered.item(i+k,j+p)
                    a.append(b)
               
            out.itemset((i+size//2,j+size//2),statistics.median(a))
    cv2.imshow('Output',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

##Point Processing

def Contrast_Streching(img,a,b):
    r_min=img.min()
    r_max=img.max()
    
    cv2.imshow("Input",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    out=np.zeros((img.shape[0],img.shape[0]),dtype=np.uint8)
    
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            x=(img[i][j])-r_min
            y=b-a
            z=r_max-r_min
            p=x*(y/z)
            out.itemset((i,j),p+a)
    
    cv2.imshow("Output",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    


def Gamma(img,c,a):
    cv2.imshow("Input",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    out=np.zeros((img.shape[0],img.shape[0]),dtype=np.uint8)
    
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            x=img[i][j]
            ans=x**a
            out.itemset((i,j),ans*c)
            
    cv2.imshow("Output",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def Inverse_Log_Transformation():
    cv2.imshow("Input",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    out=np.zeros((img.shape[0],img.shape[0]),dtype=np.uint8)
    
    a=img.max()
    
    c=255/(math.log(1+a))
    
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            data=np.exp((img[i][j])/c)
            out.itemset((i,j),data)
            
    cv2.imshow("Output",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
            
    
    


kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
a=Gaussian_kernel(1)
Convolution(img,a)
Gamma(img, 1, 1.0112)
cv2.waitKey(0)
cv2.destroyAllWindows()


 




 


  







