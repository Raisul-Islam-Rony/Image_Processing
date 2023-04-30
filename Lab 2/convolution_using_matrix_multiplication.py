import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import toeplitz

def toeplitz_matrix(a,b):
    a=np.array(a)
    b=np.array(b)
    a=a.reshape(-1,1)
    height=a.shape[0]
    width=b.shape[0]
    mat=np.zeros((height,width),np.float32)
    mat[:,0]=a[:,0]
    mat[0,1:width]=b[1:width]
    for i in range(1,width):
        mat[1:height,i]=mat[0:height-1,i-1]
    return mat



    
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)


kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])
print(kernel)

image=cv2.resize(image,(200,200))

image_h=image.shape[0]
image_w=image.shape[1]
kernel_h=kernel.shape[0]
kernel_w=kernel.shape[1]
output_h=image_h+kernel_h-1
output_w=image_w+kernel_w-1

kernel_padded=np.pad(kernel,((output_h-kernel_h,0),(0,output_w-kernel_w)),'constant',constant_values=0)


toeplitz_list=[]
for i in range(kernel_padded.shape[0]-1,-1,-1):
    c=kernel_padded[i,:]
    r=np.r_[c[0],np.zeros(image_w-1,np.float32)]
    teoplitz_temp=toeplitz_matrix(c,r)
    toeplitz_list.append(teoplitz_temp)

c=range(1,kernel_padded.shape[0]+1)
r=np.r_[c[0],np.zeros(image_h-1,dtype=int)]
double_indices=toeplitz(c,r)



toeplitz_h=kernel_padded.shape[1]
toeplitz_w=image_w
toeplitz_block_h=toeplitz_h*double_indices.shape[0]
toeplitz_block_w=toeplitz_w*double_indices.shape[1]
toeplitz_block=np.zeros((toeplitz_block_h,toeplitz_block_w),np.float32)

for i in range(double_indices.shape[0]):
    for j in range(double_indices.shape[1]):
        start_h=i*toeplitz_h
        star_w=j*toeplitz_w
        end_h=start_h+toeplitz_h
        end_w=star_w+toeplitz_w
        toeplitz_block[start_h:end_h,star_w:end_w]=toeplitz_list[double_indices[i][j]-1]



out_vector=np.zeros(image.shape[0]*image.shape[1],np.float32)
image2=np.flipud(image)
out_vector=image2.reshape(-1,1)


result_vector=np.matmul(toeplitz_block,out_vector)


output=np.zeros((output_h,output_w),np.float32)
for i in range(output.shape[0]):
    start=i*output_w
    end=start+output_w
    output[i,:]=result_vector[start:end,0]
output=np.flipud(output)

fig,axs= plt.subplots(1,2,figsize=(10,7))
axs[0].imshow(image,'gray')
axs[1].imshow(output,'gray')
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('input image')
axs[1].set_title('output image')
plt.show()




