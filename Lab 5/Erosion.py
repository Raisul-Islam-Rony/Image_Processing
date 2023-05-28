# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:59:56 2023

@author: Raisul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("binary.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input",image)

structure_element= np.ones((15,15))

str_height=structure_element.shape[0]//2
str_width=structure_element.shape[1]//2

print(image.shape)

output= cv2.copyMakeBorder(src=image, top=str_height, bottom=str_height, left=str_width, right=str_width,borderType= cv2.BORDER_WRAP)

output1= cv2.copyMakeBorder(src=image, top=str_height, bottom=str_height, left=str_width, right=str_width,borderType= cv2.BORDER_WRAP)


for i in range(output.shape[0]-structure_element.shape[0]):
    for j in range(output.shape[1]-structure_element.shape[1]):
        mini=-1000
        for k in range(structure_element.shape[0]):
            for p in range(structure_element.shape[1]):
                if(output[i+k][j+p]>mini):
                    mini=output[i+k][j+p]
        output1[i+str_height][j+str_width]=mini
   

output_res=output1[str_height:(output1.shape[0]-str_height), str_width:(output1.shape[1]-str_width)]

print(output_res.shape)
cv2.imshow("output",output_res)
cv2.waitKey(0)
cv2.destroyAllWindows()