# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:19:51 2023

@author: Raisul
"""
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:59:56 2023

@author: Raisul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.array([[4, 2, 1, 3, 4],
                  [5, 3, 1, 2, 4],
                  [6, 3, 2, 1, 3],
                  [4, 2, 1, 3, 5]])

structure_element= np.ones((3,3))

str_height=structure_element.shape[0]//2
str_width=structure_element.shape[1]//2



output= cv2.copyMakeBorder(src=image, top=str_height, bottom=str_height, left=str_width, right=str_width,borderType= cv2.BORDER_WRAP)

output1= cv2.copyMakeBorder(src=image, top=str_height, bottom=str_height, left=str_width, right=str_width,borderType= cv2.BORDER_WRAP)


(output1)
for i in range(output.shape[0]-structure_element.shape[0]):
    for j in range(output.shape[1]-structure_element.shape[1]):
        mini=1000
        for k in range(structure_element.shape[0]):
            for p in range(structure_element.shape[1]):
                if(output[i+k][j+p]<mini):
                    mini=output[i+k][j+p]
        output1[i+str_height][j+str_width]=mini
        
        
output_res=output1[str_height:(output1.shape[0]-str_height), str_width:(output1.shape[1]-str_width)]
print(output_res)


        


