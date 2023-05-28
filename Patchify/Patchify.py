# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:46:36 2023

@author: Raisul
"""

import numpy as np
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import urllib.request

def unpatchify(patches, image_shape):
    num_patches = len(patches)
    patch_height, patch_width = patches[0].shape[:2]
    height, width = image_shape
    num_patches_per_row = width // patch_width
    
    image = np.zeros((height, width), dtype=patches[0].dtype)
    
    for i in range(num_patches):
        row = i // num_patches_per_row
        col = i % num_patches_per_row
        patch = patches[i]
        image[row * patch_height:(row+1) * patch_height, col * patch_width:(col+1) * patch_width] = patch
    
    return image

def patchify(image, patch_size):
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    patches = []
    
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[i:i+patch_height, j:j+patch_width]
            patches.append(patch)
    
    return patches

# Example usage
image = cv2.imread("car.jpg",cv2.IMREAD_GRAYSCALE)


patch_size = (100, 100)
patch_list = patchify(image, patch_size)

# Print the patches
num=-1
for i, patch in enumerate(patch_list):
    patch = Image.fromarray(patch)
    display(patch)
    num=num+1
    patch.save(f"patch_{num}.jpg")

restored_image=unpatchify(patch_list,image.shape)
restored_image = Image.fromarray(restored_image)
display(restored_image)
