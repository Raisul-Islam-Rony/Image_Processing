# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:47:58 2023

@author: Raisul
"""

import numpy as np

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

# Example usage
patch_size = (2, 2)
patch1 = np.array([[1, 2], [3, 4]])
patch2 = np.array([[5, 6], [7, 8]])
patch3 = np.array([[9, 10], [11, 12]])
patch4 = np.array([[13, 14], [15, 16]])
patch_list = [patch1, patch2, patch3, patch4]

image_shape = (4, 4)
reconstructed_image = unpatchify(patch_list, image_shape)

print("Reconstructed Image:")
print(reconstructed_image)
