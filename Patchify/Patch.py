import numpy as np
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import urllib.request


image = Image.open("car.jpg")

image = np.asarray(image)
plt.imshow(image)
patches = patchify(image, (100, 100, 3), step=100)
print(patches.shape)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j,0]
        
        patch = Image.fromarray(patch)
        display(patch)
      
        
        num = i * patches.shape[1] + j
        patch.save(f"patch_{num}.jpg")
'''
image_height, image_width, channel_count = image.shape
patch_height, patch_width, step = 100, 100, 3
output_patches = np.empty(patches.shape).astype(np.uint8)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]
        output_patch = patch  # process the patch
        output_patches[i, j, 0] = output_patch

# merging back patches
output_height = image_height - (image_height - patch_height) % step
output_width = image_width - (image_width - patch_width) % step
output_shape = (output_height, output_width, channel_count)
output_image = unpatchify(output_patches, output_shape)
output_image = Image.fromarray(output_image)
output_image.save("output.jpg")
'''