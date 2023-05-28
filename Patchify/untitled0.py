'''import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the blurred image
img = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)

# Estimate the point spread function (PSF) using the Wiener deconvolution method
psf = np.ones((5, 5)) / 25
img_wiener = cv2.filter2D(img, -1, psf)
#psf_estimated, _ = cv2.deconvolve(img_wiener.astype(np.float32), img.astype(np.float32))



plt.title("CAR")
plt.imshow(img)
cv2.imshow("Car",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('deblurred_image.jpg', deconvolved)
'''

import cv2
import numpy as np
from scipy.signal import convolve2d, fftconvolve

# Load the blurred image
img = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)


# Estimate the point spread function (PSF) using the Wiener deconvolution method
psf_estimated = np.ones((5, 5)) / 25


# Apply Richardson-Lucy deconvolution algorithm
kernel = np.ones((3, 3))/9
deconvolved = img.astype(np.float32)
for i in range(30):
    blurred = fftconvolve(deconvolved, psf_estimated, mode='same')
    relative_blur = img / blurred.clip(min=1e-7)
    deconvolved *= fftconvolve(relative_blur, psf_estimated[::-1, ::-1], mode='same')
    cv2.imwrite('deblurred_image.jpg', deconvolved)
    deconvolved = cv2.filter2D(deconvolved, -1, kernel)


# Save the deblurred image
cv2.imshow('deblurred_image.jpg', deconvolved)
cv2.waitKey(0)
cv2.destroyAllWindows()

