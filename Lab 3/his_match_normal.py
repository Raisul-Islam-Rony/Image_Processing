# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:44:57 2022

@author: dipto
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dpc


# to plot histogram
def plot_histogram(hist_array, fig_title):
    plt.hist(hist_array.ravel(),L,[0,L-1])
    plt.title(fig_title)
    plt.show()

# to plot array
def plot_data(data_array, fig_title): 
    plt.plot(data_array)
    plt.title(fig_title)
    plt.show()

# to normalize the input image
def min_max_normalize(img_inp, inp_min, inp_max):
    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return img_inp
    
# to count the number of pixel with intensity k
def frequency_count(image):
    each_intensity = np.zeros((L), dtype = int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image.item(i,j)
            each_intensity[x] += 1
    return each_intensity

def get_cdf(pdf_array):
    cdf_array = np.zeros((L), dtype = float)
    trans_array = np.zeros((L), dtype = int)
    cdf_array[0] = pdf_array[0]
    trans_array[0] = round(cdf_array[0]*(L-1))
    for i in range(1,pdf_array.shape[0]):
        cdf_array[i] = cdf_array[i-1] + pdf_array[i]
        trans_array[i] = round(cdf_array[i]*(L-1))
    return cdf_array, trans_array

def hist_spec(img, miu1 = 150, sigma1 = 26,miu2=90,sigma2=14):
    #guassian_distribution (mean=150 & S.D =26)
    guass_func1 = np.random.normal(miu1, sigma1, size=(img.shape[0],img.shape[1]))
    #guassian_distribution (mean=90 & S.D =14)
    guass_func2 = np.random.normal(miu2, sigma2, size=(img.shape[0],img.shape[1]))
    #specified distribution is bimodal guassian distribution
    #construct bimodal distribution from two guassian distribution
    guass_func = np.concatenate((guass_func1,guass_func2), axis=0)
    guass_func = np.round(guass_func).astype(int);
    
    guass_func[guass_func>255]=255
    guass_func[guass_func<0]= 0
    
    # frequency of each intensity 
    freq_input = frequency_count(img)
    freq_guass = frequency_count(guass_func)
    
    # image size
    inp_img_size = img.shape[0]*img.shape[1]
    guass_size = guass_func.shape[0]*guass_func.shape[1]
   
    
    # PDF or normalized histogram
    pdf_input = freq_input/inp_img_size
    pdf_guass = freq_guass/guass_size
    
    # CDF and transformation function calculation 
    cdf_input,trans_input = get_cdf(pdf_input)  
    cdf_guass,trans_guass = get_cdf(pdf_guass) 
    
    # histogram matching
    final_map_input = np.zeros((L), dtype = int)
    final_map_input2 = np.zeros((L), dtype = int)
    for i in range(L):
        for j in range(L):
            if trans_input[i] == trans_guass[j]:
                final_map_input[i] = j
                final_map_input2[i] = j
                break
            elif trans_input[i] < trans_guass[j]:
                prev_dis = trans_input[i] - trans_guass[j-1]
                current_dis = trans_guass[j] - trans_input[i]
                if current_dis<prev_dis:
                    final_map_input[i] = j
                    final_map_input2[i] = j-1
                else:
                    final_map_input[i] = j-1
                    final_map_input2[i] = j
                break
    
    img_new = dpc(img)
    img_new2 = dpc(img)
    for i in range(img_new2.shape[0]):
        for j in range(img_new2.shape[1]):
            img_new[i][j] = final_map_input[img[i][j]]
            img_new2[i][j] = final_map_input2[img[i][j]]
    
    # frequency of each intensity 
    freq_matched_img = frequency_count(img_new)
    # PDF or normalized histogram
    pdf_matched_img = freq_matched_img/inp_img_size
    # CDF and transformation function calculation 
    cdf_matched_img,trans_matched_img = get_cdf(pdf_matched_img)
    return guass_func, cdf_input, cdf_guass, img_new, img_new2, cdf_matched_img

# read the image
img_input=cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
inp_minVal = np.min(img_input)
inp_maxVal = np.max(img_input)


#print("Input min: ",inp_minVal)
#print("Input min: ",inp_maxVal)

# normalize the input image
#img_input = min_max_normalize(img_input, inp_minVal, inp_maxVal)

# type casting
img_input = img_input.astype(np.uint8)
img = dpc(img_input)
plt.imshow(img,'gray')
plt.title('Input Image')
plt.show()
# take inputs - miu and sigma to make guassian filter
guass_miu1 = int(input('Enter value of miu: '))
guass_sigma1 = int(input('Enter value of sigma: '))
guass_miu2 = int(input('Enter value of miu2: '))
guass_sigma2 = int(input('Enter value of sigma2: '))
# maximum value for a pixel
L = 256

# histogram specification output
guass_f, img_cdf, guass_cdf, matched_img, matched2, matched_cdf = hist_spec(img, guass_miu1, guass_sigma1,guass_miu2,guass_sigma2)

# plot the input historam
plot_histogram(img_input, "Input Histogram")
# plot the gaussian histogram
plot_histogram(guass_f, "Bimodal Guassian Function Histogram")
# plot the cdf of input image
#plot_data(img_cdf, "CDF of Input Image")
# plot the cdf of guassian
#plot_data(guass_cdf, "CDF of Guassian Function")
# plot the mapped input historam
plot_histogram(matched_img, "Matched Input Histogram")
# plot the cdf of mapped input image
#plot_data(matched_cdf, "CDF of Matched Input Image")

#cv2.imshow("Input image", img_input)

#cv2.imshow("Histogram matched image", matched_img)
plt.imshow(matched_img,'gray')
plt.title('Histogram matched image')
plt.show()

#cv2.imshow("Histogram matched image2", matched2)
'''plt.imshow(matched2,'gray')
plt.title('Histogram matched image2')
plt.show()'''
cv2.waitKey(0)
cv2.destroyAllWindows()