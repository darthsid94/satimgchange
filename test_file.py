# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:31:22 2020

@author: Siddharth
"""
#import torch
from PIL import Image
#import imageio
import numpy as np
import cv2
import time
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Image Change Detection App')
st.header('This app accepts a pair of images as input and creates bounding boxes on regions with changes')
uploaded_file1 = st.file_uploader("Choose the first image", type=['txt', 'png','jpg'])
time.sleep(30)
uploaded_file2 = st.file_uploader("Choose the second image", type=['txt', 'png','jpg'],key='2')
time.sleep(20)
file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
print(file_bytes1)
file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
imageA = cv2.imdecode(file_bytes1, 1)
imageB = cv2.imdecode(file_bytes2, 1)

# convert the images to grayscale
A = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
Image.fromarray(A).save('DEMO/new/before.png')
Image.fromarray(B).save('DEMO/new/after.png')
import os
os.system('predict.py -i DEMO -o DEMO -c red -t 0.1')

from skimage import io
#import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
#import numpy as np
img_pred = "DEMO/new/predicted.png"  #predicted change mask
img_before="DEMO/new/before.png"     #before image
before=cv2.imread(img_before,1)
predicted = cv2.imread(img_pred, 1)
b, g, r = cv2.split(predicted) #splitting into components
tmp = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)   #conversion from BGR to graysclae
_,alpha = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY)  #thresholding to extract the alpha channel
#creating a white background and orange foreground for easier superposition on before image
for i in range(650):
    for j in range(650):
        if b[i,j]==255:
            predicted[i,j]=[255,128,0]
        else:
            predicted[i,j]=[255,255,255]


#splitting the transformed predicted image
blue, green, red = cv2.split(predicted)


           
rgba = [blue,green,red, alpha]  
dst = cv2.merge(rgba,4) #creating a four channel image
           
           
io.imshow(predicted)

b_channel,g_channel,r_channel=cv2.split(before)
alpha_channel=np.ones(b_channel.shape, dtype=b_channel.dtype) * 200 # multiple range 0-255. tweak to obtain different results
#performs well when>200
img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # creating a four channel image

fin=img_BGRA+dst #adding the two images

#io.imshow(fin)
st.image([fin],caption=['Change Mask'])