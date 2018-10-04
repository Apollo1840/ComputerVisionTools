# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 07:25:21 2018

@author: zouco
"""

import os
import cv2
import matplotlib.pyplot as plt

# os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\ComputerVisionTools')

# it works as input for NN to predict result
img = cv2.imread("Yolo/input/sample_office.jpg")
print(img.shape)
print(img[1,:,:])

img =  cv2.imread("Yolo/input/sample_office.jpg", 0)   # gray
img =  cv2.imread("Yolo/input/sample_office.jpg", 1)   # default one, BGR
img =  cv2.imread("Yolo/input/sample_office.jpg", -1)  # including alpha

# ------------------------------------------------------------------------------

# convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# show the picture
plt.imshow(img)

# or
cv2.imshow('image',img)
cv2.waitKey(0) # this is must


# save the picture
plt.imsave('output.png', img)

# or
cv2.imwrite('messigray.png',img)


# excercise
img = cv2.imread('messi5.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27 & 0xFF:         #  0xFF is 64-bits machine wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):        # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------

# draw something on the img matrix
# it works like img = func(img, **params)

scale = max(img.shape[0], img.shape[1])
tl = (0,0)
br = (100,100)

img = cv2.rectangle(img, tl, br, 'r', int(scale/1000))

pos = (50,50) 
label = 'haha'
img = cv2.putText(img, label, pos, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, (0, 225, 0), int(scale/(2000)))


