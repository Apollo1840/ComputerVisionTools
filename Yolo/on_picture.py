# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:24:16 2018

@author: zouco
"""

from darkflow.net.build import TFNet
import cv2
import os
from pprint import pprint
import matplotlib.pyplot as plt
import random

def plot_result(result, img, output_path):
    # inputs are the result from tfnet.return_predict(img), the image from sv2.imread, and the output_path.
    items = set([item['label'] for item in result])
    color_dict = {}
    for item in items:
        color_dict.update({item: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))})
    
    for i in range(len(result)):
        tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
        br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
        label = result[i]['label']
        
        color = color_dict[label]
        lwd = 5
        img = cv2.rectangle(img, tl, br, color, lwd)
        img = cv2.putText(img, label, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, color, lwd)  
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imsave(os.path.join(output_path, 'output.png'), img)
    plt.show()

def summary_result(result):
    items = [item['label'] for item in result]
    summary_dict = {}
    for item in items:
        try:
            summary_dict[item] += 1
        except KeyError:
            summary_dict.update({item: 1})
    return summary_dict
    


def yolo_read_img(image_path, options, output_path):
    tfnet = TFNet(options)
    
    img =  cv2.imread(image_path, cv2.IMREAD_COLOR) # get the BGR image
    # img.shape
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = tfnet.return_predict(img)
    
    '''
    result is like a list of dictionary:
        {'bottomright': {x: .. , y: ..}, 'label': xx, 'confidence': xx , ...}
    
    '''
    
            
    plot_result(result, img, output_path)
    print(summary_result(result))
    
def test_demo():
    wrk_dir = os.getcwd()
    #config InlineBackend.figure_format = 'svg'
    os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\FairyPatrol\\darkflow-master')
    
    options = {
     "model": "cfg/yolo.cfg", 
     "load": "bin/yolov2.weights", 
     "threshold": 0.7,
     "gpu": 0.8}
    
    tfnet = TFNet(options)
    
    imgcv = cv2.imread("./sample_img/sample_office.jpg")
    img =  cv2.imread("./sample_img/sample_office.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    scale = max(img.shape[0], img.shape[1])
    
    result = tfnet.return_predict(imgcv)
    pprint(result)
    
    tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
    br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
    label = result[0]['label']
    
    img = cv2.rectangle(img, tl, br, 'r', int(scale/1000))
    img = cv2.putText(img, label, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 0), int(scale/(2000)))
    plt.imshow(img)
    
    path = os.path.join(wrk_dir, 'results')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.imsave(os.path.join(path, 'output.png'), img)
    
    plt.show()
        

if __name__== "__main__":
    wrk_dir = os.getcwd()
    #config InlineBackend.figure_format = 'svg'
    os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\darkflow-master')
    
    options = {
     "model": "cfg/yolo.cfg", 
     "load": "bin/yolov2.weights", 
     "threshold": 0.3,
     "gpu": 0.8}
    
    pic_path = "C:\\Users\\zouco\\Desktop\\\pyProject\\ComputerVisionTools\\Yolo\\input\\sample_office.jpg"
    output_path =  os.path.join(wrk_dir, 'output')
    
    yolo_read_img(pic_path, options, output_path)
    
    
    
    
