# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:24:16 2018

@author: zouco
"""

from darkflow.net.build import TFNet
import cv2
import os
from pprint import pprint
import json
import matplotlib.pyplot as plt
import random




class YoloCV():
    
    def __init__(self, options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.7, "gpu": 0.8}):
        
        if isinstance(options, str):
            with open('... .txt','r',encoding = 'utf-8') as f:
                options = json.load(f)
        
        wrk_dir = os.getcwd()
        os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\darkflow-master')
        self.tfnet_ = TFNet(options)
        os.chdir(wrk_dir)
    

    def run(self, pic_path, output_path):
        self.get_output(pic_path, output_path)
        plt.imshow(self.img_output_)
        pprint(self.result_)
        print(self.result_summary_)
        
        
    def get_output(self, pic_path, output_path):
        self.get_result(pic_path)
        self.process_result()
        self.save_output(output_path)
        
    
    def get_result(self, pic_path):
        
        '''
                result is like a list of dictionary:
                    {'bottomright': {x: .. , y: ..}, 'label': xx, 'confidence': xx , ...}
    
        '''
        
        
        self.img_ = cv2.cvtColor(cv2.imread(pic_path), cv2.COLOR_BGR2RGB)
        self.result_ = self.tfnet_.return_predict(self.img_)
    
    def process_result(self):
        # add rectangles by self.result_
        
        result = self.result_
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
            img = cv2.rectangle(self.img_, tl, br, color, lwd)
            img = cv2.putText(img, label, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, color, lwd)  
            
            self.img_output_ = img
        
    def save_output(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.imsave(os.path.join(output_path, 'output.png'), self.img_output_)
    
    @property
    def result_summary_(self):
        items = [item['label'] for item in self.result_]
        summary_dict = {}
        for item in items:
            try:
                summary_dict[item] += 1
            except KeyError:
                summary_dict.update({item: 1})
        return summary_dict
        

if __name__== "__main__":
    wrk_dir = os.getcwd()
    #config InlineBackend.figure_format = 'svg'
    #os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\darkflow-master')
    
    options = {
     "model": "cfg/yolo.cfg", 
     "load": "bin/yolov2.weights", 
     "threshold": 0.3,
     "gpu": 0.8}
    
    pic_path = "input\\sample_office.jpg"
    output_path = 'output'
    
    yc = YoloCV(options)
    yc.run(pic_path, output_path)
    
    
    
    
