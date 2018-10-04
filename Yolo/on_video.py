# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:51:17 2018

@author: zouco
"""

from darkflow.net.build import TFNet
import cv2
import os
from pprint import pprint
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import time

from on_picture import YoloCV


class YoloCV_V():
    
    def __init__(self, options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.2, "gpu": 0.8}):
        
        if isinstance(options, str):
            with open(options,'r',encoding = 'utf-8') as f:
                options = json.load(f)
        
        wrk_dir = os.getcwd()
        os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\darkflow-master')
        self.tfnet_ = TFNet(options)
        os.chdir(wrk_dir)
    
    def get_result_video(self, video_path):
        capture = cv2.VideoCapture(video_path)
        
        max_num_rec = 5
        colors = [tuple(255 * np.random.rand(3)) for i in range(max_num_rec)]
        
        while (capture.isOpened()):
            t0 = time.time()
            ret, frame = capture.read()  # ret is True or False, stands for the video is playing or not
            if ret:
                results = self.tfnet_.return_predict(frame)                
                for color, result in zip(colors, results):
                    frame = YoloCV.draw_rec(frame, result, color)                    
                cv2.imshow('frame', frame)                
                print('FPS {:.1f}'.format(1 / (time.time() - t0)))
            
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1 ms for stop command
                    break
            else:
                capture.release()
                cv2.destroyAllWindows()
                break

if __name__== "__main__":
    wrk_dir = os.getcwd()
    #config InlineBackend.figure_format = 'svg'
    #os.chdir('C:\\Users\\zouco\\Desktop\\\pyProject\\darkflow-master')
    
    yc = YoloCV_V()
    yc.get_result_video('C:\\Users\\zouco\\Desktop\\\pyProject\\PicVideoForCV\\sample1.avi')
    
    
    
    
