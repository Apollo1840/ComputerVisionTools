# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:32:31 2018

@author: zouco
"""

import os
# os.listdir(), os.scandir()

import urllib.request as ulib
# ulib.Request(), ulib.urlopen(), ulib.urlretrieve(link, savepath)


from bs4 import BeautifulSoup as Soup
import json
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
# from generate_xml import write_xml

from IPython import get_ipython

import time
# -------------------------------------------------------------
def save_images(search_name):
    directory = search_name.replace(' ', '_')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    links = get_links(search_name)
    for i, link in enumerate(links):
        savepath = os.path.join(directory, '{:06}.png'.format(i))
        ulib.urlretrieve(link, savepath)
        
def get_links(search_name):
    search_name = search_name.replace(' ', '+')
    
    # searh page
    url_a = 'https://www.google.com/search?ei=1m7NWePfFYaGmQG51q7IBg&hl=en&q={}'
    
    # page number (no use)
    url_b = '\&tbm=isch&ved=0ahUKEwjjovnD7sjWAhUGQyYKHTmrC2kQuT0I7gEoAQ&start={}'
    
    # other information
    url_c = '\&yv=2&vet=10ahUKEwjjovnD7sjWAhUGQyYKHTmrC2kQuT0I7gEoAQ.1m7NWePfFYaGmQG51q7IBg'
    url_d = '\.i&ijn=1&asearch=ichunk&async=_id:rg_s,_pms:s'
    
    url_base = ''.join((url_a, url_b, url_c, url_d))

    url = url_base.format(search_name, 0)
    new_soup = from_url_to_soup(url)
    images = new_soup.find_all('img')
    links = [image['src'] for image in images]
    return links  

def from_url_to_soup(url):
    headers = {'User-Agent': 'Chrome/41.0.2228.0 Safari/537.36'}
    json_string = ulib.urlopen(ulib.Request(url, None, headers)).read()
    page = json.loads(json_string)
    return Soup(page[1][1], 'lxml')    
# -------------------------------------------------------------  
                
def combine_all_folders_in_one(keyword):
    imdir = keyword + '_combined'
    if not os.path.isdir(imdir):
        os.mkdir(imdir)
    
    key_folders = [folder for folder in os.listdir('.') if keyword in folder]
    
    n = 0
    for folder in key_folders:
        for imfile in os.scandir(folder):
            os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
            n += 1
# -------------------------------------------------------------
def demo_aerial_image():
    save_images('aerial image car')
    save_images('aerial image street')
    save_images('aerial image parking')
    combine_all_folders_in_one('aerial')
    
# -------------------------------------------------------------
img = None
tl_list = []
br_list = []
object_list = []
    
# constants
image_folder = 'aerial_combined'
savedir = 'annotations'
obj = 'car'    
    
         
def line_select_callback(clk, rls):
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)


def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    global next_step
    if event.key == 'q':
        print(object_list, tl_list, br_list)
        # write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        tl_list = []
        br_list = []
        object_list = []
        img = None
        plt.close()
        next_step = 1


def toggle_selector(event):
    toggle_selector.RS.set_active(True) 


if __name__ == '__main__':  
    get_ipython().run_line_magic('matplotlib', 'qt')
    all_images = list(enumerate(os.scandir(image_folder)))
    next_step = 1
    n = 0
    while next_step == 1 and n < len(all_images):
        next_step = 0
        image_file = all_images[n][1]
        img = image_file
        n += 1
        fig, ax = plt.subplots(1)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(250, 120, 1280, 1024)
        image = cv2.imread(image_file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        toggle_selector.RS = RectangleSelector(
                ax, line_select_callback,
                drawtype='box', useblit=True,
                button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=True
            )
        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', onkeypress)
        
        
    
    
    
    
'''    
    for n, image_file in enumerate(os.scandir(image_folder)):    
            img = image_file
            fig, ax = plt.subplots(1)
            # mngr = plt.get_current_fig_manager()
            # mngr.window.setGeometry(250, 120, 1280, 1024)
            image = cv2.imread(image_file.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            toggle_selector.RS = RectangleSelector(
                ax, line_select_callback,
                drawtype='box', useblit=True,
                button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=True
            )
            bbox = plt.connect('key_press_event', toggle_selector)
            key = plt.connect('key_press_event', onkeypress)
 '''       
