#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 18:56:22 2018

@author: aman
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import re
import os
import cPickle as pickle
from matplotlib import colors
import six
import Tkinter as tk
import tkFileDialog as tkd
import sys




def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def ListFiles(pathName, extension):
    os.chdir (pathName)
    files = []
    for file in glob.glob(extension):
        files.append(file)
    return natural_sort(files)

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'



def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def open_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj




colors_ = list(six.iteritems(colors.cnames))
# Add the single letter colors.
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

# Transform to hex color values.
hex_ = [color[1] for color in colors_]










def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def open_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

