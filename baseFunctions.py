#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:18:03 2018

@author: aman
"""
import numpy as np
import os
import glob
import re
import random
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import csv

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    return rows

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]
      








