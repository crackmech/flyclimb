#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:18:50 2018

@author: aman
"""

import cv2
import os
import glob
import numpy as np
import re
import Tkinter as tk
import tkFileDialog as tkd
import random
from datetime import datetime
import time
#import copy
import csv
import sys

import multiprocessing as mp
import itertools


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def readImStack(imfolder, imExt):
    '''
    returns a numpy array of all the images with extension 'imExt' in folder "imFolder"
    '''
    flist = natural_sort(glob.glob(os.path.join(imfolder, '*'+imExt)))
    imStack = []
    for idx, f in enumerate(flist):
        imStack.append(cv2.imread(f))
    return np.array(imStack)

def displayImstack(imStack, delay, winName):
    '''
    displays images from a given imagestack with 'delay'milliseconds
    '''
    for _,im in enumerate(imStack):
        cv2.imshow(winName, im)
        cv2.waitKey(delay)
    cv2.destroyAllWindows()

def getTrackColors(imArray):
    '''
    return the colors to be used for displaying the track
    '''
    blue = np.hstack((np.linspace(0, 255, num = len(imArray)/2),np.linspace(255, 0, num = (len(imArray)/2)+1)))
    green = np.linspace(255, 0, num = len(imArray))
    red = np.linspace(0, 255, num = len(imArray))
    return [c for c in zip(blue, green, red)]
   
def readcontStatsFile(fname):
    with open(fname, "r") as f:
        reader = csv.reader(f)
        return list(reader)



dirname = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male (3rd copy)/imageData/'
csvfname = '20171201_195953_contoursStats_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male (3rd copy).csv'
csvfname = '20171201_200012_contoursStats_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male (3rd copy).csv'
fileName = os.path.join(dirname, csvfname)
cntStats = readcontStatsFile(fileName)[1:]

imFolder = "_".join(csvfname.split('_')[:2])
imfolderPath = os.path.join(dirname, imFolder)

imgExt = '.png'
imgStack =  readImStack(imfolderPath, imgExt)

#displayImstack(imgStack, 4, 'raw')

cntrImStack = np.array([cv2.ellipse(imgStack[i].copy(),
                                    ((float(cntStats[i][1]), float(cntStats[i][2])),
                                     (float(cntStats[i][3]),float(cntStats[i][4])),
                                      float(cntStats[i][5])),
                                     (0,150,255),2) for i in xrange(len(imgStack))])
#displayImstack(cntrImStack, 4, 'cnt')

displayImstack(np.hstack((imgStack,cntrImStack)), 4, 'cnt')









