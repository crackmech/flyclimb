#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:43:42 2018

@author: aman
"""

# import cv2
import os
import glob
import numpy as np
import re
import random
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
from scipy import stats
#import xlwt
import csv
import dip
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.markers import MarkerStyle
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlecol = 1
col=1
plt.rcParams['axes.facecolor'] = (col,col,col)

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

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S')
    return (time2-time1).total_seconds()

csvExt = ['*.csv']
imgDatafolder = 'imageData'

pixelSize =0.055
avFlyBodyLen = 2.5# get fly body length in mm
blu = int(avFlyBodyLen/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)

disMovedThresh  = 1*blu     # in BodyLenghtUnit (BLU), minimum displacement of the fly to start consider fly movement as climbing
trackThresh     = 5*blu     # in BLU. Minimum distance oved by fly in a track to be considered as a track


initDir = '/media/aman/data/flyWalk_data'
baseDir = getFolder(initDir)
csvList = getFiles(baseDir, csvExt)



def readCSV(csvFName):
    '''
    returns the data of the csv file as a list of 
    '''
    with open(csvFName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csvData = []
        count = 0
        for row in csv_reader:
            if count==0:
                for _,x in enumerate(row):
                    csvData.append([x])
            else:
                for i,x in enumerate(row):
                    csvData[i].append(x)
            count+=1
    return csvData

a = readCSV(csvList[2])

def getTrackFromCSV(csvfileData):
    '''
    function to extract a list of tracks from the input csvData. number of tracks can be one or more
        each element of the list of tracks (i.e. each track) will have same details as in the CSV file
    '''
































