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
from scipy import stats
import csv

#from datetime import timedelta
#import matplotlib.pyplot as plt
#import xlwt
#import matplotlib

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
      
def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S_%f')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S_%f')
    return (time2-time1).total_seconds()

def getFPS(csvname, logFilePath, FPSsep, FPSIndex):
    '''
    returns the fps of current image folder by looking up the folder details in
    the camera log file in the parent folder
    '''
    csvdetails = csvname.split('_')
    folder = ('_').join([csvdetails[0],csvdetails[1]])
    with open(logFilePath) as f:
        lines = f.readlines()
    fpsValues = []
    for line in lines:
        if all(x in line for x in [folder, 'FPS']):
            fpsValues.append(line.split(FPSsep))
    fpsFrames = [x[FPSIndex-2] for x in fpsValues] # get all lines with required imFolder
    #return the fps from the camloop line with highest number of frames
    return fpsValues[fpsFrames.index(max(fpsFrames))][FPSIndex] 

def getTrackDirection(trackData, minDis):
    '''
    returns a +1 or -1 based on direction of fly movement.
    If the fly walks from left to right  it returns -1 (equivalent to bottom to top for climbing)
    if the fly walks from right to left, it returns +1 (equivalent to top to bottom for climbing)
    Value is calculated purely on the basis of a line fit on the track based on change of X-coordinate w.r.t frames
    '''
    dataLen = len(trackData)
    m,c,r,_,_ = stats.linregress(np.arange(dataLen), trackData[:,0])
    delta = (m*(9*(dataLen/10))+c)-(m*(dataLen/10)+c)
    if delta>=minDis:
        return -1, r
    elif delta<=-minDis:
        return 1, r
    else:
        return 0, r

def getEuDisCenter(pt1, pt2):
    return np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))

def getTotEuDis(xyArr):
    xyArr = np.array(xyArr)
    n = xyArr.shape[0]
    totDis = np.zeros((n-1))
    for i in xrange(0, n-1):
        totDis[i] = getEuDisCenter(xyArr[i], xyArr[i+1])
    return totDis

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points between p1 and p2
    https://stackoverflow.com/questions/43594646/how-to-calculate-the-coordinates-of-the-line-between-two-points-in-python
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])

def diff(x):
    return x[1]-x[0]

def getTrackBreaksPts(xyArray):
    '''
    returns the breakpoints in the XYarray of the centroids of the fly (determined by the '0' in the array)
    '''
    trackStop = 0
    nTracks = []
    for f,x in enumerate(xyArray):
        if x[0]==0 and x[1]==0:
            if trackStop == 0:
                trackStop = f
                nTracks.append([trackStop])
        else:
            if trackStop != 0:
                nTracks[-1].extend([f])
                trackStop = 0
    if trackStop != 0:
        nTracks[-1].extend([f])
    return nTracks

def extrapolateTrack(xyArray, breakPoints, skippedFrThresh, verbose=True):
    '''
    fills the gaps in the xyArray determined by breakPoints, if the gap is less
        than skippedFrThresh
    
    '''
    splitTracks = []
    trackStart = 0
    arrCopy = xyArray.copy()
    for i,x in enumerate(breakPoints):
        if len(x)>1 and diff(x)>0:
            trackBrkLen = diff(x)
            if verbose:
                print trackBrkLen, x
            if trackBrkLen >= skippedFrThresh:
                splitTracks.append([trackStart, x[0]])
                trackStart = x[1]
            else:
                arrCopy[x[0]:x[1],:] = intermediates(xyArray[x[0]-1,:], xyArray[x[1],:], trackBrkLen)
    if (x[0] - trackStart)>=skippedFrThresh:
        splitTracks.append([trackStart, x[0]])
    return splitTracks, arrCopy

pxSize = 70.00/1280     # in mm/pixel
imDataFolder = 'imageData'
headerRowId = 0         # index of the header in the CSV file
fpsSep = ' ' 
fpsIndex = -2
csvHeader = ['trackDetails',
                'track duration (frames)',
                'distance travelled (px)',
                'average instantaneuos speed (px/s)',
                'median instantaneuos speed (px/s)',
                'STD instantaneuos speed (px/s)',
                'average body angle (degrees)',
                'median body angle (degrees)',
                'STD body angle (degrees)',
                'average body length (px)',
                'median body length (px)',
                'STD body length (px)',
                'path straightness (r^2)',
                'geotactic Index',
                'latency (seconds)',
                'FPS',
                'Bin Size (frames)',
                'Pixel Size (um/px)',
                'skipped frames threshold (#frames)',
                'track duration threshold (#frames)',
                ]








