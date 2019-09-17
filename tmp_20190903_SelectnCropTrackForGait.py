#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 03:25:56 2019

@author: aman
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 02:20:00 2018

@author: aman
"""

import cv2
import os
import numpy as np
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import multiprocessing as mp
import time
import glob
import random
import csv
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
    
def getlegTipData(csvFname):
    return np.array(readCsv(csvFname)[1:])

def getCentroidData(csvFname):
    return readCsv(csvFname)[1:]

def imRead(x):
    return cv2.imread(x, cv2.IMREAD_COLOR)
    #return cv2.rotate(cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_COUNTERCLOCKWISE)

def getImStack(flist, pool):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    #startTime = time.time()
    imgStack = pool.map(imRead, flist)
    ims = np.zeros((len(imgStack), imgStack[0].shape[0], imgStack[0].shape[1], imgStack[0].shape[2]), dtype=np.uint8)
    for i,im in enumerate(imgStack):
        ims[i]=im
    imgStack = ims.copy()
    return imgStack

def setColors(trackData):
    blue = np.hstack((np.linspace(0, 255, num = len(trackData)/2),np.linspace(255, 0, num = (len(trackData)/2)+1)))
    green = np.linspace(255, 0, num = len(trackData))
    red = np.linspace(0, 255, num = len(trackData))
    return [(blue[i], green[i], red[i]) for i in xrange(len(trackData))]

def getImStackWithCentroids(flist, centroids, pool):
    imStack = getImStack(flist, pool)
    colors = setColors(centroids)
    for i in xrange(len(imStack)):
        for j in xrange(len(centroids)):
            cv2.circle(imStack[i], (int(centroids[j][0]), int(centroids[j][1])), 2, colors[j], 2)
        for j in xrange(len(centroids)):
            if j%50==0:
                cv2.putText(imStack[i], str(j), (int(centroids[j][0]), int(centroids[j][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    return imStack


def displayImgs(imgs, fps):
    f = 1000/fps
    for i, img in enumerate(imgs):
        cv2.imshow('123',img)
        key = cv2.waitKey(f) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            f = 1000/fps
            cv2.waitKey(0)
        if key == ord("n"):
            cv2.imshow('123',imgs[i+1])
            f=0
            cv2.waitKey(f)
    cv2.destroyAllWindows()


def nothing(x):
    if cv2.getTrackbarPos(addSelectionTrckBar, windowName)==1:
        cv2.setTrackbarPos(addSelectionTrckBar, windowName, 0)
    #pass

def setStartFrame(x):
    global startFrame
    startFrame = cv2.getTrackbarPos(frmBarName, windowName)
    
    return startFrame

def setStopFrame(x):
    global stopFrame
    stopFrame = cv2.getTrackbarPos(frmBarName, windowName)
    
    return stopFrame

def getFinalSelection(x):
    global frameIdx
    if x==1:
        frameIdx.append([startFrame, stopFrame])
        print 'Selected Frames: ',frameIdx
        cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
        cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)

def popLastSelection(x):
    global frameIdx
    if x==1:
        frameIdx.pop(-1)
        print 'Selected Frames: ',frameIdx
        cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
        cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)

def selectTrackFrames(flist, centroids, pool):
    
    '''
    Displays the images from the folder with overlayed centroids on the whole imstack.
    By using the slider, we can determine where to start and stop the track for legTip clustering.
    '''
    global startFrame, stopFrame
    imgs = getImStackWithCentroids(flist, centroids, pool)
    cv2.namedWindow(windowName)
    cv2.moveWindow(windowName, 30,30)
    cv2.createTrackbar(frmBarName, windowName, 0, (len(imgs)-2), nothing)
    cv2.createTrackbar(startFrmTrkBarName, windowName, 0, 1, setStartFrame)
    cv2.createTrackbar(stopFrmTrkBarName, windowName, 0, 1, setStopFrame)
    cv2.createTrackbar(addSelectionTrckBar, windowName, 0, 1, getFinalSelection)
    cv2.createTrackbar(removeSelectionTrckBar, windowName, 0, 1, popLastSelection)
    cv2.setTrackbarPos(frmBarName, windowName, 0)
    cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
    cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)
    cv2.setTrackbarPos(removeSelectionTrckBar, windowName, 0)
    while (1):
        imgNum = cv2.getTrackbarPos(frmBarName, windowName)
        cv2.imshow(windowName,imgs[imgNum] )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyWindow(windowName)

frmBarName = 'frame Number'
startFrmTrkBarName = 'Select Start Frame'
stopFrmTrkBarName = 'Select Stop Frame'
addSelectionTrckBar = 'Add frames'
removeSelectionTrckBar = 'Remove frames'

windowName = 'trackImage'

startFrame = 0
stopFrame = -1
frameIdx = []
#initDir = '/media/aman/data/flyWalk_data/'
#dirname = getFolder(initDir)
#imgExts = ['*.png','*.jpeg']

nThreads = 4

dirName = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'
o = '003656'
fileName = '_legTipsClus_n20-Climbing_'
fileName = '_legTipsClustered-Climbing_'
legTipsCsvName = dirName+'20180822_'+o+fileName+'legTipLocs.csv'
centroidsFile = dirName+'20180822_'+o+fileName+'centroids.csv'


cents1 = getCentroidData(centroidsFile)
cents = [x for _,x in enumerate(cents1) if x[1]!='noContourDetected']
frNamesAll = [x[0] for _,x in enumerate(cents) if x[1]!='noContourDetected']
centroids = np.array(cents)[:,1:].astype(dtype=np.float64)

pool = mp.Pool(nThreads)

selectTrackFrames(frNamesAll, centroids, pool)
print frameIdx

#imgs = getImStackWithCentroids(frNamesAll, centroids, pool)
#displayImgs(imgs, 50)

pool.close()

