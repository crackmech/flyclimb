#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 01:30:46 2018

@author: aman
"""


import cv2
import os
import glob
import numpy as np
import re
import Tkinter as tk
import tkFileDialog as tkd
from datetime import datetime
import time
import csv
import sys
import multiprocessing as mp
import itertools


params = {} # dict for holding parameter values for contour detection
params['threshLow'] = 45
params['threshHigh'] = 255

params['ellaxisRatioMin'] = 0.2
params['ellaxisRatioMax'] = 0.7
params['flyareaMin'] = 200 
params['flyareaMax'] = 2100 

params['blurKernel'] = 11

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

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)


def displayImgs(imgs, fps):
    for _, img in enumerate(imgs):
        cv2.imshow('123',img)
        cv2.waitKey(1000/fps)
    cv2.destroyAllWindows()


def imRead(x):
    return cv2.imread(x, cv2.IMREAD_GRAYSCALE)

def getBgIm(imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    '''
    avg = np.array((np.median(imgs, axis=0)))
    return cv2.convertScaleAbs(avg)

def getBgSubIms(inImgstack, bgIm):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    subIms = np.zeros(np.shape(inImgstack), dtype=np.uint8)
    for f in range(0, len(inImgstack)):
        subIms[f] = cv2.bitwise_not(cv2.absdiff(inImgstack[f], bgIm))
    return subIms
    
    
def getSubIms(dirname, imExts, workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = getFiles(dirname, imExts)
    nImsToProcess = len(flist)
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    pool = mp.Pool(processes=workers)
    startTime = time.time()
    imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
    bgIm = getBgIm(imgStack)
    subIms = getBgSubIms(imgStack, bgIm)
    t = time.time()-startTime
    print("imRead and bg Subtraction time for %d frames: %s Seconds at %f FPS\n"%(len(flist),t ,len(flist)/float(t)))
    return imgStack, subIms



baseDir = '/media/aman/data/flyWalk_data/'
baseDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp'
#baseDir = '/media/pointgrey/data/flywalk/climbingData/uploaded/'
#baseDir = getFolder(baseDir)

try:
    baseDir = sys.argv[1]
except:
    #baseDir = '/media/pointgrey/data/flywalk/'
    baseDir = getFolder(baseDir)
    pass
os.chdir(baseDir)

imExtensions = ['*.png', '*.jpeg']
imDataFolder = 'imageData'


outDir = baseDir+'../bg/'
nThreads = 4




imStack, subImStack = getSubIms(baseDir, imExtensions, nThreads)

kernelSize = 3
nIterations = 2
kernel = np.ones((kernelSize,kernelSize),np.uint8)



for _, img in enumerate(subImStack):
    _,th = cv2.threshold(cv2.GaussianBlur(img, (kernelSize,kernelSize), 1), 250,255,cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    erosion = cv2.erode(th,kernel,iterations = nIterations)
    dilation = cv2.dilate(erosion, kernel, iterations = nIterations)
    cv2.imshow('tmp', np.transpose(np.array((np.bitwise_xor(th, dilation), np.bitwise_xor(th, dilation)/2, img/255), dtype=np.uint8), axes=(1,2,0)))
    cv2.waitKey(1)
cv2.destroyAllWindows()
    
for _, img in enumerate(subImStack):
    _,th = cv2.threshold(cv2.GaussianBlur(img, (kernelSize,kernelSize), 1), 250,255,cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    erosion = cv2.erode(th,kernel,iterations = nIterations)
    dilation = cv2.dilate(erosion, kernel, iterations = nIterations)
    cv2.imshow('tmp', np.bitwise_xor(th, dilation))
    cv2.waitKey(1)
cv2.destroyAllWindows()
    
#_,th = cv2.threshold(cv2.GaussianBlur(subImStack[100].copy(), (kernelSize,kernelSize), 1), 250,255,cv2.THRESH_BINARY)
#cv2.imshow('tmp', th)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#
#
#cv2.imshow('tmp', dilation)
#cv2.waitKey(0)
#cv2.destroyAllWindows()





#img = imStack[100].copy()#[230:430,120:320]
##img = subImStack[100].copy()[230:430,120:320]
#_,th = cv2.threshold(cv2.GaussianBlur(img, (kernelSize,kernelSize), 1), 120,255,cv2.THRESH_BINARY)
#th = cv2.bitwise_not(th)
#gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel)
#sobelx = cv2.Sobel(gradient,cv2.CV_64F,1,0,ksize=kernelSize)
#sobely = cv2.Sobel(gradient,cv2.CV_64F,0,1,ksize=kernelSize)
#erosion = cv2.erode(th,kernel,iterations = 1)
#dilation = cv2.dilate(th, kernel, iterations = 1)
#cv2.imshow('tmp', np.vstack((np.bitwise_xor(erosion, dilation), img, gradient, th, erosion)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('tmp', np.vstack((sobelx, sobely, np.bitwise_xor(erosion, dilation))))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#
#img = imStack[100].copy()#[230:430,120:320]
#_,th = cv2.threshold(cv2.GaussianBlur(img, (kernelSize,kernelSize), 1), 120,255,cv2.THRESH_BINARY)
#th = cv2.bitwise_not(th)
#erosion = cv2.erode(th,kernel,iterations = 2)
#dilation = cv2.dilate(erosion, kernel, iterations = 2)
#cv2.imshow('tmp', np.vstack((np.bitwise_xor(th, dilation), th, img)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#









