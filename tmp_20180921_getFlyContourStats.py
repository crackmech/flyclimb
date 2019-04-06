#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 00:08:24 2018

@author: aman

This script will 
    1)  find the imageData folders which contain fly tracks
    2)  track the fly in the whole image

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

def readIms(dirname, imExts):
    flist = getFiles(dirname, imExts)
    img = cv2.imread(flist[0],cv2.IMREAD_GRAYSCALE)
    imgs = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype = 'uint8')
    for i in xrange(len(flist)):
        imgs[i] = cv2.imread(flist[i],cv2.IMREAD_GRAYSCALE)
    return imgs

def getBgIm(imgs):
    '''
    returns a background Image for subtraction using median values of pixels from stack of all images
    '''
    avg = np.array((np.median(imgs, axis=0)), dtype = 'uint8')
    return cv2.convertScaleAbs(avg)

def getBgSubIms(inImgstack, bgIm):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    subIms = np.zeros(np.shape(inImgstack), dtype=np.uint8)
    for f in range(0, len(inImgstack)):
        subIms[f] = cv2.absdiff(inImgstack[f], bgIm)
    return subIms
    
def getImContours(args):
    '''
    returns the list of detected contour
    input:
        im:  image numpy array to be processed for contour detection
        params: dictionary of all parameters used for contour detection
        params :
            blurkernel
            block
            cutoff
            ellratio
            ellAxisRatioMin
            ellAxisRatioMax
            flyAreaMin
            flyAreaMax
    '''
    im = args[1].copy()
    params = args[2]
    contour = []
    im = cv2.medianBlur(im,params['blurKernel'])
    ret,th = cv2.threshold(im, params['threshLow'], params['threshHigh'],cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #print len(contours)
    try:
        contours = sorted(contours, key = cv2.contourArea)[-10:]
        ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
        for cnt in ellRatio:
            if params['ellaxisRatioMin']<cnt[0]<params['ellaxisRatioMax'] and params['flyareaMin']<cnt[1]<params['flyareaMax']:
                contour.append([args[0], 
                                cv2.fitEllipse(cnt[2]),
                                cv2.contourArea(cnt[2]),
                                args[3]])
            else:
                pass
    except:
        pass
    if contour==[]:
        #print("no contour detected")
        contour.append([args[0],  [], args[3]])
    return contour

def imReadNCnt(args):
    '''
    input: 
        args[0] = path of the image to be read
        args[1] = parameters using which the image will be processed for contour detection
    output:
        im = array of the image read
        cnt = detected contour from the image "im"

    im = cv2.imread(x[0], cv2.IMREAD_GRAYSCALE)
    cnt = getImContours((im, x[1]))
    return [im, cnt]

    '''
    im = cv2.imread(args[0], cv2.IMREAD_GRAYSCALE)
    cnt = getImContours((args[0], im, args[1], args[2]))
    if cnt==[]:
        return [im]
    else:
        return cnt
    
def getFlycontour(dirname, imExts,
                  contourParams, header,
                  workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    params = contourParams
    flist = getFiles(dirname, imExts)
    nImsToProcess = len(flist)
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    pool = mp.Pool(processes=workers)
    startTime = time.time()
    poolArgList = itertools.izip(flist, itertools.repeat(params), np.arange(len(flist)))
    imgWithCnt = pool.map(imReadNCnt, poolArgList)
    t = time.time()-startTime
    print("imRead and Contours detection time for %d frames: %s Seconds at %f FPS\n"%(len(imgWithCnt),t ,len(imgWithCnt)/float(t)))
    contoursList = [header]
    for idx, i in enumerate(imgWithCnt):
        fname = '/'.join( i[0][0].split('/')[-4:])
        if i[0][1] != []:
            coordX = i[0][1][0][0]
            coordY = i[0][1][0][1]
            minorAxis = i[0][1][1][0]
            majorAxis = i[0][1][1][1]
            angle = i[0][1][2]
            area = i[0][2]
            contoursList.append([fname, coordX, coordY, minorAxis, majorAxis, angle, area,])
        else:
            contoursList.append([fname,0,0,0,0,0,0])

    return contoursList


baseDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/'
#baseDir = '/media/pointgrey/data/flywalk/climbingData/uploaded/'
#baseDir = getFolder(baseDir)

try:
    baseDir = sys.argv[1]
except:
    baseDir = '/media/pointgrey/data/flywalk/'
    baseDir = getFolder(baseDir)
    pass
os.chdir(baseDir)

imExtensions = ['*.png', '*.jpeg']
imDataFolder = 'imageData'
statsfName = 'contoursStats'
statsFileHeader = ['frameDetails','x-coord','y-coord','minorAxis (px)','majorAxis (px)','angle','area (px)']

nThreads = 4
#nImThresh = 100

params = {} # dict for holding parameter values for contour detection
params['blurKernel'] = 5
params['block'] = 123
params['cutoff'] = 15


params['threshLow'] = 
params['threshHigh'] = 
params['ellaxisRatioMin'] = 0.2
params['ellaxisRatioMax'] = 0.5
params['flyareaMin'] = 300 
params['flyareaMax'] = 900 


rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

for idx, rawDir in enumerate(rawdirs):
    dirs = natural_sort([ os.path.join(rawDir, imDataFolder, name) for name in os.listdir(os.path.join(rawDir, imDataFolder)) if os.path.isdir(os.path.join(rawDir, imDataFolder, name)) ])
    for _,imFolder in enumerate(dirs):
        if 'tracked' not in imFolder:
            flyContours = getFlycontour(imFolder, imExtensions, params, statsFileHeader, nThreads)
            statsFile = imFolder.rstrip('/')+'_'+statsfName+'_'+rawDir+'.csv'
            with open(statsFile, "w") as f:
                writer = csv.writer(f)
                writer.writerows(flyContours)





























