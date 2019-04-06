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

def readImStack(flist):
    '''
    returns a numpy array of all the images with extension 'imExt' in folder "imFolder"
    '''
    img = cv2.imread(flist[0], cv2.IMREAD_GRAYSCALE)
    imStack = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx, f in enumerate(flist):
        imStack[idx] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return imStack

def imRead(x):
    return cv2.imread(x, cv2.IMREAD_GRAYSCALE)

def getBgIm(imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    '''
    avg = np.array((np.median(imgs, axis=0)), dtype = np.uint8)
    return cv2.convertScaleAbs(avg)

def getBgSubIms(inImgstack, bgIm):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    subIms = np.zeros(np.shape(inImgstack), dtype = np.uint8)
    for f in range(0, len(inImgstack)):
        subIms[f] = cv2.absdiff(inImgstack[f], bgIm)
    return subIms
    

def displayImgs(imgs, fps):
    for _, img in enumerate(imgs):
        cv2.imshow('123',img)
        cv2.waitKey(1000/fps)
    cv2.destroyAllWindows()


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
    blur = cv2.GaussianBlur(im,(params['blurKernel'], params['blurKernel']),0)
    ret,th = cv2.threshold(blur, params['threshLow'], params['threshHigh'],cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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
        contour.append([args[0],  [], args[3]])
    return contour

    
def getFlycontour(dirname, imExts,
                  contourParams, header,
                  pool):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    params = contourParams
    flist = getFiles(dirname, imExts)
    nImsToProcess = len(flist)
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    startTime = time.time()
    #imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
    imgStack = readImStack(flist)
    bgIm = getBgIm(imgStack)
    subIms = getBgSubIms(imgStack, bgIm)
    poolArgList = itertools.izip(flist, subIms, itertools.repeat(params), np.arange(len(flist)))
    imgWithCnt = pool.map(getImContours, poolArgList)
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



#baseDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/'
baseDir = '/media/pointgrey/data/flywalk/climbingData/uploaded/'
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
statsfName = 'contoursStats_threshBinary'
statsFileHeader = ['frameDetails','x-coord','y-coord','minorAxis (px)','majorAxis (px)','angle','area (px)']

nThreads = 1
pool = mp.Pool(processes=nThreads)

rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
for idx, rawDir in enumerate(rawdirs):
    dirs = natural_sort([ os.path.join(rawDir, imDataFolder, name) for name in os.listdir(os.path.join(rawDir, imDataFolder))\
                        if os.path.isdir(os.path.join(rawDir, imDataFolder, name)) ])
    for _,imFolder in enumerate(dirs):
        if 'tracked' not in imFolder:
            flyContours = getFlycontour(imFolder, imExtensions, params, statsFileHeader, pool)
            statsFile = imFolder.rstrip('/')+'_'+statsfName+'_'+rawDir+'.csv'
            with open(statsFile, "w") as f:
                writer = csv.writer(f)
                writer.writerows(flyContours)








#initialDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male/imageData/'
#dirName = getFolder(initialDir)
#
#imExtensions = ['*.png', '*.jpeg']
#
#pool = mp.Pool(processes=nThreads)
#
#flist = getFiles(dirName, imExtensions)
#start = time.time()
#imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
#print("\nimRead time for %s core: %.5s Seconds\n"%(nThreads,(time.time()-start)))
#
#start = time.time()
#bgIm = getBgIm(imgStack)
#print('time used for calculating Bg image = %f'%(time.time()-start))
#start = time.time()
#subIms = getBgSubIms(imgStack, bgIm)
#print('time used for subtracting Images = %f'%(time.time()-start))
#
#cntIms = subIms.copy()
#colCntImgs = []
#noCnt = []
#for i,img in enumerate(cntIms):
#    colImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    blur = cv2.GaussianBlur(img,(params['blurKernel'], params['blurKernel']),0)
#    ret,thresh = cv2.threshold(blur, params['threshLow'], params['threshHigh'],cv2.THRESH_BINARY)
#    cv2.imshow('123',thresh)
#    cv2.waitKey(1)
#    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    try:
#        contours = sorted(contours, key = cv2.contourArea)[-10:]
#        ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
#        for cnt in ellRatio:
#            if params['ellaxisRatioMin']<cnt[0]<params['ellaxisRatioMax'] and params['flyareaMin']<cnt[1]<params['flyareaMax']:
#                cv2.drawContours(colImg, [cnt[2]], 0, (0,255,0), 1)
#                colCntImgs.append(colImg)
#    except:
#        noCnt.append(i)
#cv2.destroyAllWindows()
#
#print noCnt
#displayImgs(colCntImgs, 110)


