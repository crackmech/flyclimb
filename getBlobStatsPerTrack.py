#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:27:38 2017

@author: aman
"""

import cv2
import os
import numpy as np
import re
import sys
from datetime import datetime
from thread import start_new_thread as startNT
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
import time
import glob
import random
import multiprocessing as mp

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

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def getFPS(folderName, logFileName, FPSsep, FPSIndex):
    '''
    returns the fps of current image folder by looking up the folder details in
    the camera log file in the parent folder
    '''
    folder = folderName.split(os.sep)[-1].rstrip('_tracked')
    fname = os.path.join(os.path.dirname(os.path.dirname(folderName)),logFileName)
    with open(fname) as f:
        lines = f.readlines()
    for line in lines:
        if all(x in line for x in [folder, 'FPS']):
            return line.split(FPSsep)[FPSIndex]


def getFlyStats(folderName, 
                imExtension,
                blurkernel,
                blk,
                cutOff,
                ellaxisRatioMin,
                ellaxisRatioMax,
                flyareaMin,
                flyareaMax,
                pxSize,
                rounded,
                verbose):
    '''
    get statistics of fly shape from fly images in a folder with tracked flies
    '''

    flist = natural_sort(glob.glob(folderName+'/*'+imExtension))
    #print('Total Frames present: %i'%len(flist))
    colors = [random_color() for c in xrange(100)]
    imgs = 0
    areas = []
    minorAxis = []
    majorAxis = []
    ims = []
    start_time = time.time()
    for fnum in xrange(len(flist)):
        ims.append(cv2.imread(flist[fnum], cv2.IMREAD_GRAYSCALE))
    for fnum in xrange(0,len(flist)):
        ctrs = []
        im = cv2.medianBlur(ims[fnum],blurkernel)
        th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,blk,cutOff)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea)[-10:]
#        [cv2.drawContours(im, cnt, -1, color ,-1) for cnt, color in zip(contours, colors)]
#        cv2.imshow('123',im);
#        cv2.imshow('th',th);
#        cv2.waitKey(1)
        ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
        for cnt in ellRatio:
            if ellaxisRatioMin<cnt[0]<ellaxisRatioMax and flyareaMin<cnt[1]<flyareaMax:
                ctrs.append(cnt[2])
        if ctrs:
            imgs+=1
            M = [cv2.moments(ctrs[i]) for i in xrange(len(ctrs))]
            ellipse = cv2.fitEllipse(ctrs[0])
            areas.append(M[0]['m00'])
            minorAxis.append(ellipse[1][1])
            majorAxis.append(ellipse[1][0])
            if verbose==True:
                im =  cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                [cv2.drawContours(im, cnt, -1, color ,-1) for cnt, color in zip(ctrs, colors)]
                cv2.ellipse(im,ellipse,(0,255,0),2)
                cv2.imshow('123',im);
                cv2.waitKey(1)
                print imgs, fnum, len(flist), len(ctrs),M[0]['m00'],(float(ellipse[1][0])/ellipse[1][1]), \
                        (np.average(np.array(areas))*pxSize), (np.median(np.array(areas))*pxSize), \
                        (np.average(np.array(minorAxis))*pxSize), (np.median(np.array(minorAxis))*pxSize), \
                        (np.average(np.array(minorAxis))*pxSize), (np.median(np.array(minorAxis))*pxSize)
        else:
            pass
    cv2.destroyAllWindows()
    
    averageSize = round(np.average(np.array(areas))*pxSize, rounded)
    averageMinorAxis = round(np.average(np.array(minorAxis))*pxSize, rounded)
    averageMajorAxis = round(np.average(np.array(majorAxis))*pxSize, rounded)
    medianSize = round(np.median(np.array(areas))*pxSize, rounded)
    medianMinorAxis = round(np.median(np.array(minorAxis))*pxSize, rounded)
    medianMajorAxis = round(np.median(np.array(majorAxis))*pxSize, rounded)
    print('Done %i frames in %f seconds (%s, %0.4fFPS)\nfrom %s'%(len(flist), (time.time()-start_time), present_time(), (len(flist)/(time.time()-start_time)), folderName))
    return averageSize, averageMinorAxis, averageMajorAxis, medianSize, medianMinorAxis, medianMajorAxis, len(flist)


def processRawDirs(rawDir):
    if 'Walking' in rawDir:
        params = walkParams
    elif 'Climbing' in rawDir:
        params = climbParams
    flyAreaMin = params['flyAreaMin']
    flyAreaMax = params['flyAreaMax']
    block = params['block']
    cutoff = params['cutoff']
    pixelSize = params['pixelSize'] #pixel size in mm
    
    flist = os.listdir(os.path.join(baseDir, rawDir))
    statsFileName = statsfName+rawDir+statsFileExt
    statsFile = os.path.join(baseDir, rawDir, statsFileName)
    processTime = present_time()
    stats = open(statsFile,'a')
    if statsFileName not in flist:
        stats.write(header)
    stats.write('\n\n'+startString+'(on %s) ===>:flyAreaMin(mm^2): %i, flyAreaMax(mm^2): %i, pixelSize(mm): %0.4f, block: %i, cutoff: %i\n\n'\
                    %(processTime, flyAreaMin, flyAreaMax, pixelSize, block, cutoff))
    stats.close()
    d = os.path.join(baseDir, rawDir, imgDatafolder)
    imdirs = natural_sort([ os.path.join(d, name) for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        if 'tracked' in imdir:
            try:
                dirname = os.path.join(imdir, trackedImFolder)
    #            print dirname
                flyStats = getFlyStats(folderName = dirname, 
                           imExtension = fileExt,
                           blurkernel = blurKernel, 
                           blk = block, 
                           cutOff = cutoff,
                           ellaxisRatioMin = ellAxisRatioMin,
                           ellaxisRatioMax = ellAxisRatioMax,
                           flyareaMin = flyAreaMin,
                           flyareaMax = flyAreaMax,
                           pxSize = pixelSize,
                           rounded = decPoint,
                           verbose = verbose)
    #            print flyStats
                fps = getFPS(imdir, camFileName, fpsSep, fpsIndex)
                stats = open(statsFile,'a')
                stats.write('\n'+(str(flyStats).strip('()')+', '+str(fps)+', '+imdir.lstrip(baseDir)))
                stats.close()
            except:
                pass
#    stats = open(statsFile,'a')
#    stats.write('\n'+endString+' for: %s'%processTime)
#    stats.close()

#baseDir = '/media/aman/data/flyWalk_data/'
baseDir = '/media/pointgrey/data/flywalk/'

baseDir = '/media/flywalk/data/imaging/'

try:
    baseDir = sys.argv[1]
except:
    baseDir = '/media/pointgrey/data/flywalk/'
    baseDir = getFolder(baseDir)
    pass

print baseDir
verbose = False
fileExt = '.png'
decPoint = 5


blurKernel = 5
block = 221
cutoff = 35
flyAreaMin = 100
flyAreaMax = 4000

ellAxisRatioMin = 0.2
ellAxisRatioMax = 0.5

climbParams = {
                'flyAreaMin' : 300,
                'flyAreaMax' : 900,
                'block' : 91,
                'cutoff' : 35,
                'pixelSize' : 0.055 #pixel size in mm
                }

walkParams = {
                'flyAreaMin' : 1100,
                'flyAreaMax' : 5000,
                'block' : 221,
                'cutoff' : 35,
                'pixelSize' : 0.028 #pixel size in mm
                }



headers = ['area_average(mm^2)', 'minorAxis_average(mm)', 'majorAxis_average(mm)',\
            'area_median(mm^2)', 'minorAxis_median(mm)', 'majorAxis_median(mm)' ,\
            'nFrames', 'FPS', 'folderName']


camFileName = 'camloop.txt'
fpsSep = ' ' 
fpsIndex = -2

imgDatafolder = 'imageData'
trackedImFolder = '0_original'
statsfName = 'flyStats_'
statsFileExt = '.csv'


spacer = ''
header = ''
for i in xrange(len(headers)):
    spacer+=','
    header+=headers[i]+', '
    
startString = spacer+'Parameters'
endString = spacer+'Done Processing'

rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])


pool = mp.Pool(processes=8)

print "Started processing directories at "+present_time()
#for rawDir in rawdirs:
#    processRawDirs(rawDir)
##    startNT(processRawDirs, (rawDir,))
_ = [pool.map(processRawDirs, (rawDir,)) for rawDir in rawdirs]

#_ = [pool.apply_async(processRawDirs, (rawDir,)) for rawDir in rawdirs]

#print "Started processing directories at "+present_time()
#for rawDir in rawdirs:
#    processRawDirs(rawDir)
#    startNT(processRawDirs, (rawDir,))
    






'''
im1 = flist[0]
img = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,blurKernel)
ret,th1 = cv2.threshold(img,90,190,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,200,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,block,cutoff)
th3 = cv2.adaptiveThreshold(img,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block,cutoff)
diff1 = cv2.absdiff(img, th2/10)
diff2 = cv2.absdiff(img, th3/10)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'diff1', 'diff2']
images = [img, img, th2, th3, diff1, diff2]

nImg = len(images)
for i in xrange(nImg):
    plt.subplot(nImg/2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
'''






