#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:06:33 2018

@author: aman
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:46:57 2017

@author: aman
"""
import cv2
import os
import numpy as np
import re
import Tkinter as tk
import tkFileDialog as tkd
import random
from datetime import datetime
import copy
from thread import start_new_thread as startNT


import time
from os.path import basename
import glob
from scipy.misc import imread

nImThreshold = 100# if number of images in a folder is less than this, then the folder is not processed

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

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def getTrackColors(imArray):
    '''
    return the colors to be used for displaying the track
    '''
    blue = np.hstack((np.linspace(0, 255, num = len(imArray)/2),np.linspace(255, 0, num = (len(imArray)/2)+1)))
    green = np.linspace(255, 0, num = len(imArray))
    red = np.linspace(0, 255, num = len(imArray))
    return [c for c in zip(blue, green, red)]
   
def getFlycontour(dirname, 
                blurkernel,
                blk,
                cutOff,
                ellaxisRatioMin,
                ellaxisRatioMax,
                flyareaMin,
                flyareaMax):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    imgs=[]
    ims = []
    allContours = []
    flist = natural_sort(os.listdir(dirname))
    print '\nprocessing %i frames in\n==> %s'%(len(flist), dirname)
    if len(flist)<=nImThreshold:
	print('Less Images to process, not processing folder, nImages present: %i'%len(flist))
        pass
    else:
        img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
        ims = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype = 'uint8')
        for i in xrange(len(flist)):
            ims[i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)
        ims = np.fliplr(np.transpose(ims, (0,2,1)))
        for fnum in xrange(len(flist)):
            ctrs = []
            im = cv2.medianBlur(ims[fnum],blurkernel)
            th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,blk,cutOff)
            im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key = cv2.contourArea)[-10:]
            ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
            for cnt in ellRatio:
                if ellaxisRatioMin<cnt[0]<ellaxisRatioMax and flyareaMin<cnt[1]<flyareaMax:
                    ctrs.append(cnt[2])
            if ctrs:
                imgs.append(im)
                allContours.append(cv2.fitEllipse(ctrs[0]))
            else:
                pass
    return imgs, allContours

def displayMovie(imageArray, contourList, saveDir, displayIm = True, saveIm = False, tFPS = 1):
    '''
    displays the movie of the fly with the tracked contours, center and trajectory covered along with the raw data
    '''
    gapIm = np.ones((imageArray[0].shape[0],10, 3), dtype='uint8')*255
    colors = getTrackColors(imageArray)
    if saveIm:
        try:
            os.mkdir(saveDir)
        except:
            pass
    for i, img in enumerate(imageArray):
        im =  cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = im.copy()
        cnt = contourList[i]
        rect = copy.deepcopy(cnt)
        rect = (rect[0],(1, rect[1][1]), rect[2])
        cntRect = cv2.boxPoints(rect)
        cntRect = np.int0(cntRect)
        if i!=0:
            for j in range(i):
                if j%100==0:
                    cv2.putText(im, str(j), (int(contourList[j][0][0]),int(contourList[j][0][1])+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),2)
                try:
                    cv2.circle(im,(int(contourList[j][0][0]),int(contourList[j][0][1])), 2, colors[j], thickness=2)#draw a circle on the detected body blobs
                except:
                    pass
        cv2.drawContours(im,[cntRect],0,(255,255,255),2)
        cv2.ellipse(im,cnt,(0,150,255),2)
        img = np.hstack((img, gapIm, im))
        if saveIm:
            cv2.imwrite(os.path.join(saveDir, str(i)+'.png'), img)
        if displayIm:
            cv2.imshow('FlyWithEllipse',img)#np.flipud(np.transpose(im, axes=1)));
            cv2.waitKey(tFPS)
    cv2.destroyAllWindows()

def saveIms(imageArray, contourList, saveDir):
    '''
    saves the images from which the contour is detected into the saveDir
    '''
    


initialDir = '/media/aman/data/flyWalk_data/'
saveFolder = '/media/aman/data/flyWalk_data/tracked/'



imgDatafolder = 'imageData'
baseDir = getFolder(initialDir)
rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])


climbParams = {
                'flyAreaMin' : 300,
                'flyAreaMax' : 900,
                'block' : 91,
                'cutoff' : 35,
                'pixelSize' : 0.055 #pixel size in mm
                }
params = climbParams

ellAxisRatioMin = 0.2
ellAxisRatioMax = 0.5
verbose = True
fileExt = '.png'
decPoint = 5


blurKernel = 5
block = 221
cutoff = 35
flyAreaMin = params['flyAreaMin']
flyAreaMax = params['flyAreaMax']
block = params['block']
cutoff = params['cutoff']
pixelSize = params['pixelSize'] #pixel size in mm

def getTrackedIms(dirname):
    saveFolder = dirname+'_tracked'
    images, flyContours = getFlycontour(dirname,
                                blurkernel = blurKernel, 
                                blk = block, 
                                cutOff = cutoff,
                                ellaxisRatioMin = ellAxisRatioMin,
                                ellaxisRatioMax = ellAxisRatioMax,
                                flyareaMin = flyAreaMin,
                                flyareaMax = flyAreaMax)
    print ("done tracking, now displaying")
    displayMovie(images, flyContours, saveFolder, True, False, tFPS = 10)
    
print "Started processing directories at "+present_time()
for rawDir in rawdirs:
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir, rawDir, imgDatafolder)
    imdirs = natural_sort([ os.path.join(d, name) for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        getTrackedIms(imdir)
        # startNT(getTrackedIms, (imdir,))







