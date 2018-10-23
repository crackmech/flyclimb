#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:52:01 2018

@author: pointgrey
"""

import cv2
import os
import numpy as np
import re
import sys
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import time

import random
import glob
import multiprocessing as mp

'''
from thread import start_new_thread as startNT
import zipfile
import matplotlib.pyplot as plt
from os.path import basename
from scipy.misc import imread
import numpy as np
from datetime import datetime
from functions import your_loss
import re
import time
import cv2
import Tkinter as tk
import tkFileDialog as tkd
import zipfile
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
import os
'''






params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.minThreshold = 5
params.maxThreshold = 120
params.filterByArea = True
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByConvexity = False
params.filterByInertia = False
#params.minArea = 200# 200 for flyClimbing, 1000 for fly walking
params.maxArea = 5000
hcropBox = 100
vcropBox = 100

hcrop = 100
vcrop = 100
crop = 100

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

def createTrack(trackData, img):
    '''
    input:
        create an image of shape 'imgShape' with the x,y coordiates of the track from the array 'trackData
    returns:
        an np.array with the cv2 image array, which can be saved or viewed independently of this function
    '''
    #img = np.ones((imgShape[0], imgShape[1], 3), dtype = 'uint8')
    blue = np.hstack((np.linspace(0, 255, num = len(trackData)/2),np.linspace(255, 0, num = (len(trackData)/2)+1)))
    green = np.linspace(255, 0, num = len(trackData))
    red = np.linspace(0, 255, num = len(trackData))
    cv2.putText(img,'Total frames: '+str(len(trackData)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    for i in xrange(1,len(trackData)):
        cv2.circle(img,(int(trackData[i,0]), int(trackData[i,1])), 2, (blue[i], green[i], red[i]), thickness=2)#draw a circle on the detected body blobs
    for i in xrange(1,len(trackData)):
        if i%100==0:
            cv2.putText(img,'^'+str(i), (int(trackData[i,0]), int(trackData[i,1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    #cv2.imshow('track', img); cv2.waitKey(); cv2.destroyAllWindows()
    return img


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




def getMotion(imData, avg, weight):
    if avg is None:
        print "[INFO] starting background model..."
        avg = imData.copy().astype("float")
    cv2.accumulateWeighted(imData, avg, weight)
    return cv2.absdiff(imData, cv2.convertScaleAbs(avg)), avg


def getBgIm(dirname, imList, imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    
    '''
    avg = np.array((np.median(imgs, axis=2)), dtype = 'uint8')
    return cv2.convertScaleAbs(avg)#cv2.imread(dirname+'/'+imList[0],cv2.IMREAD_GRAYSCALE)

    
def tracknCrop_display(dirname):
    print dirname
    flist = natural_sort(os.listdir(dirname))[:256]
    print len(flist)    
    img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
    imgs = np.zeros((img.shape[0], img.shape[1], len(flist)), dtype = 'uint8')
    for i in xrange(len(flist)):
        imgs[:,:,i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)

    im = []
    bgIm = getBgIm(dirname, flist, imgs)
    for f in range(0, len(flist)):
        im = imgs[:,:,f]
        img = cv2.bitwise_not(cv2.absdiff(im, bgIm))
        eq = (img-img.min())*(255/img.min())
        img = np.vstack((eq, img))
        cv2.imshow('123',img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    return im

def getTrackData(imStack, Blobparams):
    '''
    returns the numpy array of coordinates of the centroid of blob in the stack of images provided as input numpy array 'imStack'
    
    '''
    nFrames = imStack.shape[0]
#    detector = cv2.SimpleBlobDetector_create(Blobparams)
    detector = cv2.SimpleBlobDetector(Blobparams)
    trackData = np.zeros((nFrames,2))
    im = []
    for f in xrange(nFrames):
#        if f%1000==0:
#            sys.stdout.write("\rAt %s Processing File: %d"%(present_time(),f))
#            sys.stdout.flush()
        im = imStack[f]
        keypoints = detector.detect(im)
        kp = None
        try:
            for kp in keypoints:
                trackData[f] = (kp.pt[0],kp.pt[1])
        except:
            pass
    return trackData

def cropImstack(imStack, trackData, hCropbox, vCropbox, cropbox):
    '''
    returns a list of all images, cropped as per cropBox dimensions
    '''
    ims = []
    for i in xrange(imStack.shape[0]):
        im = imStack[i]
        x,y = trackData[i]
        if (x!=0 or y!=0):
            pts = [int(y)-hCropbox, int(y)+hCropbox, int(x)-hCropbox,int(x)+hCropbox]
            im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
            ims.append((i,im_cropped))
        else:
            ims.append((i, 'NoCroppedImage'))
    return ims

def saveCroppedIms(croppedStack, ImStack, saveDir, extension, hCropbox):
    '''
    saves the output of the tracked flies in the given format (specifice by 'extension') in the given directory.
    If a fly is not detected in a continous frame, new folder is created to save the next sequence
    '''
    ext = extension
    n = 0
    outDir = saveDir+str(n)
    cropDir = outDir+'_cropped/'
    imDir = outDir+'_original/'
    os.mkdir(cropDir)
    os.mkdir(imDir)
    for i in xrange(len(croppedStack)):
        if croppedStack[i][1]!='NoCroppedImage':
            if croppedStack[i][1].size == hCropbox*hCropbox*4:
                cv2.imwrite(cropDir+str(i)+ext, croppedStack[i][1])
                cv2.imwrite(imDir+str(i)+ext, ImStack[i])
    return cropDir, imDir






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
    if len(flist)<nImThreshold:
        return []
    print('Total Frames present: %i'%len(flist))
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
        if 'tracked' not in imdir:
            try:
                dirname = os.path.join(imdir, trackedImFolder)
                dirname = imdir
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
                if flyStats!=[]:
                    fps = getFPS(imdir, camFileName, fpsSep, fpsIndex)
                    stats = open(statsFile,'a')
                    stats.write('\n'+(str(flyStats).strip('()')+', '+str(fps)+', '+imdir.lstrip(baseDir)))
                    stats.close()
            except:
                pass
def tracknCrop(dirname, saveExtension, params, nImThreshold):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = natural_sort(os.listdir(dirname))
    print '\nprocessing %i frames in\n==> %s'%(len(flist), dirname)
    if len(flist)<=nImThreshold:
	print('Less Images to process, not processing folder, nImages present: %i'%len(flist))
        pass
    else:
        outDir = dirname+"_tracked/"
        #print outDir
        try:
            os.mkdir(outDir)
        except:
            print 'Tracked directory exists, not processing!!!'
            return
        img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
        imgs = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype = 'uint8')
        startTime1 = time.time()
        for i in xrange(len(flist)):
            imgs[i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)
        startTime2 = time.time()
        print('Read Images in: %0.3f seconds, now tracking'%(startTime2-startTime1))
        trackedData = getTrackData(imStack = imgs, Blobparams = params)
        startTime3 = time.time()
        print('Tracked in: %0.3f seconds, now cropping'%(startTime3-startTime2))
        cropStack = cropImstack(imStack = imgs, trackData = trackedData, hCropbox = hcrop, vCropbox = vcrop, cropbox = crop)
        startTime4 = time.time()
        print('Cropped Images in: %0.3f seconds, now saving'%(startTime4-startTime3))
        cropDir, origDir = saveCroppedIms(croppedStack = cropStack, ImStack = imgs, saveDir = outDir, extension = saveExtension, hCropbox = hcrop)
        startTime5 = time.time()
        print('Saved Images in: %0.3f seconds'%(startTime5-startTime4))
        fname = dirname+"_trackData_"+rawDir
        trackImg = createTrack(trackedData, cv2.imread(dirname+'/'+flist[0]))
        cv2.imwrite(fname+'.jpeg', trackImg)
        np.savetxt(fname+".csv",trackedData, fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')
        print('Processed %i Images in %0.3f seconds\nAverage total processing speed: %05f FPS'\
        %(len(flist), startTime5-startTime1, (len(flist)/(time.time()-startTime1)))) 

        #getOutput(model = inModel, inDir = cropDir.rstrip('/'), outdir = segDir, batchSize = 16)


'''

initialDir = '/media/pointgrey/data/flywalk/temp/temp/'

imgDatafolder = 'imageData'
baseDir = getFolder(initialDir)
#baseDir = initialDir
print baseDir


rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

print "Started processing directories at "+present_time()
for rawDir in rawdirs:
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir, rawDir, imgDatafolder)
    print rawDir
    if 'Walking' in rawDir:
        minArea = 1000# 200 for flyClimbing, 1000 for fly walking
    elif 'Climbing' in rawDir:
        minArea = 200# 200 for flyClimbing, 1000 for fly walking
    params.minArea = minArea# 200 for flyClimbing, 1000 for fly walking
    imdirs = natural_sort([ os.path.join(d, name) for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        dirs = os.path.join(d,imdir)
        t = tracknCrop(os.path.join(d,imdir), '.png', params, nImThreshold)


'''





baseDir = '/media/pointgrey/data/flywalk/temp/temp/'

try:
    baseDir = sys.argv[1]
except:
    baseDir = '/media/pointgrey/data/flywalk/temp/temp/'
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






























































