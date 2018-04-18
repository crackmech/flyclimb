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
import time
import random
from datetime import datetime
import copy

from os.path import basename
import glob
from scipy.misc import imread


params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.minThreshold = 5
params.maxThreshold = 120
params.filterByArea = True
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 1000
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
    detector = cv2.SimpleBlobDetector_create(Blobparams)
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

def tracknCrop(dirname, saveExtension, params, nImThreshold):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    outDir = dirname+"_tracked/"
    #print outDir
    try:
        os.mkdir(outDir)
    except:
        pass
    flist = natural_sort(os.listdir(dirname))
    print '\nprocessing %i frames in\n==> %s'%(len(flist), dirname)
    if len(flist)<=nImThreshold:
	print('Less Images to process, not processing folder, nImages present: %i'%len(flist))
        pass
    else:
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
   

saveFolder = '/media/aman/data/flyWalk_data/tracked/'



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

def displayMovie(imageArray, contourList, saveDir, display = True, save=False, tFPS = 1):
    '''
    displays the movie of the fly with the tracked contours, center and trajectory covered along with the raw data
    '''
    gapIm = np.ones((imageArray[0].shape[0],10, 3), dtype='uint8')*255
    colors = getTrackColors(imageArray)
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
        if save:
            cv2.imwrite(os.path.join(saveDir, str(i)+'.png'), img)
        if display:
            cv2.imshow('FlyWithEllipse',img)#np.flipud(np.transpose(im, axes=1)));
            cv2.waitKey(tFPS)
    cv2.destroyAllWindows()



def displayFlycontour(dirname, 
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
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    imgs=0
    areas = []
    minorAxis = []
    majorAxis = []
    ims = []
    colors = [random_color() for c in xrange(100)]
    flist = natural_sort(os.listdir(dirname))
    print '\nprocessing %i frames in\n==> %s'%(len(flist), dirname)
    if len(flist)<=nImThreshold:
	print('Less Images to process, not processing folder, nImages present: %i'%len(flist))
        pass
    else:
        img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
        ims = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype = 'uint8')
        imBlack = np.zeros((10, img.shape[1],3), dtype = 'uint8')
        for i in xrange(len(flist)):
            ims[i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)
        for fnum in xrange(len(flist)):
            ctrs = []
            im = cv2.medianBlur(ims[fnum],blurkernel)
            img = cv2.imread(dirname+'/'+flist[fnum])
            th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,blk,cutOff)
            im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key = cv2.contourArea)[-10:]
            ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
            for cnt in ellRatio:
                if ellaxisRatioMin<cnt[0]<ellaxisRatioMax and flyareaMin<cnt[1]<flyareaMax:
                    ctrs.append(cnt[2])
            if ctrs:
                imgs+=1
                ellipse = cv2.fitEllipse(ctrs[0])
                print ellipse[0], ellipse
                if verbose==True:
                    im =  cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                    cv2.circle(im,(int(ellipse[0][0]),int(ellipse[0][1])), 2, (0,255,255), thickness=2)#draw a circle on the detected body blobs
                    # [cv2.drawContours(im, cnt, -1, color ,-1) for cnt, color in zip(ctrs, colors)]
                    cv2.ellipse(im,ellipse,(0,255,0),2)
                    im = np.flipud(np.transpose(np.vstack((img,imBlack, im)), axes=(1,0,2)))
                    # cv2.imwrite(saveFolder+str(fnum)+'.png', im)
                    cv2.imshow('FlyWithEllipse',im) #np.flipud(np.transpose(im, axes=1)));
                    cv2.waitKey(1)
            else:
                pass
        cv2.destroyAllWindows()


# displayFlycontour(os.path.join(d,imdir),
#                 imExtension = fileExt,
#                 blurkernel = blurKernel, 
#                 blk = block, 
#                 cutOff = cutoff,
#                 ellaxisRatioMin = ellAxisRatioMin,
#                 ellaxisRatioMax = ellAxisRatioMax,
#                 flyareaMin = flyAreaMin,
#                 flyareaMax = flyAreaMax,
#                 pxSize = pixelSize,
#                 rounded = decPoint,
#                 verbose = verbose)




initialDir = '/media/aman/data/flyWalk_data/'

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
    try:
        os.mkdir(saveFolder)
    except:
        pass
    images, flyContours = getFlycontour(dirname,
                                blurkernel = blurKernel, 
                                blk = block, 
                                cutOff = cutoff,
                                ellaxisRatioMin = ellAxisRatioMin,
                                ellaxisRatioMax = ellAxisRatioMax,
                                flyareaMin = flyAreaMin,
                                flyareaMax = flyAreaMax)
    print ("done tracking, now displaying")
    displayMovie(images, flyContours, saveFolder, False, True, tFPS = 10)
    
from thread import start_new_thread as startNT

print "Started processing directories at "+present_time()
for rawDir in rawdirs:
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir, rawDir, imgDatafolder)
    imdirs = natural_sort([ os.path.join(d, name) for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        startNT(getTrackedIms, (imdir,))







