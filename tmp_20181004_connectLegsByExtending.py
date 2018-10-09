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
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import multiprocessing as mp
import time
import glob
#from tracker_cm import Tracker
##import sys
#import matplotlib.pyplot as plt
import trackpy as tp
import random
import csv
import itertools




flyParams = cv2.SimpleBlobDetector_Params()
flyParams.blobColor = 0
flyParams.minThreshold = 5
flyParams.maxThreshold = 240#120   120 for original image, 250 for bg subtracted images
flyParams.filterByArea = True
flyParams.filterByCircularity = True
flyParams.minCircularity = 0
flyParams.filterByConvexity = False
flyParams.filterByInertia = False
flyParams.minArea = 200# 200 for flyClimbing, 1000 for fly walking
flyParams.maxArea = 8000

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(flyParams)
else : 
	detector = cv2.SimpleBlobDetector_create(flyParams)



nImThreshold = 0# if number of images in a folder is less than this, then the folder is not processed
imgDatafolder = 'imageData'



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

def random_color():
    levels = range(0,255,32)
    return tuple(random.choice(levels) for _ in range(3))

colors = [(0,200,200),(200,0,200),(200,200,0),(150,0,0),(0,0,200),(200,200,255)]
colors = [random_color() for x in xrange(1000)]
#csvOutFile = '/media/aman/data/thesis/colorPalette.csv'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(colors)

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



def getTrackData(imStack, Blobparams, blurParams):
    '''
    returns the numpy array of coordinates of the centroid of blob in the stack of images provided as input numpy array 'imStack'
    
    '''
    nFrames = imStack.shape[0]
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        	detector = cv2.SimpleBlobDetector(Blobparams)
    else : 
        	detector = cv2.SimpleBlobDetector_create(Blobparams)
    trackData = np.zeros((nFrames,2))
    kernel, sigma = blurParams
    for f in xrange(nFrames):
        im = imStack[f]
        keypoints = detector.detect(cv2.GaussianBlur(im, (kernel, kernel), sigma))
        kp = None
        try:
            for kp in keypoints:
                trackData[f] = (kp.pt[0],kp.pt[1])
        except:
            pass
    return trackData

def getContours((idx, im, contourParams, blurParams)):
    kernel, sigma = blurParams
    ret,th = cv2.threshold(cv2.GaussianBlur(im, (kernel,kernel), sigma), contourParams['threshLow'], contourParams['threshHigh'],cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else : 
        im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    try:
        contours = max(contours, key = cv2.contourArea)
#        if not contourParams['minCntArea']<=cv2.contourArea(x)<=contourParams['maxCntArea']:
#            contours = []
    except:
        contours = []
    return [idx, contours]

def getContourData(imStack, fList, contourParams, blurParams, pool):
    '''
    returns the ellipse fit data of the fly in the stack of images provided as input numpy array 'imStack'  
    '''
#    imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
#    poolArgList = itertools.izip(flist, itertools.repeat(params), np.arange(len(flist)))
#    imgWithCnt = pool.map(imReadNCnt, poolArgList)
    poolArgList = itertools.izip(fList, imStack, itertools.repeat(contourParams), itertools.repeat(blurParams))
    contours = pool.map(getContours, poolArgList)
    trackData = []
    for idx,cnt in enumerate(contours):
        if len(cnt[1])!=0:
            try:
                trackData.append([cnt[0], cv2.fitEllipse(cnt[1])])
            except:
                print ('no contour detected in frame# %s'%cnt[0])
                trackData.append([cnt[0], 'noContourDetected'])
        else:
            print ('no contour detected in frame# %s'%cnt[0])
            trackData.append([cnt[0], 'noContourDetected'])
    return trackData
#    trackData = []
#    for idx, im in enumerate(imStack):
#        frId = flist[idx]
#        contours = getContours((frId, im, contourParams, blurParams))
#        if len(contours[1])!=0:
#            trackData.append([contours[0], cv2.fitEllipse(contours[1])])
#        else:
#            print ('no contour detected in frame# %d'%frId)
#            trackData.append([contours[0], 'noContourDetected'])
#    cv2.destroyAllWindows()
#    return trackData

def cropImstack(imStack, trackData, heightCropbox, widthCropbox, blurParams, ratTailParams):
    '''
    returns a list of all images, cropped as per cropBox dimensions
    '''
    kernel, sigma = blurParams
    thresh, nIterations, erodeKernel = ratTailParams
    ims = []
    for i in xrange(imStack.shape[0]):
        im = imStack[i]
        try:
            x,y = trackData[i]
            if (heightCropbox<=y<=imStack.shape[1]-heightCropbox and widthCropbox<=x<=imStack.shape[2]-widthCropbox):
                pts = [int(y)-heightCropbox, int(y)+heightCropbox, int(x)-widthCropbox,int(x)+widthCropbox]
                im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
                _,th = cv2.threshold(cv2.GaussianBlur(im_cropped, (kernelSize,kernelSize), sigma), thresh, 255,cv2.THRESH_BINARY)
                th = cv2.bitwise_not(th)
                erosion = cv2.erode(th,erodeKernel,iterations = nIterations)
                dilation = cv2.dilate(erosion, erodeKernel, iterations = nIterations)
                ims.append([i,np.bitwise_xor(th, dilation)])
            else:
                 ims.append([i, 'NoCroppedImage'])
        except:
            pass
    return ims

def cropImstackGray(imStack, trackData, heightCropbox, widthCropbox):
    '''
    returns a list of all images, cropped as per cropBox dimensions
    '''
    ims = []
    for i in xrange(imStack.shape[0]):
        im = imStack[i]
        try:
            x,y = trackData[i]
            if (heightCropbox<=y<=imStack.shape[1]-heightCropbox and widthCropbox<=x<=imStack.shape[2]-widthCropbox):
                pts = [int(y)-heightCropbox, int(y)+heightCropbox, int(x)-widthCropbox,int(x)+widthCropbox]
                im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
                ims.append([i,im_cropped])
            else:
                 ims.append([i, 'NoCroppedImage'])
        except:
            pass
    return ims

def saveCroppedIms(croppedStack, ImStack, saveDir, extension, hCropbox):
    '''
    saves the output of the tracked flies in the given format (specifice by 'extension') in the given directory.
    If a fly is not detected in a continous frame, new folder is created to save the next sequence
    '''
    ext = extension
    outDir = saveDir
    cropDir = outDir+'_cropped/'
    imDir = outDir+'_original_subIms/'
    os.mkdir(imDir)
    os.mkdir(cropDir)
    for i in xrange(len(croppedStack)):
        if 'NoCroppedImage' not in croppedStack[i][1]:
            cv2.imwrite(cropDir+str(i)+ext, croppedStack[i][1])
            cv2.imwrite(imDir+str(i)+ext, ImStack[i])
        else:
            print i, croppedStack[i][1]
    return cropDir, imDir

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)


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


def imRead(x):
    return cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    #return cv2.rotate(cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_COUNTERCLOCKWISE)

def getBgIm(imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    '''
    avg = np.array((np.median(imgs, axis=0)))
    return cv2.convertScaleAbs(avg)

def getBgSubImStack((inImgstack, bgIm)):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    subIms = np.zeros(np.shape(inImgstack), dtype=np.uint8)
    for f in range(0, len(inImgstack)):
        subIms[f] = cv2.bitwise_not(cv2.absdiff(inImgstack[f], bgIm))
    return subIms
    
def getBgSubIm((inImg, bgIm)):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    return cv2.bitwise_not(cv2.absdiff(inImg, bgIm))    
    
def getSubIms(dirname, imExts, pool, workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = getFiles(dirname, imExts)
    nImsToProcess = len(flist)
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    startTime = time.time()
    imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
    t1 = time.time()-startTime
    print("imRead time for %d frames: %s Seconds at %f FPS"%(len(flist),t1 ,len(flist)/float(t1)))
    t1 = time.time()
    imStackChunks = np.array_split(imgStack, 4*workers, axis=1)
    imStackChunks = [x.copy() for x in imStackChunks if x.size > 0]
    bgImChunks = pool.map(getBgIm, imStackChunks)
    bgIm = np.array(np.vstack((bgImChunks)), dtype=np.uint8)
    t2 = time.time()-t1
    print("bg calculation time for %d frames: %s Seconds at %f FPS"%(len(flist),t2 ,len(flist)/float(t2)))
    t2 = time.time()
    subIms = np.array(pool.map(getBgSubIm, itertools.izip(imgStack, itertools.repeat(bgIm))), dtype=np.uint8)
    t = time.time()-t2
    print("bg Subtraction time for %d frames: %s Seconds at %f FPS"%(len(flist),t ,len(flist)/float(t)))
    return imgStack, subIms, flist

def getEuDisCenter((x1,y1)):
    return np.sqrt(np.square(x1-heightCrop)+np.square(y1-widthCrop))

def getEuDisCorner((x1,y1)):
    return np.sqrt(np.square(x1)+np.square(y1))

def getFarPoint(cnt):
    '''
    returns the coordinates of the far most point w.r.t to the origin
    '''
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    disSorted = sorted([leftmost, rightmost, topmost, bottommost], key=getEuDisCenter)
    return disSorted

def tracknCrop(dirname, imgExt, heightcrop, widthcrop, contourParams, outFname, \
               params, nImThreshold, blurParams, ratTailParams, pool, workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = natural_sort(os.listdir(dirname))
    if len(flist)<=nImThreshold:
	print('Less Images to process, not processing folder, nImages present: %i'%len(flist))
        pass
    else:
        imgs, subImgs, flist = getSubIms(dirname, imgExt, pool, workers)
        startTime1 = time.time()
#        trackedData = getTrackData(imStack = subImgs, Blobparams = params, blurParams=blurParams)
#        blobXYs = trackedData
        trackedData = getContourData(imStack = subImgs, fList = flist, contourParams= contourParams, blurParams=blurParams, pool=pool)
        print trackedData[1], '\n',trackedData[0][1][0]
        blobXYs = [x[1][0] for _,x in enumerate(trackedData)]
        startTime3 = time.time()
        print('Tracked in: %0.3f seconds, now cropping'%(startTime3-startTime1))
        cropSubImStack = cropImstack(imStack = subImgs, trackData = blobXYs, heightCropbox = heightcrop, widthCropbox = widthcrop,\
                     blurParams=blurParams, ratTailParams=ratTailParams)
        cropImStack = cropImstackGray(imStack = imgs, trackData = blobXYs, heightCropbox = heightcrop, widthCropbox = widthcrop)
        startTime4 = time.time()
        print('Cropped Images in: %0.3f seconds, now saving'%(startTime4-startTime3))
        startTime5 = time.time()
        print('Saved Images in: %0.3f seconds'%(startTime5-startTime4))
#        fname = dirname.rstrip('/')+"_trackData_"
        with open(outFname+".csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerow(['frame','((X-Coord, Y-Coord), (minorAxis, majorAxis), angle)'])
            writer.writerows(trackedData)
        #np.savetxt(fname+".csv",trackedData, fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')
        print('Processed %i Images in %0.3f seconds\nAverage total processing speed: %05f FPS'\
        %(len(flist), startTime5-startTime1, (len(flist)/(time.time()-startTime1)))) 
        return cropImStack, cropSubImStack, flist

def getLegTipLocs(rawDir, trackParams, legContourThresh, outFname, pool):
    
    imExts, height, width, cntparams, \
    flyparams, nImThresh, blurParams, ratTailparams = trackParams
    croppedImStack, croppedSubImStack, fList = tracknCrop(rawDir, imExts, height,\
                                                   width, cntparams, outFname, flyparams,\
                                                   nImThresh, blurParams, ratTailparams,\
                                                   pool)
    croppedSubIms = []
    croppedIms = []
    for i in xrange(len(croppedSubImStack)):
        if 'NoCroppedImage' not in croppedSubImStack[i][1]:
            croppedSubIms.append(croppedSubImStack[i][1])
            croppedIms.append(croppedImStack[i][1])
    croppedIms = np.array(croppedIms, dtype=np.uint8)
    croppedSubIms = np.array(croppedSubIms, dtype=np.uint8)
    allLocs = []
    for i, im in enumerate(croppedSubIms):
        _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in sorted(contours, key = cv2.contourArea)[-6:] if cv2.contourArea(x)>=legContourThresh]
        locs = []
        for j,cnt in enumerate(contours):
            locs.append(getFarPoint(cnt)[-1])
        allLocs.append(np.array(sorted([x for x in locs], key=getEuDisCorner)))
    return allLocs, croppedIms

def getAllLocs(rawDir, trackParams, legContourThresh, outFname, pool, workers):
    
    imExts, height, width, cntparams, \
    flyparams, nImThresh, blurParams, ratTailparams = trackParams
    croppedImStack, croppedSubImStack, fList = tracknCrop(rawDir, imExts, height,\
                                                   width, cntparams, outFname, flyparams,\
                                                   nImThresh, blurParams, ratTailparams,\
                                                   pool, workers)
    croppedSubIms = []
    croppedIms = []
    for i in xrange(len(croppedSubImStack)):
        if 'NoCroppedImage' not in croppedSubImStack[i][1]:
            croppedSubIms.append(croppedSubImStack[i][1])
            croppedIms.append(croppedImStack[i][1])
    croppedIms = np.array(croppedIms, dtype=np.uint8)
    croppedSubIms = np.array(croppedSubIms, dtype=np.uint8)
    #displayImgs(croppedIms, 100)
    #displayImgs(croppedSubIms, 100)
    allLocs = []
    allConts = []
    for i, im in enumerate(croppedSubIms):
        _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in sorted(contours, key = cv2.contourArea)[-8:] if cv2.contourArea(x)>=legContourThresh]
        locs = []
        for j,cnt in enumerate(contours):
            locs.append([getFarPoint(cnt)[0],getFarPoint(cnt)[-1]])
        #allLocs.append(np.array(sorted([x for x in locs], key=getEuDisCorner)))
        allLocs.append(locs)
        allConts.append(contours)
    return allLocs, croppedIms, croppedSubIms, allConts

def assignLegTips(tipLocs, pxMvmntThresh, frmSkipThresh, saveFileName, crpImStack):
    t = tp.link_iter(tipLocs, search_range = pxMvmntThresh, memory=frmSkipThresh)    #iterator of locations, distance moved between frames, memory of skipped frame
    trackedIds = []
    for idx,x in enumerate(t):
        trackedIds.append(x[1])
    legTips = [['frame#','x','y','trackId']]
    for i,loc in enumerate(tipLocs):
        for j,l in enumerate(loc):
            legTips.append([i, l[0], l[1],trackedIds[i][j]])
    
    csvOutFile = saveFileName+'.csv'
    with open(csvOutFile, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(legTips)
    
    legTipsFr = [['frame#',\
                  'x','y','trackId',\
                  'x','y','trackId',\
                  'x','y','trackId',\
                  'x','y','trackId',\
                  'x','y','trackId',\
                  'x','y','trackId']]
    for i,loc in enumerate(tipLocs):
        frLocs = [i]
        for j,l in enumerate(loc):
            frLocs.extend((l[0], l[1],trackedIds[i][j]))
        legTipsFr.append(frLocs)
    csvOutFile = saveFileName+'_FramesTogether.csv'
    with open(csvOutFile, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(legTipsFr)
    
    dispIms = []
    for i, im in enumerate(crpImStack):
        img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        locs = tipLocs[i]
        for j,loc in enumerate(locs):
            cv2.circle(img, tuple(loc), 2, colors[trackedIds[i][j]], 2)
            cv2.putText(img, str(trackedIds[i][j]), tuple(loc), cv2.FONT_HERSHEY_COMPLEX, 0.4, colors[trackedIds[i][j]])
        dispIms.append(img)
    return trackedIds, dispIms

    

initialDir = '/media/pointgrey/data/flywalk/'
initialDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/pythonTmp/'
initialDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/copiedLegTrackingTrackData/'
baseDir = getFolder(initialDir)

outDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'

imExtensions = ['*.png', '*.jpeg']
heightCrop = 80
widthCrop = 100
legCntThresh = 3

nThreads = 4
kernelSize = 5
gauBlurParams = (kernelSize,1)

threshVal = 250
nIterations = 2
kernel = np.ones((kernelSize,kernelSize),np.uint8)

pxMvdByLegBwFrm = 50
legTipFrmSkipthresh = 60

rattailparams = (threshVal, nIterations, kernel)
#baseDir = initialDir
print baseDir
cntParams = {'maxCntArea'   :   7000,\
             'minCntArea'   :   2000,\
             'threshLow'    :   230,\
             'threshHigh'   :   255}


trackparams = [imExtensions, heightCrop, widthCrop, cntParams, flyParams,\
               nImThreshold, gauBlurParams, rattailparams]


rawDirs = getDirList(baseDir)
pool = mp.Pool(processes=nThreads)
procStartTime = time.time()
totalNFrames = 0


imdir = baseDir
#imdir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/20180822_003656/'


startTime = time.time()
nFrames = len(getFiles(imdir, imExtensions))
fname = imdir.rstrip(os.sep)+'_legTips-Climbing_allPts_'
legTipLocs = getAllLocs(imdir, trackparams, legCntThresh, fname, pool, nThreads)
allLocs, croppedIms, croppedSubIms, contours = legTipLocs
extrmLocs = [np.array([y[-1] for y in x]) for x in allLocs]
#LegTipsAssigned, \
#trackedCrpImStack = assignLegTips(tipLocs = extrmLocs, pxMvmntThresh = pxMvdByLegBwFrm,\
#                                 frmSkipThresh = legTipFrmSkipthresh, saveFileName = fname, \
#                                 crpImStack = croppedIms)
#displayImgs(trackedCrpImStack,5)
#
#vidObj = cv2.VideoWriter(fname+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2*widthCrop,2*heightCrop))
#for _, im in enumerate(trackedCrpImStack):
#    vidObj.write(im)
#vidObj.release()
print('Processed %i frames in %0.3f seconds\nAverage total processing speed: %05f FPS'\
%(nFrames, time.time()-startTime, (nFrames/(time.time()-startTime)))) 
totalNFrames +=nFrames

pool.close()
totSecs = time.time()-procStartTime
print('Processing finished at: %05s, in %sSeconds, total processing speed: %05f FPS'\
      %(present_time(),totSecs , totalNFrames/totSecs))


displayImgs(croppedIms,111)
#

from sklearn import cluster
allVerts = np.vstack((extrmLocs))
X = allVerts.copy()
default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 7}
params = default_base.copy()

spectral = cluster.SpectralClustering(
    n_clusters=params['n_clusters'], eigen_solver='arpack',
    affinity="nearest_neighbors")
spectral.fit(X)
y_pred = spectral.labels_.astype(np.int)

labels = y_pred
blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
for i,v in enumerate(allVerts):
    blIm[v[1],v[0]] = colors[labels[i]]
cv2.imshow("Original", blIm)
key = cv2.waitKey(0)
cv2.destroyAllWindows()


allVertsList = [list(x) for _,x in enumerate(allVerts)]
frLegTipLabels = []
for i, tips in enumerate(extrmLocs):
    ltlabels = []
    for j, tip in enumerate(tips):
        ltlabels.append(labels[allVertsList.index(list(tip))])
    frLegTipLabels.append(ltlabels)

blIms = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in croppedIms.copy()]
for idx,im in enumerate(blIms):
    for j, pt in enumerate(extrmLocs[idx]):
        cv2.circle(im, tuple(pt), 2, colors[frLegTipLabels[idx][j]+10], thickness=3)

displayImgs(blIms,10)

allLocs1 = [np.hstack((np.zeros((len(x),1))+i,np.arange(len(x),0, -1).reshape((len(x),1)), x)) for i,x in enumerate(locs)]
allLocs1 = np.vstack((allLocs1))
allLocs1 = np.hstack((allLocs1, np.reshape(y_pred, (len(y_pred),1))))
outData = []
for i in xrange(len(allLocs1)):
    frData = [allFrLabels[i]]
    for _,x in enumerate(allLocs1[i]):
        frData.extend(x)
    outData.append(frData)
#    outData = [[allFrLabels[i],allLocs1[i]] for i in xrange(len(allLocs1))]
csvOutFile = fname+'_legTipLocs.csv'
with open(csvOutFile, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(outData)





#blIms = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#cnt = contours[0][-1]
#
#for _,c in enumerate(cnt):
#    print c[0]
#    blIms[c[0][1],c[0][0]] = 255
#for i,cnt in enumerate(contours[0]):
#    cv2.fillPoly(blIms, contours[0], colors[i])
#cv2.imshow("Original", blIms)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cntThres = 2
#blIms = np.zeros((len(extrmLocs), 2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for i,cnt in enumerate(contours):
#    cnt = [x for x in cnt if cv2.contourArea(x)>=cntThres]
#    cv2.drawContours(blIms[i], cnt, -1, colors[0], thickness = -1)
#displayImgs(blIms,5)
#
#
#
#
#blIms = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)  for x in croppedIms.copy()]
#radPx = 1
#for idx,im in enumerate(blIms):
#    for j,pt in enumerate(extrmLocs[idx]):
#        #im[pt[1]-radPx:pt[1]+radPx,pt[0]-radPx:pt[0]+radPx] = [100,255,200]
#        cv2.line(im, tuple(pt[0]), (widthCrop, heightCrop), colors[j], thickness=2)
#        cv2.line(im, tuple(pt[0]), tuple(pt[1]), colors[j], thickness=3)
#displayImgs(blIms,5)
#
#
#blIms = np.zeros((len(extrmLocs), 2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#radPx = 1
#for idx,im in enumerate(blIms):
#    for j,pt in enumerate(extrmLocs[idx]):
#        im[pt[1]-radPx:pt[1]+radPx,pt[0]-radPx:pt[0]+radPx] = [100,255,200]
#        #cv2.line(im, tuple(pt), (widthCrop, heightCrop), (255,255,255), thickness=3)
#
#displayImgs(blIms,100)
#
#
#
#img_array = []
#orig_array = []
#
#background = cv2.createBackgroundSubtractorKNN()
#
#for idx,frame in enumerate(croppedIms):
#    orig_array.append(frame)
#    frame = background.apply(frame)  
#    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(frame, (7,7), 0)
#    img_array.append(blur)
#    
#X = np.array(img_array)
#
## We need this only for displaying the result
#O = np.array(orig_array)
#t=time.time()
#q = np.fft.fftn(X)
#
## Compute the phase angle
#angle = np.arctan2(q.imag, q.real)
#
## Compute phase spectrum array from q
#phase_spectrum_array = np.exp(1j*angle)
#reconstructed_array = np.fft.ifftn(phase_spectrum_array)
#t1 = time.time()-t
#print t1
#
#for i in range(0,O.shape[0]):
#    # Smooth the frame using the averaging filter
#    frame = abs(reconstructed_array[i])
#    
#    filteredFrame = cv2.GaussianBlur(frame, (13,13), 0)
#    
#    # Convert the frame into binary image using mean value as threshold
#    mean_value = np.mean(filteredFrame)
#    
#    # median_value = np.median(filteredFrame)
#    ret, binary_frame = cv2.threshold(filteredFrame, 1.6*mean_value, 255, cv2.THRESH_BINARY)
#    
#    # Denoise the binary_frame
#    npbinary = np.array(binary_frame, dtype = np.uint8)
#    # denoised = cv2.fastNlMeansDenoising(src=npbinary, h=120, templateWindowSize=7, searchWindowSize=21)
#    
#    # Perform morphological operations
#    # kernel = np.ones((13,13), np.uint8)
#    
#    # closing = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
#    # opening = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
#                
#    (_, cnts, _) = cv2.findContours(npbinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#    #cv2.drawContours(O[i], cnts, -1, (0,255,0), 2 )
#
#    #cv2.drawContours(denoised, cnts, -1, (157,192,12), -1)
#    # cv2.drawContours(O[i], cnts, -1, (0, 255, 0), 3)
#	 # Loop over the contours
#    for c in cnts:
#		 # If the contour is too small, ignore it
#        if cv2.contourArea(c) < 1:
#            continue
#		# Compute the bounding box for the contour, draw it on the frame,
#		# and update the text
#        (x, y, w, h) = cv2.boundingRect(c)
#        cv2.rectangle(O[i], (x, y), (x + w, y + h), (0, 255, 0), 2)
#        # cv2.rectangle(denoised, (x, y), (x + w, y + h), (255, 255, 0), 2)
#    
#    #cv2.imshow("Denoised", denoised)
#    cv2.imshow("Binary", binary_frame)
#    # cv2.imshow("Opening", opening)
#    cv2.imshow("Original", O[i])
#    key = cv2.waitKey(20) & 0xFF
#    if key == ord("q"):
#        break
#cv2.destroyAllWindows()
#
#
#displayImgs(croppedIms,100)
#
#
#def getOccList(ids):
#    '''
#    input: a list of detections from every frame
#    output: a list of number of occurence of each trackedId in ascending order
#    '''
#    aa = []
#    for i,x in enumerate(ids):
#        aa.extend(x)
#    aaSet = set(aa)
#    occ = [[x,aa.count(x)] for x in aaSet]
#    return sorted(occ, key=max)
#occList = getOccList(LegTipsAssigned)
#   
#
#
#SlImgs = blIms.transpose(1,0,2,3)
#displayImgs(SlImgs,100)
#
#
#cv2.imshow('sliceIm', SlImgs[60])
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#
#blIms = np.zeros((len(extrmLocs), 2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for idx,im in enumerate(blIms):
#    pt = extrmLocs[idx][0]
#    cv2.line(im, tuple(pt), (widthCrop, heightCrop), (255,255,255), thickness=3)
#SlImgs = blIms.transpose(1,0,2,3)
#cv2.imshow('sliceIm', cv2.convertScaleAbs(np.sum(SlImgs[:70], axis=0)))
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#def random_color():
#    levels = range(0,110,3)
#    return tuple(random.choice(levels) for _ in range(3))
#
#colors = [random_color() for x in xrange(50)]
#nLegs = 6
#nColors = colors[:len(occList)-nLegs]
#nColors.extend([(255,255,255) for x in xrange(nLegs)])
#
#occPts = [[x] for x in xrange(len(occList))]
#
#for i,loc in enumerate(extrmLocs):
#    for j,pt in enumerate(loc):
#        occPts[LegTipsAssigned[i][j]].append(pt)
#
#
#blIms = np.zeros((len(occPts),2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for idx,pts in enumerate(occPts):
#    for j, pt in enumerate(pts):
#        if j>0:
#            cv2.circle(blIms[idx], tuple(pt), 2, nColors[idx], 2)
#            #cv2.line(blIms[idx], tuple(pt), (widthCrop, heightCrop), nColors[idx], thickness=3)
#        im[heightCrop-h:heightCrop+h, widthCrop-w:widthCrop+w] = 0
#displayImgs(blIms,1)
#
#cv2.imshow('sliceIm', cv2.convertScaleAbs(np.sum(SlImgs[:70], axis=0)))
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#
#
#
#
#
#
#h=heightCrop/10
#w=widthCrop/10
#blIms = np.zeros((len(extrmLocs), 2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for idx,im in enumerate(blIms):
#    for j, pt in enumerate(extrmLocs[idx]):
#        cv2.line(im, tuple(pt), (widthCrop, heightCrop), nColors[LegTipsAssigned[idx][j]], thickness=3)
#        im[heightCrop-h:heightCrop+h, widthCrop-w:widthCrop+w] = 0
#SlImgs = blIms.transpose(1,0,2,3)
#cv2.imshow('sliceIm', cv2.convertScaleAbs(np.sum(SlImgs[:70], axis=0)))
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#
#displayImgs(blIms,100)
#displayImgs(SlImgs,100)
#
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#img = SlImgs[60][:,:,0]+10
#rows, cols = img.shape
#crow,ccol = rows/2 , cols/2
#
#
#mask = np.zeros((rows,cols),np.uint8)
#cutX = 21
#cutY = 21
#mask[crow-cutX:crow+cutX, ccol-cutY:ccol+cutY] = 1
#
#f = np.fft.fftn(img)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))
#
#fshiftCut = fshift.copy()*mask
#magnitude_spectrum1 = 20*np.log(np.abs(fshiftCut))
#
#f_ishift = np.fft.ifftshift(fshiftCut)
#img_back = np.fft.ifft2(f_ishift)
#img_back = np.abs(img_back)
#
#plt.subplot(151),plt.imshow(img, cmap = 'jet')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(152),plt.imshow(magnitude_spectrum, cmap = 'jet')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.subplot(153),plt.imshow(magnitude_spectrum1, cmap = 'jet')
#plt.title('Magnitude Spectrum Cut'), plt.xticks([]), plt.yticks([])
#plt.subplot(154),plt.imshow(img_back, cmap = 'jet')
#plt.title('Out Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(155),plt.imshow(mask, cmap = 'jet')
#plt.title('Mask'), plt.xticks([]), plt.yticks([])
#plt.show()
#
#
#rows, cols = img.shape
#crow,ccol = rows/2 , cols/2
#
## create a mask first, center square is 1, remaining all zeros
#mask = np.zeros((rows,cols,2),np.uint8)
#mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#
#
#
#
#
##legTipLocs = getLegTipLocs(baseDir, trackparams, legCntThresh)
##allLocs, croppedIms = legTipLocs
##fname = baseDir.rstrip(os.sep)+'_legTips-Climbing_'
##LegTipsAssigned, \
##trackedCrpImStack = assignLegTips(tipLocs = allLocs, pxMvmntThresh = pxMvdByLegBwFrm,\
##                                 frmSkipThresh = legTipFrmSkipthresh,saveFileName = fname, \
##                                 crpImStack = croppedIms)
###displayImgs(trackedCrpImStack,50)
##vidObj = cv2.VideoWriter(fname+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2*widthCrop,2*heightCrop))
##
##for _, im in enumerate(trackedCrpImStack):
##    vidObj.write(im)
##vidObj.release()
#
#
#
##t = tp.link_iter(allLocs, search_range = 50, memory=41)    #iterator of locations, distance moved between frames, memory of skipped frame
##ts = []
###tp.logger.propagate = False
##for idx,x in enumerate(t):
##    ts.append(x[1])
##
##legTips = [['frame#','x','y','trackId']]
##for i,loc in enumerate(allLocs):
##    for j,l in enumerate(loc):
##        legTips.append([i, l[0], l[1],ts[i][j]])
##
##fname = baseDir.rstrip(os.sep)+'_legTips'
##csvOutFile = fname+'.csv'
##with open(csvOutFile, "wb") as f:
##    writer = csv.writer(f)
##    writer.writerows(legTips)
##
##legTipsFr = [['frame#',\
##              'x','y','trackId',\
##              'x','y','trackId',\
##              'x','y','trackId',\
##              'x','y','trackId',\
##              'x','y','trackId',\
##              'x','y','trackId']]
##for i,loc in enumerate(allLocs):
##    frLocs = [i]
##    for j,l in enumerate(loc):
##        frLocs.extend((l[0], l[1],ts[i][j]))
##    legTipsFr.append(frLocs)
##csvOutFile = fname+'Fr.csv'
##with open(csvOutFile, "wb") as f:
##    writer = csv.writer(f)
##    writer.writerows(legTipsFr)
##
##dispIms = []
##for i, im in enumerate(croppedIms):
##    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
##    locs = allLocs[i]
##    for j,loc in enumerate(locs):
##        cv2.circle(img, tuple(loc), 2, colors[ts[i][j]], 2)
##    dispIms.append(img)
##
##displayImgs(dispIms,50)
##
##vidObj = cv2.VideoWriter(fname+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2*widthCrop,2*heightCrop))
##
##for _, im in enumerate(dispIms):
##    vidObj.write(im)
##vidObj.release()
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cluster import KMeans
#from sklearn.datasets import make_blobs
#
#plt.rcParams['figure.figsize'] = (16, 9)
#
#
#nClusters = 5
## Creating a sample dataset with 4 clusters
#X, y = make_blobs(n_samples=800, n_features=3, centers=nClusters)
#X.shape
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2])
#
#
#
## Initializing KMeans
#kmeans = KMeans(n_clusters=nClusters)
## Fitting with inputs
#kmeans = kmeans.fit(X)
## Predicting the clusters
#labels = kmeans.predict(X)
## Getting the cluster centers
#C = kmeans.cluster_centers_
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
#
#
#
#
#allVerts = np.vstack((extrmLocs))
#blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for _,v in enumerate(allVerts):
#    blIm[v[1],v[0]] = (255,255,255)
#cv2.imshow("Original", blIm)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
## Initializing KMeans
#kmeans = KMeans(n_clusters=6)
## Fitting with inputs
#kmeans = kmeans.fit(allVerts)
## Predicting the clusters
#labels = kmeans.predict(allVerts)
## Getting the cluster centers
#C = kmeans.cluster_centers_
#
#blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for i,v in enumerate(allVerts):
#    blIm[v[1],v[0]] = colors[labels[i]]
#cv2.imshow("Original", blIm)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#allVertsList = [list(x) for _,x in enumerate(allVerts)]
#frLegTipLabels = []
#for i, tips in enumerate(extrmLocs):
#    ltlabels = []
#    for j, tip in enumerate(tips):
#        ltlabels.append(labels[allVertsList.index(list(tip))])
#    frLegTipLabels.append(ltlabels)
#
#blIms = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in croppedIms.copy()]
#for idx,im in enumerate(blIms):
#    for j, pt in enumerate(extrmLocs[idx]):
#        cv2.circle(im, tuple(pt), 2, colors[frLegTipLabels[idx][j]+10], thickness=3)
#
#displayImgs(blIms,10)
#
#
#
#
#
#from sklearn.cluster import AffinityPropagation
#import matplotlib.pyplot as plt
#from itertools import cycle
#
## Make Dummy Data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
#X = allVerts.copy()
## Setup Affinity Propagation
#af = AffinityPropagation(preference=-1112).fit(X)
#cluster_centers_indices = af.cluster_centers_indices_
#labels = af.labels_
#
#no_clusters = len(cluster_centers_indices)
#
#print('Estimated number of clusters: %d' % no_clusters)
#
## Plot exemplars
#
#plt.close('all')
#plt.figure(1)
#plt.clf()
#
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#for k, col in zip(range(no_clusters), colors):
#    class_members = labels == k
#    cluster_center = X[cluster_centers_indices[k]]
#    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
#    for x in X[class_members]:
#        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
#plt.show()
#           
#
#from sklearn import cluster, datasets, mixture
#allVerts = np.vstack((extrmLocs))
#X = allVerts.copy()
#default_base = {'quantile': .3,
#                'eps': .3,
#                'damping': .9,
#                'preference': -200,
#                'n_neighbors': 10,
#                'n_clusters': 7}
#params = default_base.copy()
#
#spectral = cluster.SpectralClustering(
#    n_clusters=params['n_clusters'], eigen_solver='arpack',
#    affinity="nearest_neighbors")
#spectral.fit(X)
#y_pred = spectral.labels_.astype(np.int)
#
#labels = y_pred
#blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for i,v in enumerate(allVerts):
#    blIm[v[1],v[0]] = colors[labels[i]]
#cv2.imshow("Original", blIm)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#allVertsList = [list(x) for _,x in enumerate(allVerts)]
#frLegTipLabels = []
#for i, tips in enumerate(extrmLocs):
#    ltlabels = []
#    for j, tip in enumerate(tips):
#        ltlabels.append(labels[allVertsList.index(list(tip))])
#    frLegTipLabels.append(ltlabels)
#
#blIms = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in croppedIms.copy()]
#for idx,im in enumerate(blIms):
#    for j, pt in enumerate(extrmLocs[idx]):
#        cv2.circle(im, tuple(pt), 2, colors[frLegTipLabels[idx][j]+10], thickness=3)
#
#displayImgs(blIms,10)
#
#
#
#X = allVerts.copy()
#
#
#from sklearn.cluster import AgglomerativeClustering
#
#dbscan = cluster.DBSCAN(eps=params['eps'])
#dbscan = cluster.DBSCAN(eps=4)
#dbscan.fit(X)
#y_pred = dbscan.labels_.astype(np.int)
#labels = y_pred
#print np.max(labels)
#blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for i,v in enumerate(allVerts):
#    blIm[v[1],v[0]] = colors[labels[i]]
#cv2.imshow("Original", blIm)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#print("Compute unstructured hierarchical clustering...")
#st = time.time()
#ward = AgglomerativeClustering(n_clusters=6, linkage='complete').fit(X)
#elapsed_time = time.time() - st
#labels = ward.labels_
#print("Elapsed time: %.2fs" % elapsed_time)
#print("Number of points: %i" % labels.size)
#
#blIm = np.zeros((2*heightCrop, 2*widthCrop,3), dtype=np.uint8)
#for i,v in enumerate(allVerts):
#    blIm[v[1],v[0]] = colors[labels[i]]
#cv2.imshow("Original", blIm)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#allVertsList = [list(x) for _,x in enumerate(allVerts)]
#frLegTipLabels = []
#for i, tips in enumerate(extrmLocs):
#    ltlabels = []
#    for j, tip in enumerate(tips):
#        ltlabels.append(labels[allVertsList.index(list(tip))])
#    frLegTipLabels.append(ltlabels)
#
#blIms = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in croppedIms.copy()]
#for idx,im in enumerate(blIms):
#    for j, pt in enumerate(extrmLocs[idx]):
#        cv2.circle(im, tuple(pt), 2, colors[frLegTipLabels[idx][j]+10], thickness=3)
#
#displayImgs(blIms,10)
#
#
#
#
#'''
#
#
#
#
##croppedImStack, croppedSubImStack = tracknCrop(baseDir, imExtensions, '.png', heightCrop, widthCrop, \
##                                               cntParams, flyParams, nImThreshold, gauBlurParams, rattailparams, nThreads)
##croppedSubIms = []
##croppedIms = []
##for i in xrange(len(croppedSubImStack)):
##    if 'NoCroppedImage' not in croppedSubImStack[i][1]:
##        croppedSubIms.append(croppedSubImStack[i][1])
##        croppedIms.append(croppedImStack[i][1])
##croppedIms = np.array(croppedIms, dtype=np.uint8)
##croppedSubIms = np.array(croppedSubIms, dtype=np.uint8)
###displayImgs(croppedIms, 100)
###displayImgs(croppedSubIms, 100)
##
##
##
##allLocs = []
##for i, im in enumerate(croppedSubIms):
##    _, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    contours = [x for x in sorted(contours, key = cv2.contourArea)[-6:] if cv2.contourArea(x)>=cntThresh]
##    locs = []
##    for j,cnt in enumerate(contours):
##        locs.append(getFarPoint(cnt)[-1])
##    allLocs.append(np.array(sorted([x for x in locs], key=getEuDisCorner)))
#
#
#
#dispIms2 = []
#for i, im in enumerate(croppedIms):
#    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#    locs = allLocs[i]
#    for j,loc in enumerate(locs):
#        cv2.circle(img, loc, 2, colors[j], 2)
#    dispIms2.append(img)
#
#displayImgs(dispIms2,20)
#
#trackIm = np.zeros((img.shape), dtype=np.uint8)
#for i, im in enumerate(croppedIms):
#    locs = allLocs[i]
#    for j,loc in enumerate(locs):
#        cv2.circle(trackIm, loc, 1, colors[1], 1)
#cv2.imshow('tmp', trackIm)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#trackIms = np.zeros((len(croppedIms),img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
#for i, im in enumerate(croppedIms):
#    locs = allLocs[i]
#    for j,loc in enumerate(locs):
#        cv2.circle(trackIms[i], loc, 1, (255,255,255), 2)
#displayImgs(trackIms,200)
#
#
#for idx,im in enumerate(trackIms):
#    cv2.imwrite(outDir+str(idx)+'.png', im)
#cv2.imshow('tmp', trackIm)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#'''
#
#
#
#
##rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
##
##print "Started processing directories at "+present_time()
##for rawDir in rawdirs:
###    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
##    d = os.path.join(baseDir, rawDir, imgDatafolder)
##    print rawDir
##    if 'Walking' in rawDir:
##        minArea = 1000# 200 for flyClimbing, 1000 for fly walking
##    elif 'Climbing' in rawDir:
##        minArea = 200# 200 for flyClimbing, 1000 for fly walking
##    params.minArea = minArea# 200 for flyClimbing, 1000 for fly walking
##    imdirs = natural_sort([ os.path.join(d, name) for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
##    for imdir in imdirs:
##        dirs = os.path.join(d,imdir)
##        t = tracknCrop(os.path.join(d,imdir), '.png', params, nImThreshold)
#
#
#
#
#
#
#
#
#
#
#
#
#
