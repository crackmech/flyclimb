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
import sys
from datetime import datetime
from thread import start_new_thread as startNT
import Tkinter as tk
import tkFileDialog as tkd
import zipfile
import matplotlib.pyplot as plt
import time

import os
from os.path import basename

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

#from keras.models import load_model
import numpy as np
from datetime import datetime
from functions import your_loss
import glob
from scipy.misc import imread
import re
import time
import cv2
import Tkinter as tk
import tkFileDialog as tkd
import zipfile


#from functions import ysize, overlap, xsize, colors, n_labels, n_channels
#from functions import outBatchSize, modelsDir, srcDir, dstDir





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
params.maxArea = 7000 # 5000 for centroid tracking, 7000 for leg tracking
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
'''
        imData = croppedStack[i]
        if imData[1]=='NoCroppedImage':
            n+=1
            outDir = saveDir+str(n)
            os.mkdir(outDir+'_cropped/')
            os.mkdir(outDir+'_original/')
        elif imData[1]!='NoCroppedImage':
            cv2.imwrite(outDir+'_cropped/'+str(i)+ext, croppedStack[i][1])
            cv2.imwrite(outDir+'_original/'+str(i)+ext, ImStack[i])
'''


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_tiles(img, ysize, overlap):
    img_padded = np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect')
    
    xs = []
    
    for i in xrange(0, img.shape[0], ysize):
        for j in xrange(0, img.shape[1], ysize):
            #print(i-overlap+overlap,i+ysize+overlap+overlap,j-overlap+overlap, j+ysize+overlap+overlap)
            img_overlapped = img_padded[i:i+ysize+overlap+overlap,j:j+ysize+overlap+overlap]
            xs.append(img_overlapped)
            
    return xs

def getImsFromYs(segmentedY, nlabels, outDir, inImgs, fnames, ysize, colors):
    '''
    get output of the model as segmentedY and convert it into individual images and save in outDir
    '''
    for ix,y in enumerate(segmentedY):
            count= 0
            img = inImgs[ix]
            zeros = np.zeros((img.shape[0],img.shape[1],nlabels))
            im = np.zeros((img.shape[0],img.shape[1],3))

            for i in xrange(0, img.shape[0], ysize):
                for j in xrange(0, img.shape[1], ysize):
                    zeros[i:i+ysize,j:j+ysize] = y[count]
                    count += 1
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    color =  np.argmax(zeros[i,j])
                    im[i,j] = colors[color]
            fname = outDir+'/%s_im.png'%(fnames[ix])
            cv2.imwrite(fname, im)

def getOutput(model, inDir, outdir, batchSize):
    '''
    return the output image segmented by based on the input model
    '''
    flist = natural_sort(glob.glob(inDir+'/*'))
    imdims = imread(flist[0]).shape[0]
    if imdims%float(ysize)==0:
        offset = 0
    else:
        offset = (((imdims/ysize + 1)*ysize) - imdims)/2
    print ('Offset: %d'%offset)
    file_chunks = chunks(flist, batchSize)
    for idx, files in enumerate(file_chunks):
        file_names = [basename(path) for path in files]
        imgs = np.array([np.pad(imread(fl, mode='L'), (offset,offset), mode='reflect').astype(float)/255 for fl in files])
        for i in xrange(len(imgs)):
            cv2.imshow('imgs',imgs[i]);cv2.waitKey()
        cv2.destroyAllWindows()
        tiles = np.array([get_tiles(img, ysize, overlap) for img in imgs])
        
        #Create input tensor
        xs = tiles.reshape(imgs.shape[0]*len(tiles[0]),xsize,xsize,n_channels)
        start_time = time.time()
        # Predict output
        print '5', xs.shape
        ys = model.predict(xs)
        print '6'
        print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
        ys = ys.reshape(imgs.shape[0],len(tiles[0]), ysize, ysize, n_labels)
        getImsFromYs(segmentedY=ys , nlabels = n_labels, outDir = outdir, inImgs = imgs, fnames = file_names, ysize = ysize, colors = colors)

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


initialDir = '/media/pointgrey/data/flywalk/'

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













