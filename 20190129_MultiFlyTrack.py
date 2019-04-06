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
import random
import itertools
import imageio
from thread import start_new_thread as startNT


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

colors = [random_color() for x in xrange(1000)]

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

def getBgSubIm((inImg, bgIm)):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    return cv2.bitwise_not(cv2.absdiff(inImg, bgIm))    
    

def getImstkBgIm(dirname, imExts, imNThresh, pool, workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = getFiles(dirname, imExts)
    nImsToProcess = len(flist)
    startTime = time.time()
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    imWidth, imHeight = imRead(flist[0]).shape
    imgStack = np.zeros((len(flist), imWidth, imHeight), dtype=np.uint8)
    for i in xrange(0, len(flist), imNThresh):
        if i+imNThresh>len(flist):
            j = len(flist)
        else:
            j = i+imNThresh
        imgStack[i:j] = np.array(pool.map(imRead, flist[i:j]), dtype=np.uint8)
    t1 = time.time()-startTime
    print("imRead time for %d frames: %0.2f Seconds at %0.2f FPS"%(len(flist),t1 ,len(flist)/float(t1)))
    t1 = time.time()
    imStackChunks = np.array_split(imgStack[int(len(imgStack)*0.05):int(len(imgStack)*0.5)], 4*workers, axis=1)
    bgImChunks = pool.map(getBgIm, imStackChunks)
    bgIm = np.array(np.vstack((bgImChunks)), dtype=np.uint8)
    t2 = time.time()-t1
    print("parallel bg calculation time for %d frames: %0.2f Seconds at %0.2f FPS\n"%(len(flist),t2 ,len(flist)/float(t2)))
    t2 = time.time()
    pool.close()
    return imgStack, bgIm


from detectors import Detectors
from tracker import Tracker
"""Initialize variable used by Tracker class
Args:
    dist_thresh: distance threshold. When exceeds the threshold,
                 track will be deleted and new track is created
    max_frames_to_skip: maximum allowed frames to be skipped for
                        the track object undetected
    max_trace_length: trace path history length
    trackIdCount: identification of each track object
Return:
    None
"""
import tifffile
outFName = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/20180927_213843_20180924_0130_W1118-flyBowl/imageData/temp.tif'
memmap_image = tifffile.memmap(outFName, shape=(256, 256), dtype='float32')
memmap_image[255, 255] = 1.0
memmap_image.flush()

def main(tiffStep):
    """Main function for multi object tracking
    """
    #tiffStep = 512

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(200, 50, 25, 100)
    
    # Variables initialization
    pause = False
    track_colors = [random_color() for x in xrange(256)]
    # Infinite loop to process video frames
    stTmAv = time.time()
    outFName = imFolder+'_traced_0-'+str(tiffStep) +'.tiff'
    #memmap_image = tifffile.memmap(outFName, shape=(tiffStep, newx, newy, 3), dtype='uint8')
    imgs = np.zeros((tiffStep, newy, newx, 3), dtype = np.uint8)
    tTm = 0
    stTm = time.time()
    for fr in xrange(len(flyContours[0])):
        # Capture frame-by-frame
        frame = getBgSubIm((flyContours[0][fr], flyContours[1]))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        outFrame = cv2.cvtColor(flyContours[0][fr], cv2.COLOR_GRAY2BGR)

        # Detect and return centeroids of the objects in the frame
        centers = detector.DetectCM(frame)
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id
                        
                        cv2.line(outFrame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)
                        cv2.circle(outFrame, (int(x2), int(y2)), 2, track_colors[clr], 2)
                        #cv2.circle(outFrame, (int(x1), int(y1)), 2, (255,25,255), 2)
                    cv2.circle(outFrame, (int(x2), int(y2)), 2, track_colors[clr], 1)

            ## Display the resulting tracking frame
            #cv2.imshow('Tracking', outFrame)
            #cv2.waitKey(1)
            # outFName = imFolder+'_traced/'+flist[fr].split('/')[-1]
            # cv2.imwrite(outFName, outFrame)
            img = cv2.resize(outFrame,(newx,newy))
            imN = (fr%tiffStep)
            if (imN==0 and fr>0):
                outFName = imFolder+'_traced_'+str(fr-tiffStep) + '-'+str(fr)  +'.tiff'
                startNT(imageio.mimwrite, (outFName,imgs))
                imgs = np.zeros((tiffStep, newy, newx, 3), dtype = np.uint8)
                #memmap_image = tifffile.memmap(outFName, shape=(tiffStep, newx, newy, 3), dtype='uint8')
                #memmap_image[imN] = img
                tm = time.time()
                fps= (tiffStep/(tm-stTm))
                tTm += tm-stTm
                print('FPS: %0.3f (frame# %d)'%(fps, fr))
                stTm = tm
            #else:
            #    #print fr, imN
            imgs[imN] = img
    imageio.mimwrite(imFolder+'_traced_'+str((fr/tiffStep)*tiffStep) + '-'+str(fr)  +'.tiff',imgs[:imN])
    print('Tracking average FPS: %0.3f'%(float(fr)/(time.time()-stTmAv)))#(1.0/(tm-stTm)))
    cv2.destroyAllWindows()

#main(1024)
#128    :  Tracking average FPS: 122.351
#256    :  Tracking average FPS: 125.891
#512    :  Tracking average FPS: 126.113
#1024   :  Tracking average FPS: 125.399


#            #cv2.imshow('BgSub', frame)
#
#        # Check for key strokes
#        k = cv2.waitKey(1) & 0xff
#        if k == 27:  # 'esc' key has been pressed, exit program.
#            break
#        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
#            pause = not pause
#            if (pause is True):
#                print("Code is paused. Press 'p' to resume..")
#                while (pause is True):
#                    # stay in this loop until
#                    key = cv2.waitKey(1) & 0xff
#                    if key == 112:
#                        pause = False
#                        print("Resume code..!!")
#                        break
#        if (imN==0 and fr>0):
#            imageio.mimwrite(imFolder+'_traced_'+str(fr-tiffStep) + '-'+str(fr)  +'.tiff',imgs)
#            imgs = np.zeros((tiffStep, newx, newy, 3), dtype = np.uint8)
#            tm = time.time()
#            fps= (fps+(tiffStep/(tm-stTm)))/(2)
#            print('FPS: %0.3f (frame# %d)'%(fps, fr))#(1.0/(tm-stTm)))
#            stTm = tm
#    cv2.destroyAllWindows()
#    print('Tracking average FPS: %0.3f'%(float(fr)/(time.time()-stTmAv)))#(1.0/(tm-stTm)))

#main()



imExtensions = ['*.png', '*.jpeg']
imDataFolder = 'imageData'
statsfName = 'contoursStats_threshBinary'
statsFileHeader = ['frameDetails','x-coord','y-coord','minorAxis (px)','majorAxis (px)','angle','area (px)']

imNThresh = 4096
nThreads = 8
pool = mp.Pool(processes=nThreads)


baseDir = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/'
imFolder = baseDir + '20180927_000154_20180922_2230_KCNJ10SS--flyBowl/imageData/20180927_000433'
#imFolder = baseDir + '20180927_213843_20180924_0130_W1118-flyBowl/imageData/20180927_213857'
imFolder = '/media/aman/data/work/powerpoints/flyMovies/flyVRL/multiFlyFrames_1'
flist = getFiles(imFolder, imExtensions)
imData = imRead(flist[0])
scaleFactor = 0.5
newx,newy = int(imData.shape[1]*scaleFactor), int(imData.shape[0]*scaleFactor) #new size (w,h)

flyContours = getImstkBgIm(imFolder, imExtensions, imNThresh, pool, nThreads)
pool.close()
cv2.imshow('123', flyContours[1])
cv2.waitKey()
cv2.destroyAllWindows()


main(512)

"""FIX TRACKS WITH NO FLIES BY CROSS CORRELATING FLY CENTROIND DETECTION WITH TRACKING DATA"""
#128 Tracking average FPS: 136.503
#127 Tracking average FPS: 137.085
#139 Tracking average FPS: 133.637
#149 Tracking average FPS: 134.051
#109 Tracking average FPS: 127.671
#128 Tracking average FPS: 126.158







##https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html#
#fName = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/20180927_213843_20180924_0130_W1118-flyBowl/imageData/20180927_213857_1000-1099.tif'
#tifName = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/20180927_213843_20180924_0130_W1118-flyBowl/imageData/20180927_213857_1000-1099.tif_123.tiff'
#import tifffile
#
#with tifffile.TiffFile(fName) as tif:
#    images = tif.asarray()
#    for page in tif.pages:
#        for tag in page.tags.values():
#            print tag.name, tag.value
#        print "====="
#        image = page.asarray()
#
#tifs = tifffile.TiffFile(fName)
#
#
#with tifffile.TiffFile(tifName) as tif:
#    images = tif.asarray()
#    for page in tif.pages:
#        for tag in page.tags.values():
#            print tag.name, tag.value
#        print "====="
#        image = page.asarray()
#
##https://stackoverflow.com/questions/20529187/what-is-the-best-way-to-save-image-metadata-alongside-a-tif-with-python
#frames = tifffile.TiffFile(tifName)
#frames.imagej_metadata
#

#Create an empty TIFF file and write to the memory-mapped numpy array:
#
#import tifffile
#outFName = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/20180927_213843_20180924_0130_W1118-flyBowl/imageData/temp.tif'
#memmap_image = tifffile.memmap(outFName, shape=(256, 256), dtype='float32')
#memmap_image[255, 255] = 1.0
#memmap_image.flush()
#memmap_image.shape, memmap_image.dtype



