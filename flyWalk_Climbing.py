#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:40:22 2017

@author: flywalk
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:58:22 2016

@author: pointgrey
"""

import flycapture2 as fc2
import numpy as np
import cv2
from datetime import datetime
import sys
from thread import start_new_thread as startNT
import os
#import tkFileDialog as tkd
#import Tkinter as tk
#import matplotlib.pyplot as plt
#from multiprocessing.dummy import Pool as ThreadPool 
#import shutil
from glob import glob




camNumber = 12110917L
saveImDel = True
assay = "-Climbing"

dirname = '/media/pointgrey/data/flywalk/'

try:
    imFly = sys.argv[1]
except:
    imFly = ''
    pass



imData=[]


imDuration = 5      #in minutes
startSegment = 0
totalLoops = 2    #total number of times the imaging loop runs
fps = 250
extension = '.png'

processScaleFactor = 0.15
displayRateFraction = 15
displayScale = 2
framedelThreshMin = 1
framedelThreshMax = 10
avgWeight = 0.01#0.001
avg = None
dirThresh = 250

imShowBorders = ((1,1), (1,1))
borderValue = 128


print 'Enter the fly details : <genotype> etc..'
pupaDetails = raw_input(imFly) or imFly





SaveDirDuration = int(120/imDuration) #variable for creating directory in loop
nFrames = (imDuration*60*fps)+1


def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def logFileWrite(content):
    '''Create log files'''
    try:
        logFile = open(logFileName,'a')# Trying to create a new file or open one
        logFile.write('\n')
        logFile.write(content)
        logFile.close
    except:
        print('Something went wrong! Can\'t create log file')


def createDirs(dirname, dirDetails):
    '''
    creates directories for saving images and csvs
    '''
    #create base directory for all data using current date as the foldername
    try:
        presentDate = present_time()
        os.mkdir(dirname+presentDate+'_'+dirDetails+'/')
        baseDir = dirname+presentDate+'_'+dirDetails+'/'
    except:
        try:
            presentDate = datetime.now().strftime('%Y%m%d')
            os.mkdir(dirname+presentDate+'_1/')
            baseDir = dirname+presentDate+'_1/'
        except:
            try:
                os.mkdir(dirname+'tmp/')
                baseDir = dirname+'tmp/'
            except:
                print "Not able to create directories"
                pass
    try:
        #create directory for saving captured images (saved every 2 hours)  in the base directory
        imDir=baseDir+'imageData/'
        os.mkdir(imDir)
        #create directory for saving ROI images and ROI files in the base directory
        roiDir = baseDir+'roi/'
        os.mkdir(roiDir)
        return baseDir, imDir, roiDir
    except:
        pass

def resizeImage(imData):
    '''
    resizes the image to half of the original dimensions
    '''
    newx,newy = imData.shape[1]/2,imData.shape[0]/2 #new size (w,h)
    resizedImg = cv2.resize(imData,(newx,newy))
    return(resizedImg)

def ShowImage(windowName, imData):
    '''
    Loads and displays nth image from the current directory
    '''
    #cv2.namedWindow(windowName,  cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(windowName, resizeImage(imData))

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
def displayCam(c, im):
    '''
    Starts display from the connected camera. ROIs can be updated by pressing 'u'
    '''
    while(1):
        c.retrieve_buffer(im)
        imData = np.array(im)
        imDataColor=cv2.cvtColor(imData,cv2.COLOR_GRAY2BGR)
        windowName = "Camera Display, Press 'u' to update ROIs or 'Esc' to close"
        ShowImage(windowName, imDataColor)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyWindow(windowName)
    return imData

def getMotion(imData, avg, weight):
    if avg is None:
        print "[INFO] starting background model..."
        avg = imData.copy().astype("float")
    cv2.accumulateWeighted(imData, avg, weight)
    return cv2.absdiff(imData, cv2.convertScaleAbs(avg)), avg

def showIm(frameDelta, displayScale, imShowBorders, borderValue):
    cv2.imshow('img', np.flipud(np.transpose(np.pad(cv2.resize(frameDelta, (0,0), fx = displayScale, fy = displayScale)\
                                , pad_width=imShowBorders, mode='constant', constant_values=borderValue))))
    cv2.waitKey(1)
    

def tracknSave_accWeight(imData, nFrame, weight, avg, imDir, displayRateFraction, flyFirstFrame, extension):
    '''
    tracks object in the given frame on the basis of parameters set in params
    tracking is done by cv2.accumulateWeighted
    Input parameters: 
    imData: image array
    nFrame: frame sequence number
    weight: weight for the acuumulated weight for image averaging
    avg: the averaged image from previous frames
    imDir: directory for saving the image
    flyFirstFrame: boolean value for checking the first frame in the directory
    
    Output parameters:
    avg:  the averaged image produced by cv2.accumulateWeighted using previous avg image
    np.std(frameDelta): standard deviation of the image array, this is used to determine movement in the image
    flyFirstFrame: boolean to determine if the image is first in the directory
    '''
    small = cv2.resize(imData, (0,0), fx = processScaleFactor, fy = processScaleFactor)
    cv2.accumulateWeighted(small, avg, weight)
    frameDelta = cv2.absdiff(small, cv2.convertScaleAbs(avg))
    if nFrame%displayRateFraction==0:
	showIm(frameDelta, displayScale, imShowBorders, borderValue)
    if flyFirstFrame == True:
        startNT(cv2.imwrite,(imDir+str(nFrame)+extension,imData))
    return avg, np.std(frameDelta), flyFirstFrame

def CapNProc(c, im, nFrames, imDir, baseDir, roiDir, saveIm, dirTime, framedelThreshMin, framedelThreshMax, avgWeight, dirThresh, displayRateFraction, extension):
    '''
    Function for capturing images from the camera and then tracking already defined
    templates. The template images are updated (using ROIs defined earlier)
    everytime the function is called. Pressing 'Ctrl+c' pauses the tracking loop
    and starts displaying live images from the camera. This can be used to select
    new templates while the function is running.
    
    Input parameters:
    c:              camera context, used for getting image buffer from the camera
    im:             
    nFrames:        total number of frames to be saved
    imDir:          directory where all imaging folders will be saved
    baseDir:        imaging directory
    roiDir:         directory where all ROI images are saved
    saveIm:         boolean to confirm whether to save images or not
    dirTime:        present_time when this function was called and all images are saved in directory with this as directory name
    framedelThreshMin: threshold to determine whether the standard deviation of the frame is enough to detect movement and hence a walking fly
    avgWeight:      weight for the funciton cv2.accummulatedWeighted to average over previous frames. Higher value means low number of previous frames are averaged
    dirThresh:      Minimum number of files to be present in a folder, if number of files in an imaging folder is less than dirThresh, the folder is deleted
    '''
    print present_time()+" Press 'Ctrl+C' to pause analysis and start live display"
    logFileWrite(present_time())
    tick = datetime.now()
    prevFrame = 0
    flyFirstFrame = False #variable to keep track when the fly entered the imaging arena
    avg = None # variable to hold the averaged image
    currDir = imDir+present_time()+'/'
    for nFrame in range (0,nFrames):
        try:
            if nFrame%250==0:
                tock = datetime.now()
                currFPS = (nFrame-prevFrame)/((tock -tick).total_seconds())
                sys.stdout.write("\rAverage FPS: %0.4f"%(currFPS))
                sys.stdout.flush()
                prevFrame = nFrame
                tick=tock
            c.retrieve_buffer(im)
            imData = np.array(im)
            if saveIm==True:
                if avg is None:
                    print "\n[INFO] starting background model..."
                    avg = cv2.resize(imData, (0,0), fx=processScaleFactor, fy=processScaleFactor).copy().astype("float")
                avg,frDelta, flyFirstFrame = tracknSave_accWeight(imData, nFrame, avgWeight,\
                                            avg, currDir, displayRateFraction, flyFirstFrame, extension)
            elif saveIm==False:
                if nFrame%1000==0:
                    startNT(cv2.imwrite,(currDir+str(nFrame)+'.jpeg',imData,))
            if framedelThreshMin < frDelta < framedelThreshMax and flyFirstFrame == False:
                flyFirstFrame = True
                startFrDelta = str(frDelta)
                saveTick = datetime.now()
                currDir = imDir+present_time()+'/'
#                currDir = imDir+present_time()+'_'+str(frDelta)+'/'
#                print 'creating directory at:',nFrame, frDelta
                try:
                    os.mkdir(currDir)
                    os.chdir(currDir)
                    cv2.imwrite(imDir+ '../roi/' + os.getcwd().split('/')[-1] +\
                                        '_in_'+str(nFrame)+'.jpeg',imData)
#                    print "\rCreated new directory: "+ currDir
                except:
                    pass
            if (frDelta < framedelThreshMin or frDelta > framedelThreshMax) and flyFirstFrame == True:
                flyFirstFrame = False
                stopFrDelta = str(frDelta)
                saveTock = datetime.now()
                fList = os.listdir(currDir)
                try:
                    cv2.imwrite(imDir+ '../roi/' + os.getcwd().split('/')[-1] +\
                                    '_out_'+str(nFrame)+'_'+str(len(fList))+'.jpeg',imData)
                except:
                    pass
                saveFPS = (float(len(fList))/((saveTock -saveTick).total_seconds()))
                if len(fList) >= dirThresh:
                    print ("\nNumber of frames saved in last dir (%s): %d at %0.2f FPS"\
                            %(os.getcwd().split('/')[-1], len(fList), saveFPS))
                    print startFrDelta, stopFrDelta
#                    print str(len(fList)) + ' less frames'
#                    try:
#                        [os.remove(currDir+x) for x in fList]
#                        rmDir = os.getcwd().split('/')[-1]
#                        os.chdir(imDir)
##                        print imDir+rmDir
#                        try:
#                            roiList = glob(imDir+ '../roi/'+rmDir+'*.jpeg')
#                            [os.remove(x) for x in roiList]
#                        except:
#                            print '\ncould not remove rois'
#                            pass
#                        #os.removedirs(imDir+rmDir)
#                        sys.stdout.write('\rRemoved %s with %d frames\n'%(rmDir, len(fList)))
#                        sys.stdout.flush()
#                    except:
#                        print "\rNot able to remove the dir with less images"
#                        pass
#                else:
#                    saveFPS = (float(len(fList))/((saveTock -saveTick).total_seconds()))
#                    print ("\nNumber of frames saved in last dir (%s): %d at %0.2f FPS"\
#                                %(os.getcwd().split('/')[-1], len(fList), saveFPS))
#                    logFileWrite("\nNumber of frames saved in last dir (%s): %d at %0.2f FPS"\
#                                %(os.getcwd().split('/')[-1], len(fList), saveFPS))            
                logFileWrite("\nNumber of frames saved in last dir (%s): %d at %0.2f FPS"\
                            %(os.getcwd().split('/')[-1], len(fList), saveFPS))            
#            elif (frDelta < framedelThreshMin or frDelta > framedelThreshMax):
#            elif ( frDelta > framedelThreshMax):
#                print nFrame, frDelta

        except KeyboardInterrupt:
            print "\nCamera display started on "+present_time()
            logFileWrite("Camera display started on "+present_time())
            avgDisplay = displayCam(c, im)
            avg = cv2.resize(avgDisplay, (0,0), fx=processScaleFactor, fy=processScaleFactor).copy().astype("float")
            print "Camera display exited on  "+present_time()
            logFileWrite("Camera display exited on  "+present_time())
            logFileWrite(present_time())
            tick = datetime.now()
    logFileWrite('----------------------')


#os.chdir(dirname)
date = present_time().split('_')[0]
dirname+=date+'/'

try:
    os.mkdir(dirname)
except:
    print('Date directory (%s) already present'%date)

dirDetails = pupaDetails+assay
try:
    baseDir, imDir, roiDir = createDirs(dirname, dirDetails)
except:
    print "No directories available, please check!!!"
    sys.exit()

logFileName = baseDir+"camloop.txt"
logFileWrite(pupaDetails+assay)
logFileWrite('----------------------')

os.chdir(imDir)

currDir = None
c = fc2.Context()
nCam = c.get_num_of_cameras()

for i in range(0, nCam):
    c.connect(*c.get_camera_from_index(i))
    if c.get_camera_info()['serial_number'] == camNumber:
        camSerial = i

c.connect(*c.get_camera_from_index(camSerial))
p = c.get_property(fc2.FRAME_RATE)
print "Frame Rate: "+str(p['abs_value'])
logFileWrite("Camera Frame Rate: "+str(p['abs_value']))

im = fc2.Image()
c.start_capture()
imData = displayCam(c, im)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imData, 'Starting Image', (400, 300), font, 2, (0,255,100),5)
startNT(cv2.imwrite,(roiDir+'start.jpeg',imData))

for nLoop in range (startSegment,totalLoops):
    dirTime = present_time()
    print('==========  Starting loop #%i  =========='%(nLoop+1))
#    saveDir = imDir+dirTime+'/'
#    os.mkdir(saveDir)
    CapNProc(c, im, nFrames, imDir, baseDir, roiDir, saveImDel, dirTime,\
                    framedelThreshMin, framedelThreshMax, avgWeight, dirThresh, displayRateFraction, extension)
    print "\r\nWaiting for loop number: "+str(nLoop+1)
c.stop_capture()
c.disconnect()

logFileWrite('----------------------')
#
#


