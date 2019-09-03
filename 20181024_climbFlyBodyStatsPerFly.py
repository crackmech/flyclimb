#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:03:19 2018

@author: aman
"""

import numpy as np
import os
import glob
import re
import random
from datetime import datetime
from datetime import timedelta
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
from scipy import stats
import xlwt
import matplotlib
import csv



imgDatafolder = 'imageData'
trackImExtension = '.jpeg'
csvExt = 'trackData*.csv'
pixelSize =0.055



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


def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    return rows
    
def calcAngle3Pts(a, b, c):
    '''
    returns angle between a and c with b as the vertex 
    '''
    ba = a.flatten() - b.flatten()
    bc = c.flatten() - b.flatten()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def getConsecData(csvData, consecStep, eudDisMinThresh, eudDisMaxThresh, fps, bodyLen):
    '''
    Input: list of lists of 
        a) csvdata 
        b) CSV file and Image file path details
    
    returns a list containing 
        a) X coordinate
        b) Y coordinate
        c) Angle b/w  i-consecStep, i, i+consecStep 
        d) Eucledian distance between i,i+consecStep
        
    CSV file and Image file path details
    '''
    allangles = []
    consecStep = int(consecStep)
    for _, data in enumerate(csvData):
        csvdata = data[0]
        angles = [] 
        for i in xrange(consecStep, len(csvdata)-consecStep-1, consecStep):
            p0 = csvdata[i-consecStep]
            p1 = csvdata[i]
            p2 = csvdata[i+consecStep]
            euDis = np.linalg.norm(csvdata[i+consecStep]-csvdata[i])
            speed = (euDis*fps)/(consecStep*bodyLen)
            angle = (calcAngle3Pts(p0,p1,p2))
            if eudDisMinThresh<euDis<eudDisMaxThresh:
                angles.append(np.array([csvdata[i][0], csvdata[i][1], angle, euDis, speed]))
        allangles.append((np.array(angles), data[1], data[2])) #data[1] contains csv filename, data[2] contins img filename
    return allangles

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]
      
def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S_%f')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S_%f')
    return (time2-time1).total_seconds()

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


def getTrackDirection(trackData, minDis):
    '''
    returns a +1 or -1 based on direction of fly movement.
    If the fly walks from left to right  it returns -1 (equivalent to bottom to top for climbing)
    if the fly walks from right to left, it returns +1 (equivalent to top to bottom for climbing)
    Value is calculated purely on the basis of a line fit on the track based on change of X-coordinate w.r.t frames
    '''
    dataLen = len(trackData)
    m,c,r,_,_ = stats.linregress(np.arange(dataLen), trackData[:,0])
    delta = (m*(9*(dataLen/10))+c)-(m*(dataLen/10)+c)
    if delta>=minDis:
        return -1, r
    elif delta<=-minDis:
        return 1, r
    else:
        return 0, r

def getTrackData(csvdata, skipFrames, consecStep, eudDisMinThresh, eudDisMaxThresh, bodyLen, fps):
    '''
    Input: list of lists of 
        a) csvdata 
        b) CSV file and Image file path details
    
    returns a list containing 
        a) X coordinate
        b) Y coordinate
        c) Angle b/w  i-consecStep, i, i+consecStep 
        d) Eucledian distance between i,i+consecStep
        
    CSV file and Image file path details
    '''
    consecStep = int(consecStep)
    angles = []
    startFrame = consecStep*skipFrames
    stopFrame = len(csvdata)-(consecStep*skipFrames)-1
    for i in xrange(startFrame, stopFrame, consecStep):
        p0 = csvdata[i-consecStep]
        p1 = csvdata[i]
        p2 = csvdata[i+consecStep]
        euDis = np.linalg.norm(csvdata[i+consecStep]-csvdata[i])
        angle = (calcAngle3Pts(p0,p1,p2))
        if eudDisMinThresh > euDis or euDis > eudDisMaxThresh:
            angle = np.nan
            euDis = np.nan
        speed = (euDis*fps)/(consecStep*bodyLen)
        angles.append(np.array([csvdata[i][0], csvdata[i][1], angle, euDis, speed]))
    angles = np.array(angles)
    speedTrack = angles[:,4]
    trackAvInsSpeed = np.nanmean(speedTrack)
    trackDis = np.nansum(speedTrack)
    trackAvSpeed = trackDis*fps/(consecStep*bodyLen*len(angles))
    trackDirection = getTrackDirection(angles, bodyLen)
    trackDetails = [trackAvInsSpeed, trackAvSpeed, trackDis, trackDirection]
    trackDetailsHeader = ['Average InsSpeed for the track', 'AverageSpeed','Total distance of the track', 'Track Direction']
    return angles, trackDetails, trackDetailsHeader , len(angles)#data[1] contains csv filename, data[2] contins img filename

def getFlyDetails(allStats, selParamIndex):
    '''
    returns average FPS, body length for a fly by getting details from its folder
    '''
    fps = allStats[1][-1][0,-1]
    pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])
    param = allStats[1][-1][0,selParamIndex]# get selected parameter size in mm
    blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)
    return blu, fps


#--- avAvspeed for each fly-----
def getFlySpeedDisData(flyTrackData, timeThresh, trackLenThresh, unitTime, imFolder):
    '''
    returns the 
        average speed
        STDEV of average speed
        distanceTravelled in timeThresh
        number of tracks in timeThresh
        distanceTravelled in unitTime
        nTracks in unitTime
    '''
    print flyTrackData[0][-1][0].split('/')[-3]
    flyAllData = []
    flyAllInsSpeeds = []
    flyGeoIndex = 0
    for _,tr in enumerate(flyTrackData):
        if tr[2]<timeThresh:
            if tr[0][1][2]>trackLenThresh:
                avSpeed = tr[0][1][1] # average speed of the fly
                dis = tr[0][1][2] # distance covered by the fly
                insSpeeds = tr[0][0][:,-1] # list of instantaneous speed of the track
                flyGeoIndex+=tr[1][0] # geotactic index of the fly
                pathR = abs(tr[1][1]) # value of 'r' value of the path
                flyAllData.append([avSpeed, dis,flyGeoIndex, tr[0][-1], pathR, tr[2]])
                flyAllInsSpeeds.extend(insSpeeds[~np.isnan(insSpeeds)])
    flyAllData = np.array(flyAllData)
    flyDisPerUnitTime = []
    print flyAllData.shape
    for j in xrange(unitTime, timeThresh+1, unitTime):
        disPerUT = []
        for i in xrange(len(flyAllData[:,-1])):
            if (j-unitTime)<=flyAllData[i,-1]<j:
                disPerUT.append(flyAllData[i,:])
        flyDisPerUnitTime.append(np.array(disPerUT))
        '''
        flyAllData contains: avSpeed per track, distance moved per track, geotactic Index, nFrames per track, time from starting imaging of the fly
        flyAllInsSpeeds contains: a single arrray of all instaneous speeds of the fly
        flyDisPerUnitTime contains: a list of avSpeed,DisMoved,geotactic Index,nFrames,timeFromStarting per unit time, for time segment plots
        '''
    return np.array(flyAllData), np.array(flyAllInsSpeeds), flyTrackData[0][-1][0].split(imFolder)[0], flyDisPerUnitTime


def getLenTrackStats(trackLenArray):
    if trackLenArray.size > 0:
        return np.median(trackLenArray)
    else:
        return 0


def getEuDisCenter(pt1, pt2):
    return np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))

def getTotEuDis(xyArr):
    xyArr = np.array(xyArr)
    n = xyArr.shape[0]
    totDis = np.zeros((n-1))
    for i in xrange(0, n-1):
        totDis[i] = getEuDisCenter(xyArr[i], xyArr[i+1])
    return totDis

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points between p1 and p2
    https://stackoverflow.com/questions/43594646/how-to-calculate-the-coordinates-of-the-line-between-two-points-in-python
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])

def diff(x):
    return x[1]-x[0]

def getTrackBreaksPts(xyArray):
    '''
    returns the breakpoints in the XYarray of the centroids of the fly (determined by the '0' in the array)
    '''
    trackStop = 0
    nTracks = []
    for f,x in enumerate(xyArray):
        if x[0]==0 and x[1]==0:
            if trackStop == 0:
                trackStop = f
                nTracks.append([trackStop])
        else:
            if trackStop != 0:
                nTracks[-1].extend([f])
                trackStop = 0
    if trackStop != 0:
        nTracks[-1].extend([f])
    return nTracks

def extrapolateTrack(xyArray, breakPoints, skippedFrThresh, verbose=True):
    '''
    fills the gaps in the xyArray determined by breakPoints, if the gap is less
        than skippedFrThresh
    
    '''
    splitTracks = []
    trackStart = 0
    arrCopy = xyArray.copy()
    for i,x in enumerate(breakPoints):
        if len(x)>1:
            trackBrkLen =diff(x)
            if verbose:
                print trackBrkLen, x
            if trackBrkLen >= skippedFrThresh:
                splitTracks.append([trackStart, x[0]])
                trackStart = x[1]
            else:
                arrCopy[x[0]:x[1],:] = intermediates(xyArray[x[0]-1,:], xyArray[x[1],:], trackBrkLen)
    if (x[0] - trackStart)>=skippedFrThresh:
        splitTracks.append([trackStart, x[0]])
    return splitTracks, arrCopy

pxSize = 70.00/1280     # in mm/pixel
headerRowId = 0         # index of the header in the CSV file
fpsSep = ' ' 
fpsIndex = -2
#---- declare all the hyperparameters here----#
skpdFrThresh = 0.05       # number of frames that can be skipped in a contigous track
binSize = 7             # number of frames to be binned
threshTrackDur = 50     # threshold of track duration, in number of frames
threshTrackLen = 1      # in BLU
#---- declared all the hyperparameters above ----#

baseDir = '/media/aman/data/flyWalk_data/climbingData/'
baseDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male/imageData/'
csvName = '20171201_200012_contoursStats_Otsu_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male.csv'
csvName = '20171201_200101_contoursStats_Otsu_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male.csv'
csvName = '20171201_200107_contoursStats_Otsu_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male.csv'
fname = baseDir+csvName
baseDir = getFolder(baseDir)
csvDetails = csvName.split('_')
trackStartDate = csvDetails[0]
trackStartTime = csvDetails[1]
fps = float(getFPS(baseDir+csvDetails[0]+'_'+csvDetails[1], 'camloop.txt', fpsSep, fpsIndex))

skpdFramesThresh = skpdFrThresh*fps
csvData = readCsv(fname)
header = csvData[headerRowId]
colIdXCoord = header.index('x-coord')
colIdYCoord = header.index('y-coord')
colIdBodyWidth = header.index('minorAxis (px)')
colIdBodyLen = header.index('majorAxis (px)')
colIdAngle = header.index('angle')
colIdArea = header.index('area (px)')
#---get contig track from one csv file----#
centroids = np.array([x[1:3] for i,x in enumerate(csvData) if i>0], dtype=np.float64)
brkPts = getTrackBreaksPts(centroids)
contigTracks, cnts = extrapolateTrack(centroids, brkPts, skpdFramesThresh, verbose=False)
trackLengths = [diff(x) for _,x in enumerate(contigTracks)]
thresholdedTracksDur = [x for _,x in enumerate(trackLengths) if x>threshTrackDur]
threshTrackFrNum = [contigTracks[trackLengths.index(x)] for _,x in enumerate(thresholdedTracksDur)]

trackDataOutput = []
#---- get data from each contig track from a csv file ----#
trackNumber = 0
csvDetails = csvName.split('_')
trackStartDate = csvDetails[0]
trackStartTime = csvDetails[1]
trackDetails = ('_').join([csvDetails[0],csvDetails[1]])
trackTime = datetime.strptime(trackDetails, '%Y%m%d_%H%M%S')
for i,trk in enumerate(threshTrackFrNum):
    trackData = csvData[headerRowId+1:][trk[0]:trk[1]]
    centroids = np.array([x[1:3] for i_,x in enumerate(trackData)], dtype=np.float64)
    brkPts = getTrackBreaksPts(centroids)
    if brkPts!=[]:
        _, cnts = extrapolateTrack(centroids, brkPts, skpdFramesThresh, verbose=False)
    else:
        cnts = centroids
    bodyLen = np.array([x[colIdBodyLen] for i_,x in enumerate(trackData) if float(x[colIdBodyLen])>0], dtype=np.float64)
    angle = np.array([x[colIdAngle] for i_,x in enumerate(trackData) if float(x[colIdAngle])>0], dtype=np.float64)
    instanDis = getTotEuDis(cnts)
    speedAv = np.mean(instanDis)
    speedMedian = np.median(instanDis)
    speedStd = np.std(instanDis)
    angleAv = np.mean(angle)
    angleMedian = np.median(angle)
    angleStd = np.std(angle)
    gti, rSquared = getTrackDirection(cnts, threshTrackLen)
    # calculate the starting time of the track using track starting frame number from the list of detections
    trackStartT_curr = trackTime+timedelta(seconds=(trk[0]/fps))  
    # calculate the stoping time of the track using track stoping frame number from the list of detections
    trackStopT_curr = trackTime+timedelta(seconds=(trk[1]/fps))
    if trackNumber==0:
        trackStopT_old = trackStartT_curr
    trackStopDelT = (trackStartT_curr-trackStopT_old).total_seconds()
    trackStopT_old = trackStopT_curr
    trackDataOutput.append([csvName.rstrip('.csv')+'_'+str(i),
                            len(cnts),
                            np.sum(instanDis),
                            np.mean(instanDis),
                            np.median(instanDis),
                            np.std(instanDis),
                            np.mean(angle),
                            np.median(angle),
                            np.std(angle),
                            np.mean(bodyLen),
                            np.median(bodyLen),
                            np.std(bodyLen),
                            rSquared,
                            gti,
                            trackStopDelT,
                            binSize,
                            fps,
                            pxSize
                            ])
    print np.sum(instanDis), gti, rSquared, np.mean(bodyLen), np.median(bodyLen), np.std(bodyLen)
    print trackStopDelT
    trackNumber+=1



"""
For each track find:
    1)  track time point                    (separated by _0, _1 for the same time point)
    2)  track duration                      (number of frames)
    3)  track length                        (in BLU)
    4)  Average speed                       (in BLU/s)
    5)  Median speed                        (in BLU/s)
    6)  Std Dev speed                       (in BLU/s)
    7)  Average body angle                  (in degrees)
    8)  Median body angle                   (in degrees)
    9)  Std dev body angle                  (in degrees)
    10) Average body Length                 (in pixels)
    11) Median body Length                  (in pixels)
    12) Std dev body Length                 (in pixels)
    13) Path Straightness                   (r^2 value)
    14) Geotactic Index                     (GTI)
    15) Stopping time difference            (in seconds)
    16) Skipped frame threshold             (in seconds)
    17) Track  stop Time threshold          (in seconds)
    18) Bin size for calculating distance   (integer)
    19) FPS                                 (from camloop.txt)
    20) Pixel size                          (mm/px)

"""




bodyLenHist = plt.hist(bodyLen, alpha=0.5)
plt.show()












