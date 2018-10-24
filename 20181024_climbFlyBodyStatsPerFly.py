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
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S')
    return (time2-time1).total_seconds()



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



skpdFrThresh = 10   # number of frames that can be skipped in a contgous track

baseDir = '/media/aman/data/flyWalk_data/climbingData/'
baseDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male/imageData/'
fname = baseDir+'20171201_200107_contoursStats_Otsu_tmp_20171201_195931_CS_20171128_0245_11-Climbing_male.csv'
#baseDir = getFolder(baseDir)

csvData = readCsv(fname)

centroids = np.array([x[1:3] for i,x in enumerate(csvData) if i>0], dtype=np.float64)
'''
get contigous data!!!
keep threshold for minimum number of frames, distance
Use max skipped frames
use kalman or intermediate points function to fix data for skipped frames


'''

def getContigTrack(xyArray, skipFrames):
    '''
    returns the x-y coordinates of the longest contigous track from the list of given XY coordinates
    '''
    i=0
    nTracks = []
    trackStart = 0
    for f,x in enumerate(xyArray):
        if f==0:
            trackStart=f
        if x[0]==0 and x[1]==0:
            if i<=skipFrames:
                i+=1
            else:
                nTracks.append([trackStart, f])
                trackStart=f
                i=0
        else:
            pass
    nTracks.append([trackStart, f])
    return nTracks

aa = getContigTrack(centroids, 10)

