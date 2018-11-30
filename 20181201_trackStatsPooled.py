#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:03:19 2018

@author: aman

For each track find:
    1)  track time point                    (separated by _0, _1 for the same time point)
    2)  track duration                      (number of frames)
    3)  track length                        (in BLU)
    4)  Average speed                       (in BLU/s)
    5)  Median speed                        (in BLU/s)
    6)  StdDev speed                        (in BLU/s)
    7)  Average body angle                  (in degrees)
    8)  Median body angle                   (in degrees)
    9)  StdDev body angle                   (in degrees)
    10) Average body Length                 (in pixels)
    11) Median body Length                  (in pixels)
    12) StdDev body Length                  (in pixels)
    13) Path Straightness                   (r^2 value)
    14) Geotactic Index                     (GTI)
    15) Stopping time difference            (in seconds)
    16) Time difference w.r.t first track   (in seconds)
    17) FPS                                 (from camloop.txt)
    18) Bin size for calculating distance   (integer)
    19) Pixel size                          (mm/px)
    20) Skipped frame threshold             (in frames)
    21) Track duration threshold            (in frames)


"""
import baseFunctions as bf
import baseFunctions_trackStats as bfTrkStats
import numpy as np
import os
from datetime import datetime
from datetime import timedelta
import csv


headerRowId = 0         # index of the header in the CSV file
outCsvHeader = bfTrkStats.csvHeader

baseDir = '/media/aman/data/flyWalk_data/climbingData/'
csvExt = ['*trackStats*.csv']
#baseDir = '/media/pointgrey/data/flywalk/'
#csvExt = ['*trackStats*.csv']
baseDir = bf.getFolder(baseDir)

dirs = bf.getDirList(baseDir)


'''
for each fly, if trackLen>threshTrackLen and tracktimePoint<maxTimeDuration, then,
    for that track, get everything into one list
from this list get a list of timeSeries Data
use the timeseries list to plot total Data
'''
threshTrackTime         = 300   # in seconds, maximum duration of behaviour to be analysed
threshTrackLenMulti     = 3     # multipler of BLU for minimum trackLength w.r.t BLU
unitTimeDur             = 60    # unit time in seconds for pooling data for timeSeries analysis
unitTimeN = threshTrackTime/unitTimeDur
#minTrackLen = blu*3
#disMinThres = blu/20
#disMaxThres = blu
#consecWin = 7
#trackLenThresh = 10*blu


colIdTrackLen   = 2
colIdtrackTmPt  = 15
colIdBodyLen    = 10

genotypeData = []
for i,d in enumerate(dirs):
    fList = bf.getFiles(d, csvExt)
    for _, f in enumerate(fList):
        trackDataOutput = bf.readCsv(f)[1:]
        flyStats = []
        blu = np.nanmean(np.array([x[colIdBodyLen] for _,x in enumerate(trackDataOutput)], dtype=np.float64))
        threshTrackLen = blu*threshTrackLenMulti
        #print d,i, blu, threshTrackLen
        for ix,x in enumerate(trackDataOutput):
            if np.float(x[colIdTrackLen])>=threshTrackLen and np.float(x[colIdtrackTmPt])<threshTrackTime:
                flyStats.append(x)
    genotypeData.append(flyStats)

genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]

for _,d in enumerate(genotypeData):
    for i in xrange(unitTimeN):
        unitTimeData = []
        for _,f in enumerate(d):
            if i*unitTimeDur<=np.float(f[colIdtrackTmPt])<(i+1)*unitTimeDur:
                unitTimeData.append(f)
        genotypeUnitTimeData[i].append(unitTimeData)


def getMedianValue(array):
    '''
    return the "median" value of the array to be used as median value of array
    '''

def getData(trackStatsData):
    '''
    converts the raw trackStatsData to data for final plotting
        1)  Total number of tracks
        2)  Median duration of tracks
        3)  Total distance travelled
        4)  Average Speed
        5)  Average of StdDev bodyAngle
        6)  Average body Length
        7)  Average Path Straightness
        8)  Average Geotactic Index
        9)  Median latency
    '''

    












