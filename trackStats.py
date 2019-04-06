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
    6)  StdDev speed                       (in BLU/s)
    7)  Average body angle                  (in degrees)
    8)  Median body angle                   (in degrees)
    9)  StdDev body angle                  (in degrees)
    10) Average body Length                 (in pixels)
    11) Median body Length                  (in pixels)
    12) StdDev body Length                 (in pixels)
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


pxSize = 70.00/1280     # in mm/pixel
imDataFolder = 'imageData'
headerRowId = 0         # index of the header in the CSV file
fpsSep = ' ' 
fpsIndex = -2
outCsvHeader = bfTrkStats.csvHeader

#---- declare all the hyperparameters here----#
mvmntStopThresh = 0.05  # number of seconds that can be skipped fo fly detection to be counted as a contigous track
binSize = 7             # number of frames to be binned
threshTrackDur = 50     # threshold of track duration, in number of frames
threshTrackLen = 1      # in BLU
#---- declared all the hyperparameters above ----#

#baseDir = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/'
#csvExt = ['*contoursStats_tmp*.csv']
baseDir = '/media/pointgrey/data/flywalk/'
csvExt = ['*contoursStats_threshBinary_*.csv']
baseDir = bf.getFolder(baseDir)

dirs = bf.getDirList(baseDir)
for _,rawDir in enumerate(dirs):
    print rawDir
    csvDir = os.path.join(rawDir, imDataFolder)
    fList = bf.getFiles(csvDir, csvExt)
    
    trackNumber = 0
    trackDataOutput = []
    for iF,f in enumerate(fList):
        csvName = f.split(os.sep)[-1]
        #print csvName
        csvDetails = csvName.split('_')
        trackStartDate = csvDetails[0]
        trackStartTime = csvDetails[1]
        try:
            fps = float(bfTrkStats.getFPS(csvName, os.path.join(rawDir, 'camloop.txt'), fpsSep, fpsIndex))
        except:
            fps = 250.0
        skpdFramesThresh = mvmntStopThresh*fps
        trackDetails = ('_').join([csvDetails[0],csvDetails[1]])
        trackTime = datetime.strptime(trackDetails, '%Y%m%d_%H%M%S')
        
        csvData = bf.readCsv(f)
        header = csvData[headerRowId]
        colIdXCoord = header.index('x-coord')
        colIdYCoord = header.index('y-coord')
        colIdBodyWidth = header.index('minorAxis (px)')
        colIdBodyLen = header.index('majorAxis (px)')
        colIdAngle = header.index('angle')
        colIdArea = header.index('area (px)')
        #---get contig track from one csv file----#
        centroids = np.array([x[1:3] for i,x in enumerate(csvData) if i>0], dtype=np.float64)
        if len(centroids)>skpdFramesThresh:
            brkPts = bfTrkStats.getTrackBreaksPts(centroids)
            if brkPts!=[]:
                contigTracks, cnts_ = bfTrkStats.extrapolateTrack(centroids, brkPts, skpdFramesThresh, verbose=False)
            else:
                print 'no track to split, full length track present in ==>\n', csvName
                contigTracks = [[0, len(centroids)]]
            trackLengths = [bfTrkStats.diff(x) for _,x in enumerate(contigTracks)]
            thresholdedTracksDur = [x for _,x in enumerate(trackLengths) if x>threshTrackDur]
            threshTrackFrNum = [contigTracks[trackLengths.index(x)] for _,x in enumerate(thresholdedTracksDur)]
            
            #---- get data from each contig track from a csv file ----#
            for i,trk in enumerate(threshTrackFrNum):
                trackData = csvData[headerRowId+1:][trk[0]:trk[1]]
                centroids = np.array([x[1:3] for i_,x in enumerate(trackData)], dtype=np.float64)
                brkPts = bfTrkStats.getTrackBreaksPts(centroids)
                if brkPts!=[]:
                    _, cnts = bfTrkStats.extrapolateTrack(centroids, brkPts, skpdFramesThresh, verbose=False)
                else:
                    cnts = centroids
                bodyLen = np.array([x[colIdBodyLen] for i_,x in enumerate(trackData) if float(x[colIdBodyLen])>0], dtype=np.float64)
                angle = np.array([x[colIdAngle] for i_,x in enumerate(trackData) if float(x[colIdAngle])>0], dtype=np.float64)
                instanDis = bfTrkStats.getTotEuDis(cnts)
                gti, rSquared = bfTrkStats.getTrackDirection(cnts, threshTrackLen)
                # calculate the starting time of the track using track starting frame number from the list of detections
                trackStartT_curr = trackTime+timedelta(seconds=(trk[0]/fps))  
                # calculate the stoping time of the track using track stoping frame number from the list of detections
                trackStopT_curr = trackTime+timedelta(seconds=(trk[1]/fps))
                if trackNumber==0:
                    trackStopT_old = trackStartT_curr
                    trackDeltaT = 0
                trackStopDeltaT = (trackStartT_curr-trackStopT_old).total_seconds()
                trackDeltaT+= trackStopDeltaT+(trackStopT_curr-trackStartT_curr).total_seconds()
                trackStopT_old = trackStopT_curr
                trackDataOutput.append([csvName.rstrip('.csv')+'_'+str(i), len(cnts), np.sum(instanDis),
                                        np.mean(instanDis), np.median(instanDis), np.std(instanDis),
                                        np.mean(angle), np.median(angle), np.std(angle),
                                        np.mean(bodyLen), np.median(bodyLen), np.std(bodyLen),
                                        rSquared, gti, trackStopDeltaT,trackDeltaT,
                                        fps, binSize, pxSize,
                                        skpdFramesThresh, threshTrackDur,
                                        ])
                #print iF,i, trackStopDeltaT, trk
                trackNumber+=1
    outCsvName = os.path.join(rawDir,'trackStats_'+rawDir.split(os.sep)[-1]+'.csv')
    with open(outCsvName, 'wb') as csvfile: 
        csvWriter = csv.writer(csvfile) 
        csvWriter.writerow(outCsvHeader)
        csvWriter.writerows(trackDataOutput)

'''
fix for distance jumps,
'''

threshTrackTime         = 300   # in seconds, maximum duration of behaviour to be analysed
threshTrackLenMulti     = 3     # multipler of BLU for minimum trackLength w.r.t BLU
unitTimeDur             = 60    # unit time in seconds for pooling data for timeSeries analysis
#minTrackLen = blu*3
#disMinThres = blu/20
#disMaxThres = blu
#consecWin = 7
#trackLenThresh = 10*blu


'''
for each fly, if trackLen>threshTrackLen and , then , for that track, calculate:
'''
colIdTrackLen   = 2
colIdtrackTmPt  = 15
colIdBodyLen    = 10
flyStats = []
blu = np.average(np.array([x[colIdBodyLen] for i_,x in enumerate(trackDataOutput)]))
threshTrackLen = blu*threshTrackLenMulti
for i,x in enumerate(trackDataOutput):
    if x[colIdTrackLen]>=threshTrackLen and x[colIdtrackTmPt]<threshTrackTime:
        flyStats.append(x)












