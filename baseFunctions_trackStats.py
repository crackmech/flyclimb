#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:18:03 2018

@author: aman

functions file for generating a single track statistics file for each fly
"""

import baseFunctions as bf
import numpy as np
from datetime import datetime
from scipy import stats
import os

def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S_%f')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S_%f')
    return (time2-time1).total_seconds()

def getFPS(csvname, logFilePath, FPSsep, FPSIndex):
    '''
    returns the fps of current image folder by looking up the folder details in
    the camera log file in the parent folder
    '''
    csvdetails = csvname.split('_')
    folder = ('_').join([csvdetails[0],csvdetails[1]])
    with open(logFilePath) as f:
        lines = f.readlines()
    fpsValues = []
    for line in lines:
        if all(x in line for x in [folder, 'FPS']):
            fpsValues.append(line.split(FPSsep))
    fpsFrames = [x[FPSIndex-2] for x in fpsValues] # get all lines with required imFolder
    #return the fps from the camloop line with highest number of frames
    return fpsValues[fpsFrames.index(max(fpsFrames))][FPSIndex] 

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
        if len(x)>1 and diff(x)>0:
            trackBrkLen = diff(x)
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
imDataFolder = 'imageData'
headerRowId = 0         # index of the header in the CSV file
fpsSep = ' ' 
fpsIndex = -2
csvHeader = ['trackDetails',
                'track duration (frames)',
                'distance travelled (px)',
                'average instantaneuos speed (px/s)',
                'median instantaneuos speed (px/s)',
                'STD instantaneuos speed (px/s)',
                'average body angle (degrees)',
                'median body angle (degrees)',
                'STD body angle (degrees)',
                'average body length (px)',
                'median body length (px)',
                'STD body length (px)',
                'path straightness (r^2)',
                'geotactic Index',
                'latency (seconds)',
                'timeDelta (seconds)',
                'FPS',
                'Bin Size (frames)',
                'Pixel Size (um/px)',
                'skipped frames threshold (#frames)',
                'track duration threshold (#frames)',
                ]







def getMedVal(array):
    '''
    return the "median" value of the array to be used as median value of array
    '''
    array = np.array(array, dtype=np.float64)
    return np.median(array)

def getTotalNTracks(array, colIdtracknumber):
    '''
    returns the total number of tracks in the given array
    '''
    return len(array)

def getMedTrackDur(array, colIdtrackdur):
    '''
    returns the median track duration from the input array of tracks for a single fly
    '''
    trackDurs = [x[colIdtrackdur] for _,x in enumerate(array)]
    return getMedVal(trackDurs)

def getTotDisTrvld(array, colIdtracklen):
    '''
    returns the total distance travelled by a single fly
    '''
    return np.sum(np.array([x[colIdtracklen] for _,x in enumerate(array)], dtype=np.float64))

def getAvFps(array, colIdfps):
    '''
    returns the average standard deviation of body angle of a single fly 
    '''
    return np.average(np.array([x[colIdfps] for _,x in enumerate(array)], dtype=np.float64))

def getAvSpeed(array, colIdspeedmedian, colIdfps, colIdbodylenmedian):
    '''
    returns the average speed of a single fly by taking the average of the
        median speed of each track
    '''
    return np.average(np.array(
            [(np.float(x[colIdspeedmedian])*np.float(x[colIdfps]))/np.float(x[colIdbodylenmedian])\
            for _,x in enumerate(array)], dtype=np.float64))

def getBodyAngleStd(array, colIdbodyangstd):
    '''
    returns the average standard deviation of body angle of a single fly 
    '''
    return np.average(np.array([x[colIdbodyangstd] for _,x in enumerate(array)], dtype=np.float64))

def getAvBodyLen(array, colIdbodylenmedian):
    '''
    returns the average body length of a single fly by taking the average of the
        median body length of each track
    '''
    return np.average(np.array([x[colIdbodylenmedian] for _,x in enumerate(array)], dtype=np.float64))

def getAvStraightness(array, colIdstraightness):
    '''
    returns the average straightness of a single fly 
    '''
    return np.average(np.array([np.square(np.float(x[colIdstraightness])) for _,x in enumerate(array)], dtype=np.float64))

def getAvGti(array, colIdgti):
    '''
    returns the average geotactic Index of a single fly 
    '''
    return np.average(np.array([x[colIdgti] for _,x in enumerate(array)], dtype=np.float64))

def getMedlatency(array, colIdlatency):
    '''
    returns the median track duration from the input array of tracks for a single fly
    '''
    trackLatencies = [x[colIdlatency] for _,x in enumerate(array)]
    return getMedVal(trackLatencies)

def getTotDur(array, colIdtrackdur):
    '''
    returns the total distance travelled by a single fly
    '''
    return np.sum(np.array([x[colIdtrackdur] for _,x in enumerate(array)], dtype=np.float64))

def getAvDisPerTrk(array, colIdtracklen):
    '''
    returns the average distance covered per track by a single fly
    '''
    return np.average(np.array([x[colIdtracklen] for _,x in enumerate(array)], dtype=np.float64))

def getSex(array, colIdtrackdetails, colors):
    '''
    determine if the fly is male or female, based on the name of the track,
        name of the track contains the sex, either the fly is male or female
    If the sex data is not present, then the fly is determined as unknown
    Returns:
        0 for female
        1 for male
        2 for unknown
    '''
    if len(array)>0:
        trackDetails = array[0][colIdtrackdetails]
        if '_female' in trackDetails:
            return 0, colors['female']
        elif '_male' in trackDetails:
            return 1, colors['male']
        else:
            return 2, colors['unknownSex']
    else:
        return None

def getPooledData(trackStatsData, csvHeader, sexcolors):
    '''
    converts the raw trackStatsData to data for stats and final plotting
        0)  Determine sex, male or female
        1)  Color for the animal (dependent on sex)
        2)  Total number of tracks
        3)  Median duration of tracks
        4)  Total distance travelled
        5)  Average Speed
        6)  Average of StdDev bodyAngle
        7)  Average body Length
        8)  Average Path Straightness
        9)  Average Geotactic Index
        10) Median latency
        11) Total time spent climbing
        12) Average distance per track
        13) Body length Unit size 
        14) fps
    '''
    """FIX THE INPUT colIds and OUTPUT FOR THE DICT OF TITLES"""
    colIdTrackDetails = [csvHeader.index(x) for x in csvHeader if 'trackDetails' in x][0]
    colIdTrackDur = [csvHeader.index(x) for x in csvHeader if 'track duration' in x][0]
    colIdTrackLen = [csvHeader.index(x) for x in csvHeader if 'distance travelled' in x][0]
    colIdTrackSpeed = [csvHeader.index(x) for x in csvHeader if 'median instantaneuos speed ' in x][0]
    colIdBdAngStd = [csvHeader.index(x) for x in csvHeader if 'STD body angle' in x][0]
    colIdTrackBdLen = [csvHeader.index(x) for x in csvHeader if 'median body length' in x][0]
    colIdTrackStrght = [csvHeader.index(x) for x in csvHeader if 'path straightness' in x][0]
    colIdTrackGti = [csvHeader.index(x) for x in csvHeader if 'geotactic Index' in x][0]
    colIdTrackLtncy = [csvHeader.index(x) for x in csvHeader if 'latency' in x][0]
    colIdTrackFps = [csvHeader.index(x) for x in csvHeader if 'FPS' in x][0]
    colIdPxSize = [csvHeader.index(x) for x in csvHeader if 'Pixel Size' in x][0]
    avBdLen = getAvBodyLen(trackStatsData, colIdTrackBdLen)
    fps = getAvFps(trackStatsData, colIdTrackFps)
    pxSize = np.float(trackStatsData[0][colIdPxSize])
    blu = avBdLen*pxSize
    sex, sexColor = getSex(trackStatsData, colIdTrackDetails, sexcolors)
    trackTotNum = getTotalNTracks(trackStatsData, None)
    trackDurMed = getMedTrackDur(trackStatsData, colIdTrackDur)/fps         # in seconds
    totDis = getTotDisTrvld(trackStatsData, colIdTrackLen)/avBdLen          # in BLUs
    avSpeed = getAvSpeed(trackStatsData, colIdTrackSpeed, colIdTrackFps, \
                        colIdTrackBdLen)                                    # in BLU/sec
    avStdBdAng = getBodyAngleStd(trackStatsData, colIdBdAngStd)
    avTrackStrght = getAvStraightness(trackStatsData, colIdTrackStrght)
    avGti = getAvGti(trackStatsData, colIdTrackGti)
    medLatency = getMedlatency(trackStatsData, colIdTrackLtncy)
    totTime = getTotDur(trackStatsData, colIdTrackDur)/fps                  # in seconds
    avDis = getAvDisPerTrk(trackStatsData, colIdTrackLen)/avBdLen           # in BLUs
    return [sex, sexColor, trackTotNum, trackDurMed, totDis, avSpeed, \
            avStdBdAng, avBdLen, avTrackStrght, avGti, medLatency, \
            totTime, avDis, blu, fps]

def getUnitTimePltData(untTmData, colId):
    '''
    get the data for plotting the timeSeries plots
    '''
    outData = [[x[colId] for i_,x in enumerate(d)] for i,d in enumerate(untTmData)]
    dataAv = [np.average(x) for i_,x in enumerate(outData)]
    dataerr = [np.std(x)/np.sqrt(len(x)) for i_,x in enumerate(outData)]
    return dataAv, dataerr


def pooledData(dirName, csvext, unittimedur, threshtotbehavtime, threshtracklenmultiplier,
                  csvheader, csvheaderrow, colIdpooleddict, sexcolors, pltparamlist):
    '''
    returns:
        1)  a list of all data from the input dirName (directory with data from all flies of a single genotype)
        2)  a list of timeSeries data, with time difference of unitTime between two time points, upto threshtotbehavtime
        3)  a list of data for plotting behaviour from total time of behaviour, deteremined by threshtotbehavtime
        4)  a list of data for plotting timeSeries data, with DataAverage and DataError for each parameter for each timepoint
    '''
    #---- read all stats data from each fly folder in a genotype folder ----#
    dirs = bf.getDirList(dirName)
    genotype = dirName.split(os.sep)[-1]
    print('Total fly data present in %s : %d'%(genotype, len(dirs)))
    colIdbodylen = [csvheader.index(x) for x in csvheader if 'median body length' in x][0]
    colIdtracklen = [csvheader.index(x) for x in csvheader if 'distance' in x][0]
    colIdtracktmpt = [csvheader.index(x) for x in csvheader if 'timeDelta' in x][0]
    genotypedata = []
    for i,d in enumerate(dirs):
        fList = bf.getFiles(d, csvext)
        for _, f in enumerate(fList):
            trackDataOutput = bf.readCsv(f)[csvheaderrow+1:]
            if len(trackDataOutput)>0:
                flyStats = []
                blu = np.nanmean(np.array([x[colIdbodylen] for _,x in enumerate(trackDataOutput)], dtype=np.float64))
                threshTrackLen = blu*threshtracklenmultiplier
                #print d,i, blu, threshTrackLen
                for ix,x in enumerate(trackDataOutput):
                    if np.float(x[colIdtracklen])>=threshTrackLen and np.float(x[colIdtracktmpt])<threshtotbehavtime:
                        flyStats.append(x)
                genotypedata.append(flyStats)
            else:
                print('No track data found for :\n%s'%f.split(dirName)[-1])
    #---- read all stats data from each fly folder in a genotype folder ----#
    
    #---- get plot data of the total data from behaviour from total time measured ----#
    pooledtotaldata = [getPooledData(x, csvheader, sexcolors) for i_,x in enumerate(genotypedata) if len(x)>0]
    
    
    unitTimeN = threshtotbehavtime/unittimedur
    #---- convert all data into a list of unitTimePoints with wach element containing raw
    #       data from the stats csv file for each fly in a genotype folder ----#
    genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]
    pooledUnitTimeData = [[] for x in xrange(unitTimeN)]
    for i_,d in enumerate(genotypedata):
        #print d[0][0]
        for i in xrange(unitTimeN):
            unitTimeData = []
            for i_,f in enumerate(d):
                if i*unittimedur<=np.float(f[colIdtracktmpt])<(i+1)*unittimedur:
                    unitTimeData.append(f)
            genotypeUnitTimeData[i].append(unitTimeData)
            if len(unitTimeData)>0:
                pooledUnitTimeData[i].append(getPooledData(unitTimeData, csvheader, sexcolors))
    
    #---- get plot data of the timeSeries data from behaviour from total time measured ----#
    pltDataUnitTime = []
    pltDataTotal = []
    for i in xrange(len(pltparamlist)):
        #print pltParamList[i], colIdPooledDict[pltParamList[i]]
        pltDataUnitTime.append(getUnitTimePltData(pooledUnitTimeData, colIdpooleddict[pltparamlist[i]]))
        pltDataTotal.append([x[colIdpooleddict[pltparamlist[i]]] for i_,x in enumerate(pooledtotaldata)])
    
    return genotypedata, genotypeUnitTimeData, pooledtotaldata, pooledUnitTimeData, pltDataTotal, pltDataUnitTime

















