#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 00:57:22 2018

@author: aman
"""

imgDatafolder = 'imageData'
trackImExtension = '.jpeg'
csvExt = '.csv'
headers = ['area_average(mm^2)', 'minorAxis_average(mm)', 'majorAxis_average(mm)',\
            'area_median(mm^2)', 'minorAxis_median(mm)', 'majorAxis_median(mm)' ,\
            'nFrames', 'FPS', 'folderName']

selectedParameter = 'minorAxis_median(mm)'
statsfName = 'flyStats_'
blankLinesAfterParam = 2
blankLinesBeforeParam = 1
startString = 'Parameters'
selParamIndex = headers.index(selectedParameter)
pixelSize =0.055
param = 2.5# get selected parameter size in mm
blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)

AngBinMin = 0
AngBinMax = 180
AngBinStep = 1


# import time
# import copy
# import sys
# from math import atan2, degrees
# from thread import start_new_thread as startNT

# import cv2
import os
import glob
import numpy as np
import re
import random
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
from scipy import stats
import xlwt
import dip
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

angleBins = np.arange(AngBinMin, AngBinMax, AngBinStep)


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

def getStringLineIndex(fname, string):
    '''
    returns the list of line numbers containing parameterString (Parameters)
    '''
    stringIndex = []
    with open(fname) as f:
        for num, line in enumerate(f, 1):
            if string in line:
                stringIndex.append(num)
    if stringIndex !=[]:
        print('Found parameters')
        return stringIndex
    else:
        print('No parameters found')

def getAllStats(fname, header, blanksAfterParams, blanksBeforeParams, startString):
    '''
    Get a list of all stats from Stats csv file
    returns:
        1) average stats of a single fly with STD
        2) a list of lists containing all stats read from the Stats csv file
            
    '''
    print fname
    parInd = getStringLineIndex(fname, startString) #get a list of indices with paramters line
    parInd.reverse()
    with open(fname) as f:
        lines = f.readlines()
    nLines = len(lines)
    statslen = len(header[:-1]) #remove the folder name to calculate stats
    allStats = []
    allStats.append(('parameter Details',header))
    for i in xrange(len(parInd)):
        startIndex = parInd[i]+blanksAfterParams
        if parInd[i]!=(nLines-1):
            if i==0:
                stopIndex = nLines
            else:
                stopIndex = parInd[i-1]-blanksBeforeParams-1
            stats = np.zeros((stopIndex-startIndex,statslen))
            for line in xrange(startIndex,stopIndex):
                lineSplit = (lines[line]).split(',')[:-1]
                nan = [True for x in lineSplit if 'nan' in x]
                none = [True for x in lineSplit if 'None' in x]
                if nan==[] and none==[]:
                    stats[line-startIndex,:] = lineSplit
                else:
                    stats[line-startIndex,:] = [np.nan for x in lineSplit]
            avStats = np.zeros((2, statslen))
            avStats[0,:] = np.nanmean(stats, axis=0)
            avStats[1,:] = np.nanstd(stats, axis=0)
            allStats.append((lines[parInd[i]-1], header, stats, avStats))
        else:
            break
    return allStats

def calcAngle3Pts(a, b, c):
    '''
    returns angle between a and c with b as the vertex 
    '''
    ba = a.flatten() - b.flatten()
    bc = c.flatten() - b.flatten()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def getData(basedir, foldername, imDatafolder, trackImExt, csvExt):
    '''
    returns the arrays containing various datasets from a folder containing data of a single fly
    '''
    d = os.path.join(basedir, foldername, imDatafolder)
    csvlist = natural_sort(glob.glob(d+'/*'+csvExt))
    imglist = natural_sort(glob.glob(d+'/*'+trackImExt))
    data = []
    for _, ids in enumerate(zip(csvlist, imglist)):
        csvData = np.genfromtxt(ids[0], dtype='float',delimiter = ',', skip_header=1)
        data.append([csvData, ids[0], ids[1]])
    return data

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

def getHistMode(array, bins):
    '''
    Calculates the mode value based on the histogram of a given array
    returns: the upper and lower limit of the most common value in the array
    '''
    a = array.copy()
    freq, edgs  = np.histogram(a[~np.isnan(a)],bins)
    maxValMin = edgs[np.argmax(freq)]
    maxValMax = maxValMin+np.max(np.diff(bins))
    return maxValMin, maxValMax

def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]
      
def getFlyStats(genotypeData, consecFrameStep, minDisThres, maxDisThres):
    '''Calculates statistics of data from a genotype (folder with 'n' flies)
    
    input: 
        genotypeData:       a list of lists containing trackCSVData and fly TrackStats for each fly in the genotype
        consecFrameStep:    step size of consecutive frames to calculate 
                            a) Angle of fly calculated between (currentFrame-consecFrameStep), currentFrame, (currentFrame+consecFrameStep)
                            b) Distance covered by the fly between (currentFrame) and (currentFrame+consecFrameStep)
                            
        minDisThres:      minimum distance moved by a fly between (currentFrame) and (currentFrame+consecFrameStep)
        maxDisThres:      maximum distance moved by a fly between (currentFrame) and (currentFrame+consecFrameStep)
    
    returns:
        genotypeStats:    a list of all stats of a genotype
                            each list element contains data from all tracks of a fly, with:
                                meanAngle, medianAngle, stdAngle, meanInstantSpeed, medianInstanSpeed, stdInstantSpeed,\
                                meanInstantSpeed per BodyLengthUnit, medianInstanSpeed per BodyLengthUnit, stdInstantSpeed per BodyLengthUnit
        angles:            a list of arrays of all tracks of the each fly completed to the length of the longest track, shorter track data padded by np.nan
        angsData:          a list of lists of all trackData calculated by given parameters
    '''
    genotypeStats = []
    angles = []
    angsData = []
    for i,d in enumerate(genotypeData):
        csvData, allStats = d
        fps = allStats[1][-1][0,-1]
        pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])
        param = allStats[1][-1][0,selParamIndex]# get selected parameter size in mm
        blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)
        print blu, fps, pixelSize
        angData = getConsecData(csvData,consecFrameStep, minDisThres, maxDisThres, fps, blu)
        #print len(angData), len(angData[0])
        maxTrackLen = max([len(x[0]) for x in angData])
        angs = np.zeros((len(angData),maxTrackLen,3))# array of angle, distance and speeds for each track of a fly
        flyStats = np.zeros((maxTrackLen,9))# array of median angle, median distance and median speed and their STDEV for each fly
        angs[:] = np.nan
        for i,ang in enumerate(angData):
            for j in xrange(len(ang[0])):
                angs[i,j,0] = ang[0][j,2]# copy angles
                angs[i,j,1] = ang[0][j,3]# copy distance
                angs[i,j,2] = ang[0][j,4]# copy speeds
        for i in xrange(0, len(flyStats[0]),3):
            data = angs[:,:,i/3]
            flyStats[:,i] = np.nanmean(data, axis=0)
            flyStats[:,i+1] = getMedian(data, i)
            flyStats[:,i+2] = np.nanstd(data, axis=0)
        genotypeStats.append(flyStats)
        angles.append(angs)
        angsData.append(angData)
    return genotypeStats, angles, angsData

def getMedian(dataArray, i):
    '''
    returns the "median" value of dataArray
        median is calculated by the function needed to replace the np.median for the dataArray
    '''
    med = np.zeros((len(dataArray[0])))
    med = np.median(dataArray, axis=0)
    #return med
    if i==0:
        bins = angleBins
    else:
        bins = speedBins
    for j in xrange(len(med)):
        med[j] = getHistMode(dataArray[:,j], bins)[0]
    return med

maxTimeThresh = 300 # time for calculation of data from tracks under this much seconds
chukFrames = 20 # number of frames to be chucked from start and end of the track to initiate data calculation
minTrackLen = blu*10
unitTime = 60


disMinThres = blu/20
disMaxThres = blu
consecWin = 7
trackLenThresh = 10*blu


speedBinMin = disMinThres
speedBinMax = disMaxThres
speedBinStep = 0.1
speedBins = np.arange(speedBinMin, speedBinMax, speedBinStep)

baseDir = '/media/aman/data/flyWalk_data/climbingData/'
#baseDir = '/media/pointgrey/data/flywalk/20180104/'
colors = [random_color() for c in xrange(1000)]

baseDir = getFolder(baseDir)
print "Started processing directories at "+present_time()

def getAllFlyStats(genotypeDir):
    '''
    returns allFlyStats for a all fly folders in a given folder (genotypeDir)
    
    '''
    rawdirs = natural_sort([ name for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    cs = []
    for rawDir in rawdirs:
        print'=====raw'
        csvData = getData(genotypeDir, rawDir, imgDatafolder, trackImExtension, csvExt)
        fname = glob.glob(os.path.join(genotypeDir, rawDir, statsfName+'*'))[0]
        trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
        cs.append([csvData, trackStats])
    
    return getFlyStats(cs, consecWin, disMinThres, disMaxThres)

def getAllFlyCsvData(genotypeDir):
    '''
    returns data of all Csv's of all tracks for all fly folders in a given folder (genotypeDir)
    
    '''
    rawdirs = natural_sort([ name for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    cs = []
    for rawDir in rawdirs:
        print'=====raw'
        csvData = getData(genotypeDir, rawDir, imgDatafolder, trackImExtension, csvExt)
        fname = glob.glob(os.path.join(genotypeDir, rawDir,statsfName+'*'))[0]
        trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
        cs.append([csvData, trackStats])
    return cs

def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S')
    return (time2-time1).total_seconds()

dirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
genotypes = ['CS','Dop2R','Park25','PINK1RV', r'Trp-$\gamma$']

genotypes = []
colors = []
for i, gt in enumerate(dirs):
    if 'W1118' in gt:
        genotypes.append(r'W$^1$$^1$$^1$$^8$')
    elif 'Trp' in gt:
        genotypes.append(r'Trp-$\gamma$')
    elif 'PINK1RV' in gt:
        genotypes.append(r'PINK1$^R$$^V$')
    else:
        genotypes.append(gt)

alfa = 0.71
div = 255.0
colors = [(0/div,0/div,0/div,alfa),
             (200/div,129/div,0/div,alfa),
             (86/div,180/div,233/div,alfa),
             (204/div,121/div,167/div,alfa),
             (0/div,158/div,115/div,alfa),
             (0/div,114/div,178/div,alfa),
             (213/div,94/div,0/div,alfa),
             (240/div,228/div,66/div,alfa),
             (204/div,121/div,167/div,alfa)
             ]


allGenotypesCsvData = []
for _,d in enumerate(dirs):
    path = os.path.join(baseDir, d)
    allGenotypesCsvData.append([path, getAllFlyCsvData(path)])


genoTypeDataProcessed = []
for g, genotype in enumerate(allGenotypesCsvData):
    for f, fly in enumerate(genotype[1]):
        flyAlltracks = []
        flyTimeThTracks = []
        print "------",f,"------"
        for t, tracks in enumerate(fly[0]):
            print t

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

def getFlyDetails(allStats):
    '''
    returns average FPS, body length for a fly by getting details from its folder
    '''
    fps = allStats[1][-1][0,-1]
    pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])
    param = allStats[1][-1][0,selParamIndex]# get selected parameter size in mm
    blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)
    return blu, fps


genoTypeDataProcessed = []
for g, genotype in enumerate(allGenotypesCsvData):
    allFlyAllTracks = []
    for f, fly in enumerate(genotype[1]):
        blu, fps = getFlyDetails(fly[1])
        flyAlltracks = []
        flyTimeThTracks = []
        print "------",f,"------"
        for t, tracks in enumerate(fly[0]):
            if t==0:
                trackTimeDiff = 0
                startTrackCsvName = fly[0][t][1]
                startTrackTime = startTrackCsvName.split('/')[-1].split('_trackData')[0]
            else:
                currTrackCsvName = fly[0][t][1]
                currTrackTime = currTrackCsvName.split('/')[-1].split('_trackData')[0]
                trackTimeDiff  = getTimeDiffFromTimes(currTrackTime, startTrackTime)
            if len(tracks[0])>(2+(2*consecWin*chukFrames)):
                trackData = getTrackData(tracks[0], chukFrames, consecWin, blu/50.0, blu, blu, fps)
                flyAlltracks.append([trackData, trackData[1][-1],  trackTimeDiff, tracks[1:]])
        allFlyAllTracks.append(flyAlltracks)
    genoTypeDataProcessed.append(allFlyAllTracks)



#--- avAvspeed for each fly-----
def getFlySpeedDisData(flyTrackData, timeThresh, trackLenThresh, unitTime):
    '''
    returns the 
        average speed
        STDEV of average speed
        distanceTravelled in timeThresh
        number of tracks in timeThresh
        distanceTravelled in unitTime
        nTracks in unitTime
    '''
    flyAllData = []
    flyAllInsSpeeds = []
    flyGeoIndex = 0
    for _,tr in enumerate(flyTrackData):
        print tr[3][0].rsplit('/')[-1]
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
    for j in xrange(unitTime, timeThresh+1, unitTime):
        disPerUT = []
        for i in xrange(len(flyAllData[:,-1])):
            if (j-unitTime)<=flyAllData[i,-1]<j:
                disPerUT.append(flyAllData[i,:])
        flyDisPerUnitTime.append(np.array(disPerUT))
        print"---"
        '''
        flyAllData contains: avSpeed per track, distance moved per track, geotactic Index, nFrames per track, time from starting imaging of the fly
        flyAllInsSpeeds contains: a single arrray of all instaneous speeds of the fly
        flyDisPerUnitTime contains: a list of avSpeed,DisMoved,geotactic Index,nFrames,timeFromStarting per unit time, for time segment plots
        '''
    return np.array(flyAllData), np.array(flyAllInsSpeeds), flyDisPerUnitTime

maxTimeThresh = 300 # time for calculation of data from tracks under this much seconds
chukFrames = 20 # number of frames to be chucked from start and end of the track to initiate data calculation
minTrackLen = blu*3
unitTime = 60
nUnitTimes = maxTimeThresh/unitTime
nParams = 6
unitDataIndex = -1
xlGapColumns = 2



allGenotypeMovementData = []
for i, genotypeTracks in enumerate(genoTypeDataProcessed):
    genotypeMovementData = []
    for _,flAllTrks in enumerate(genotypeTracks):
        flyMovementData = getFlySpeedDisData(flAllTrks, maxTimeThresh, minTrackLen, unitTime)
        genotypeMovementData.append(flyMovementData)
    allGenotypeMovementData.append(genotypeMovementData)     



genotypeAvSpeed = []
genotypeDis = []
genotypeNTracks = []
genotypeGeoTacInd = []
genotypeLenTrack = []
genotypeStraight =[]
genotypeInsSpeed = []

# genotypeDisPerUT = []
# genotypeNTrackPerUT = []

for _,genotype in enumerate(allGenotypeMovementData):
    flyAvSpeed = []
    flyDis = []
    flyNTracks = []
    flyGeoTacInd = []
    # flyDisPerUT = []
    # flyNTrackPerUT = []
    flyInsSpeed = []
    flylenTrack = []
    flyStraight = []
    for i,fly in enumerate(genotype):
            flyAvSpeed.append(np.mean(fly[0][:,0]))
            flyDis.append(np.nansum(fly[0][:,1]))
            if i==0:
                flyInsSpeed = fly[1]
                #flylenTrack = fly[0][:,3]# for track duration
            else:
                flyInsSpeed = np.hstack((flyInsSpeed, fly[1]))
                #flylenTrack = np.hstack((flylenTrack,fly[0][:,3]))# for track duration
            flylenTrack.append(np.median(fly[0][:,3]))# for track duration
            flyGeoTacInd.append(fly[0][-1,2]/len(fly[0]))
            flyNTracks.append(len(fly[0]))
            flyStraight.append(np.nanmean(fly[0][:,4]))
    genotypeLenTrack.append(np.array(flylenTrack))
    genotypeAvSpeed.append(np.array(flyAvSpeed))
    genotypeDis.append(np.array(flyDis))
    genotypeNTracks.append(np.array(flyNTracks))
    genotypeGeoTacInd.append(np.array(flyGeoTacInd))
    genotypeInsSpeed.append(flyInsSpeed)
    genotypeStraight.append(flyStraight)


markers = ['^','s','v','d','o']

plotTitles = ['Number of Tracks\nin 5 minutes',
              'Duration of Tracks',
              'Total Distance Travelled\nin 5 minutes',
              'Average Speed',
              'Geotactic Index',
              'Path Straightness'
              ]

plotTitlesPerUT = ['Number of Tracks',
              'Duration of Tracks',
              'Total Distance Travelled',
              'Average Speed',
              'Geotactic Index',
              'Path Straightness'
              ]

plotYLabels = ['Number of Tracks',
                'Seconds',
                'Body Lengths',
                'Body Length / S',
                'Geotactic Index',
                r'R$^2$ Value'
                ]

def getLenTrackStats(trackLenArray):
    if trackLenArray.size > 0:
        return np.median(trackLenArray)
    else:
        return 0

sWidth = 0.012
sSize = 20
sMarker = 'o'
sAlpha = 0.99
sLinewidth = 1
sEdgCol = (0,0,0)
sCol = (1,1,1)

scatterDataWidth = 0.012

def plotScatter(axis, data, scatterWidth = sWidth, \
                scatterRadius = sSize , scatterColor = sCol,\
                scatterMarker = sMarker, scatterAlpha = sAlpha, \
                scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot
    '''
    return axis.scatter(np.linspace(scatterWidth+1, -scatterWidth+1,len(data)), data,\
            s=scatterRadius, color = scatterColor, marker=scatterMarker,\
            alpha=scatterAlpha, linewidths=scatterLineWidth, edgecolors=scatterEdgeColor )


#---get the per unit time data ----
allGenotypePerUT_Data = []
for _,genotype in enumerate(allGenotypeMovementData):
    genotypePerUT_Data = []
    for i,fly in enumerate(genotype):
        flyDataPerUT = np.zeros((nUnitTimes, nParams))
        nTracks = 0
        for t in xrange(nUnitTimes):
            if fly[unitDataIndex][t].size > 0:
                nTracks+=len(fly[unitDataIndex][t])
                flyDataPerUT[t,0] = (len(fly[unitDataIndex][t]))# nTracks
                flyDataPerUT[t,1] = (getLenTrackStats(fly[unitDataIndex][t][:,3])) # duration of track
                flyDataPerUT[t,2] = (np.nansum(fly[unitDataIndex][t][:,1])) # Total Distance
                flyDataPerUT[t,3] = (np.nanmean(fly[unitDataIndex][t][:,0])) #Average Speed
                flyDataPerUT[t,4] = (fly[unitDataIndex][t][-1, 2])/nTracks # Geotactic Index
                flyDataPerUT[t,5] = (np.nanmean(fly[unitDataIndex][t][:,4])) # Path Straightness
        genotypePerUT_Data.append(flyDataPerUT)
    allGenotypePerUT_Data.append(genotypePerUT_Data)
    # allGenotypePerUT_Data.append(np.array(genotypePerUT_Data))

#------- CHECK for NORMALITY --------
params = ['nTracks', 'trackDuration', 'Distance',\
          'Speed', 'GeotacticIndex', 'PathStraightness']
genotypeParams = [genotypeNTracks,
                  genotypeLenTrack,
                  genotypeDis,
                  genotypeAvSpeed,
                  genotypeGeoTacInd,
                  genotypeStraight]
f = open("/media/aman/data/thesis/ClimbingPaper/data/climbing5MinutesStats.csv", 'wa')
#--Check normality for 5 minutes data----
print '\n\n\n----Check normality for 5 minutes data----'
for p, par in enumerate(genotypeParams):
    print '------', params[p],'------'
    f.write('\n\n------Normality check for %s------\n'%params[p])
    for g, gt in enumerate(par):
        print stats.normaltest(gt)
        f.write('%s: %s\n'%(genotypes[g],str(stats.normaltest(gt))))
f.close()    
#--Check normality for Per minute data----

f = open("/media/aman/data/thesis/ClimbingPaper/data/climbingPerMinuteStats.csv", 'wa')
fn = open("/media/aman/data/thesis/ClimbingPaper/data/climbingPerMinuteNormalityStats.csv", 'wa')
f.write('\nKruskal-Wallis test for: ')
fn.write('\nD’Agostino-Pearson’s Normality test for: \n')
print '\n\n\n---Checking normality for Per minute data----'
for t in xrange(nUnitTimes):
    for p, par in enumerate(params):
        gtData = []
        for i in xrange(len(genotypes)):
            parData = [allGenotypePerUT_Data[i][x][t,p] for x in xrange(len(allGenotypePerUT_Data[i]))]
            gtData.append(parData)
            print 'Normality value for:%s for %s (%d minute)'%(params[p], genotypes[i], (t+1))
            fn.write('Normality value for:%s for %s (%d minute):%s\n'%(params[p], genotypes[i], (t+1), str(stats.normaltest(parData))))
            print stats.normaltest(parData)
            print stats.shapiro(parData)
        print '\n\n\n---',stats.kruskal(*gtData)
        print '---',stats.f_oneway(*gtData)
        f.write('\n:%s (%d minute): %s'%(params[p], t+1, str(stats.kruskal(*gtData))))
f.close()


trackFPS = 35
ax00 = {'title': plotTitlesPerUT[0],\
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes', \
       'ylabel': plotYLabels[0], 'yticks': [0, 1, 2, 3, 4, 5] }
nSecs = 8
ax01 = {'title': plotTitlesPerUT[1], \
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes', \
       'yticklabels':  np.arange(0,nSecs,2), 'ylabel': plotYLabels[1], 'yticks': np.arange(0, trackFPS*nSecs, 2*trackFPS) }
ax02 = {'title': plotTitlesPerUT[2], \
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes', \
        'ylabel': plotYLabels[2], 'yticks': [0, 1000, 2000, 3000, 4000], 'yticklabels': ['', 1000, '', 3000, ''] }
ax03 = {'title': plotTitlesPerUT[3], 'ylabel': plotYLabels[3], \
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes', \
       'yticks': [0, 1, 2, 3, 4, 5, 6] }
ax04 = {'title': plotTitlesPerUT[4], \
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes',\
       'ylim': (1.2, -1.2), 'yticks': [-1, -0.5, 0, 0.5, 1], 'ylabel': plotYLabels[4] }
ax05 = {'title': plotTitlesPerUT[5], \
       'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes',\
       'ylim': (-0.2, 1.2), 'yticks': [0, 0.5, 1], 'ylabel': plotYLabels[5] }


ax10 = {'title': plotTitles[0], 'ylabel': plotYLabels[0]}
nSecs = 10
ax11 = {'title': plotTitles[1], 'ylabel': plotYLabels[1], 'yticks': np.arange(0, trackFPS*nSecs, 2*trackFPS), \
       'yticklabels':  np.arange(0,15,2), 'ylim':(0,trackFPS*nSecs) }
ax12 = {'title': plotTitles[2], 'yticks': [0, 5000,10000, 15000, 20000], 'ylabel': plotYLabels[2], \
       'yticklabels': [0, '5,000','10,000', '15,000', ''], 'ylim':(0,21000) }
ax13 = {'title': plotTitles[3], 'ylabel': plotYLabels[3], 'xticks': [0,1,2], 'xticklabels':['','CS','']}
ax14 = {'title': plotTitles[4], 'yticks': [-1, 0, 1], 'ylabel': plotYLabels[4], 'ylim': (1.2, -1.2) }
ax15 = {'title': plotTitles[5],\
       'ylim': (-0.2, 1.2), 'yticks': [0, 0.5, 1], 'ylabel': plotYLabels[5] }

axP = [[ax00, ax01, ax02, ax03, ax04, ax05],\
      [ax10, ax11, ax12, ax13, ax14, ax15]]

axLabelRot = [2,4,5]

bAlpha = 0.5
vAlpha = 0.7
    
csGT = allGenotypePerUT_Data[0]
data = np.nanmean(csGT[:], axis=0)
sem = stats.sem(csGT[:], axis=0)

fig, ax = plt.subplots(2,nParams, gridspec_kw = {'height_ratios':[1,2]})
for i in xrange(len(data[0])):
        ax[0, i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], color=colors[0], fmt='-'+markers[0])

plotData = genotypeNTracks[0]
bp0 = ax[1,0].boxplot(plotData, sym='', notch=True, patch_artist=True)
plotScatter(ax[1,0], plotData)

plotData = genotypeLenTrack[0]
bp2 = ax[1,1].boxplot(plotData, sym='', notch=True, patch_artist=True)
plotScatter(ax[1,1], plotData)

plotData = genotypeDis[0]
bp1 = ax[1,2].boxplot(plotData, sym='', notch=True, patch_artist=True)
plotScatter(ax[1,2], plotData)

plotData = genotypeAvSpeed[0]
plotScatter(ax[1,3], plotData)
violPlot = ax[1,3].violinplot(plotData, showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')

plotData = genotypeGeoTacInd[0]
bp4 = ax[1,4].boxplot(genotypeGeoTacInd[0], sym='', notch=True, patch_artist=True)
plotScatter(ax[1,4], plotData)

plotData = genotypeStraight[0]
bp5 = ax[1,5].boxplot(plotData, sym='', notch=True, patch_artist=True)
plotScatter(ax[1,5], plotData)

for bplot in (bp0, bp1, bp2, bp4, bp5):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(bAlpha)
for pc, color in zip(violPlot['bodies'], colors):
    pc.set_color(color)
    pc.set_alpha(vAlpha)

violPlot['cmedians'].set_color('orange')

plt.setp([axs for axs in ax[1,:]], xticklabels = genotypes, xlabel = 'Genotype')
plt.setp(ax[1,3], xticks = [1], xlabel = 'Genotype')
plt.setp([axs.get_xticklabels() for axs in ax[1,:]], rotation=45, horizontalalignment='center')
plt.setp([axs.spines['right'].set_visible(False) for axs in ax[0,:]])
plt.setp([axs.spines['top'].set_visible(False) for axs in ax[0,:]])
plt.setp([axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5) for axs in ax[0,:]])
plt.setp([axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5) for axs in ax[1,:]])

for i in xrange(len(axP)):
    for j in xrange(len(axP[i])):
        plt.setp(ax[i,j], **axP[i][j])
    for j in xrange(len(axLabelRot)):
        plt.setp(ax[i,axLabelRot[j]].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
plt.show()




scatterDataWidth = 0.062

ax12 = {'title': plotTitles[2], 'yticks': [0, 5000,10000, 15000, 20000], 'ylabel': plotYLabels[2], \
       'yticklabels': [0, '5,000','10,000', '15,000', '20,000'] }
ax13 = {'title': plotTitles[3], 'xticks': [1,2,3,4,5], 'ylabel': plotYLabels[3]}

axP = [[ax00, ax01, ax02, ax03, ax04, ax05],\
      [ax10, ax11, ax12, ax13, ax14, ax15]]

bAlpha = 0.6
vAlpha = 0.5

fig, ax = plt.subplots(2,nParams, gridspec_kw = {'height_ratios':[1,2]})
for c,gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    for i in xrange(len(data[0])):
        ax[0,i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], color=colors[c], fmt='-'+markers[c], label=genotypes[c])
        if i==3:
            ax[0,i].legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=2).draggable()

bp0 = ax[1,0].boxplot([d for d in genotypeNTracks], sym='', notch=True, patch_artist=True)
for x, da in enumerate(genotypeNTracks):
    ax[1,0].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

bp2 = ax[1,1].boxplot([d for d in genotypeLenTrack], sym='', notch=True, patch_artist=True)
for x, da in enumerate(genotypeLenTrack):
    ax[1,1].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

bp1 = ax[1,2].boxplot([d for d in genotypeDis], sym='', notch=True, patch_artist=True)
for x, da in enumerate(genotypeDis):
    ax[1,2].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

for x, da in enumerate(genotypeAvSpeed):
    ax[1,3].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )
violPlot = ax[1,3].violinplot([da for da in genotypeAvSpeed], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')

bp4 = ax[1,4].boxplot([d for d in genotypeGeoTacInd], sym='', notch=True, patch_artist=True)
for x, da in enumerate(genotypeGeoTacInd):
    ax[1,4].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

bp5 = ax[1,5].boxplot([d for d in genotypeStraight], sym='', notch=True, patch_artist=True)
for x, da in enumerate(genotypeStraight):
    ax[1,5].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
                s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

for bplot in (bp0, bp1, bp2, bp4, bp5):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(bAlpha)
for pc, color in zip(violPlot['bodies'], colors):
    pc.set_color(color)
    pc.set_alpha(vAlpha)
violPlot['cmedians'].set_color('orange')
plt.setp([axs for axs in ax[1,:]], xticklabels = genotypes, xlabel = 'Genotype')
plt.setp(ax[1,3], xticks = [1], xlabel = 'Genotype')
plt.setp([axs.get_xticklabels() for axs in ax[1,:]], rotation=45, horizontalalignment='center')
plt.setp([axs.spines['right'].set_visible(False) for axs in ax[0,:]])
plt.setp([axs.spines['top'].set_visible(False) for axs in ax[0,:]])
plt.setp([axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5) for axs in ax[0,:]])
plt.setp([axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5) for axs in ax[1,:]])

for i in xrange(len(axP)):
    for j in xrange(len(axP[i])):
        plt.setp(ax[i,j], **axP[i][j])
    for j in xrange(len(axLabelRot)):
        plt.setp(ax[i,axLabelRot[j]].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')



# plt.scatter([np.mean(d) for d in genotypeDis],[np.mean(d) for d in genotypeStraight])
# for x, da in enumerate(genotypeLenTrack):
#     ax[1,1].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )
# bp5 = ax[1,5].boxplot([d for d in genotypeStraight], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeStraight):
#     ax[1,5].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )






ax00 = {'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': plotYLabels[0], \
        'ylabel': plotYLabels[4], 'ylim': [1,-1], 'yticks':[-1, -0.5, 0, 0.5, 1] }
markers = ['^','s','v','d','o']



fig, ax = plt.subplots()
for c,gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    std = np.nanstd(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    ax.scatter(data[:,0], data[:,4], color=colors[c], marker=markers[c], label = genotypes[c])
ax.legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=1).draggable()
plt.setp(ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
plt.setp(ax,**ax00)




fig, ax = plt.subplots()
for c,gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    std = np.nanstd(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    ax.scatter(data[:,2], data[:,5], color=colors[c], marker=markers[c], label = genotypes[c])
ax.legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=1).draggable()
plt.setp(ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
plt.setp(ax,**ax00)


from matplotlib.markers import MarkerStyle

fs = ['none', 'left', 'bottom', 'right', 'top', 'full']

fig, ax = plt.subplots()
for c,gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    std = np.nanstd(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    ax.scatter(data[-1,0], data[-1,4], color=colors[c], marker=markers[c],label = genotypes[c] )
    for i in xrange(len(data[:,0])):
        ax.scatter(data[i,0], data[i,4], color=(1,1,1), marker=markers[c],\
                   linewidths=sLinewidth, edgecolors=sEdgCol )
        ax.scatter(data[i,0], data[i,4], color=colors[c], \
                   marker=MarkerStyle(markers[c], fillstyle=fs[i]))
ax.legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=1).draggable()
plt.setp(ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
plt.setp(ax,**ax00)




fig, ax = plt.subplots()
for c,gt in enumerate(allGenotypePerUT_Data):               
    data = np.nanmean(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    ax.scatter(data[:,0], data[:,4], color=colors[c], marker=markers[c], label = genotypes[c])
    ax.scatter(data[:,0], data[:,4], color=(1,1,1), marker=markers[c])
    for i in xrange(len(data[:,0])):
        ax.scatter(data[i,0], data[i,4], color=colors[c],s=30*(i+1),\
                   marker=markers[c])
ax.legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=1).draggable()
plt.setp(ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
plt.setp(ax,**ax00)



geoVsnTracks = []
for c,gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    geoVsnTracks.append(np.array((data[:,0],data[:,4])).transpose())


fig = plt.figure()
ax = fig.gca(projection='3d')
for i, d in enumerate(geoVsnTracks):
    ax.plot(d[:,0], np.arange(5),d[:,1], marker=markers[i], color=colors[i], label=genotypes[i])

ax.legend()

plt.show()

nParams
sheetNames = ['NumTracks','TrackDuration','TotalDistance',\
              'AvSpeed','GeotacticIndex', 'Straightness']

columnHeader = 'TimePoint'
skipheaderCells = 2

#---- Save sheet for Per minute data------------
paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]
for g, gt in enumerate(allGenotypePerUT_Data):
    for f, fly in enumerate(gt):
        for timepoint in xrange(fly.shape[0]):
            for parameter in xrange(fly.shape[1]):
                col = g+(timepoint*(len(allGenotypePerUT_Data)+xlGapColumns))
                if f==0:
                    if g==0:
                        timepointHeader =  '%s: %d minute'%(columnHeader, timepoint+1)
                        sheets[parameter].write(f,col+len(allGenotypePerUT_Data)/2,timepointHeader)
                    sheets[parameter].write(f+1, col, genotypes[g])
                row = f+skipheaderCells
                sheets[parameter].write(row,col, fly[timepoint, parameter])
paramBook.save("/media/aman/data/thesis/ClimbingPaper/data/climbingPerMinuteParameters_genotypesTogether_.xls")


paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]
for g, gt in enumerate(allGenotypePerUT_Data):
    for f, fly in enumerate(gt):
        for timepoint in xrange(fly.shape[0]):
            for parameter in xrange(fly.shape[1]):
                col = timepoint+(g*(len(allGenotypePerUT_Data)+xlGapColumns))
                if f==0:
                    timepointHeader =  '%d minute'%(timepoint+1)
                    sheets[parameter].write(f+1,col,timepointHeader)
                    if timepoint==0:
                        sheets[parameter].write(f, col+len(allGenotypePerUT_Data)/2, genotypes[g])
                row = f+skipheaderCells
                sheets[parameter].write(row,col, fly[timepoint, parameter])
    
paramBook.save("/media/aman/data/thesis/ClimbingPaper/data/climbingPerMinuteParameters_timepointsTogether_.xls")

#---- Save sheet for 5minutes data------------
genotypeParams = [genotypeNTracks,
                  genotypeLenTrack,
                  genotypeDis,
                  genotypeAvSpeed,
                  genotypeGeoTacInd,
                  genotypeStraight]
paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]

for s in xrange(len(sheets)):
    sheet = sheets[s]
    for g in xrange(len(genotypeParams[s])):
        for row in xrange(len(genotypeParams[s][g])):
            if row==0:
                sheet.write(row,g,genotypes[g])
            sheet.write(row+skipheaderCells,g,genotypeParams[s][g][row])
paramBook.save("/media/aman/data/thesis/ClimbingPaper/data/climbingParameters_genotypesTogether_.xls")



# csDip = dip.dip(genotypeAvSpeed[0])[0]

# dop2RDip = dip.dip(genotypeAvSpeed[1])[0]

# park25Dip = dip.dip(genotypeAvSpeed[2])[0]

# pink1Dip = dip.dip(genotypeAvSpeed[3])[0]

# trp_GDip = dip.dip(genotypeAvSpeed[4])[0]




'''

colors = [(0.12549019607843137, 0.12549019607843137, 0.12549019607843137, alfa),
          (0.65098039215686274, 0.15098039215686274, 0.1529411764705882, alfa),
          (0.15098039215686274, 0.6184313725490196, 0.1784313725490196, alfa),
          (0.1019607843137255, 0.1019607843137255, 0.6174509803921569, alfa),
          (0.8274509803921569, 0.61098039215686274, 0.25098039215686274, alfa),
          (0.5019607843137255, 0.8784313725490196, 0.25098039215686274, alfa),
          (0.5019607843137255, 0.3764705882352941, 0.12549019607843137, alfa),
          (0.8784313725490196, 0.25098039215686274, 0.3764705882352941, alfa),
          (0.5019607843137255, 0.6274509803921569, 0.8784313725490196, alfa),
          (0.5019607843137255, 0.5019607843137255, 0.5019607843137255, alfa)]



allInsSpeeds = []

for i, d in enumerate(allFlyStats):
    a = None
    for x, ids in enumerate(d[1][0]):
        if x==0:
            a = ids
        else:
            a = np.vstack((a, ids))
    allInsSpeeds.append((x+1,a))

krus = stats.kruskal(allInsSpeeds[0][1][:,6],
              allInsSpeeds[1][1][:,6],
              allInsSpeeds[2][1][:,6],
              allInsSpeeds[3][1][:,6],
              allInsSpeeds[4][1][:,6]
              )
print krus


print stats.f_oneway(
              allInsSpeeds[0][1][:,6],
              allInsSpeeds[1][1][:,6],
              allInsSpeeds[2][1][:,6],
              allInsSpeeds[3][1][:,6],
              allInsSpeeds[4][1][:,6]
              )


fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allInsSpeeds):
    try:
        ax.boxplot(da[1][:,6], positions=[x+0.125])
    except:
        pass
    ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(da[1])), da[1][:,6],\
                s=2, color = (0.25,0,0), alpha=0.11, linewidths=1, edgecolors=(0,0,1) )
plt.ylim(0,35)
plt.xlim(-1,5)
plt.show()
'''
# def plotThetaDis(csvAngData, tracklenthresh, c, angs, consecStep, bodyLen):
#     f, axarr = plt.subplots(4,1)
#     im = np.ones((450, 1280, 3), dtype='uint8')*255
#     for i, ids in enumerate(csvAngData):
#         if np.sum(ids[:,3])>tracklenthresh:
#             x = len(ids)
#             #c[i] = (0,0,0)
#             axarr[0].plot(np.arange(len(angs[i])), angs[i][:, -1], '. ', color = c[i], alpha = 0.2)
#             axarr[1].plot(np.arange(len(angs[i])), angs[i][:, 0],'.',color = c[i], alpha = 0.2)
#             axarr[2].plot(angs[i][:, -1], angs[i][:, 0], '.' ,color = colors[i], alpha = 0.1)
#             for p in xrange(x):
#                 cv2.circle(im, (int(ids[p,0:2][0]), int(ids[p,0:2][1])), 2, [col*255 for col in c[i]], -1)
#     axarr[3].imshow(im)
#     axarr[0].set_ylabel('speed\n(Body length units per second)')
#     axarr[1].set_ylabel('angle')
#     axarr[0].set_xlabel('Body Length Units')
#     axarr[1].set_xlabel('Body Length Units')
#     axarr[1].set_ylim(200,-10)
#     labels = [item.get_text() for item in axarr[1].get_xticklabels()]
#     labels = np.array(np.linspace(0, (axarr[1].get_xlim()[-1]*consecStep)/blu, len(labels)), dtype='uint16')
#     axarr[0].set_xticklabels(labels)
#     axarr[1].set_xticklabels(labels)
#     axarr[0].set_ylim(-1,30)
#     plt.show()

# def plotHist(allflystats, bins, ax):
#     for i, ids in enumerate(allflystats):
#         ax.hist(ids[:, 6], bins, alpha=0.2)
#     return ax

# def plots(allflystats):
#     marker = '.'
#     marker2 = '.'
#     maxAngle = 200
#     minAngle = 0
#     f, axarr = plt.subplots(4,2)
#     for i, ids in enumerate(allflystats):
#         axarr[0,0].plot(np.arange(len(ids)), ids[:, 0], marker, alpha = 0.2)# plot mean angles
#         axarr[1,0].plot(np.arange(len(ids)), ids[:, 3], marker, alpha = 0.2) # plot mean distance
#         axarr[2,0].plot(np.arange(len(ids)), ids[:, 6], marker, alpha = 0.2) # plot mean speeds
#         axarr[3,0].plot(ids[:, 6], ids[:, 0], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        
#         axarr[0,1].plot(np.arange(len(ids)), ids[:, 1], marker, alpha = 0.2) # plot median angle
#         axarr[1,1].plot(np.arange(len(ids)), ids[:, 4], marker, alpha = 0.2) # plot mean instaneous velocities
#         axarr[2,1].plot(np.arange(len(ids)), ids[:, 7], marker, alpha = 0.2) # plot median speed
#         axarr[3,1].plot(ids[:, 7], ids[:, 1], marker2, alpha = 0.1) # plot median angle vs. median speed
        
#         axarr[1,0].set_ylim(axarr[1,1].get_ylim())
#         axarr[2,0].set_ylim(axarr[2,1].get_ylim())
#         axarr[3,0].set_xlim(axarr[3,1].get_xlim())
#         axarr[3,0].set_ylim(minAngle,maxAngle)
#         axarr[3,1].set_ylim(0,200)
#         axarr[0,0].set_ylim(minAngle,maxAngle)
#         axarr[0,1].set_ylim(0,200)
#         axarr[0,1].set_xlim(axarr[1,1].get_xlim())
#     plt.show()
#     f, axarr = plt.subplots(4,2)
#     for i, ids in enumerate(allflystats):
#         axarr[0,0].plot(np.arange(len(ids)), ids[:, 0], marker, alpha = 0.2)# plot mean angles
#         axarr[1,0].plot(ids[:, 3], ids[:, 4], marker2, alpha = 0.1) #plot mean angle vs. mean speed
#         axarr[2,0].plot(ids[:, 6], ids[:, 7], marker2, alpha = 0.1) #plot mean angle vs. mean speed
#         axarr[3,0].plot(ids[:, 6], ids[:, 0], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        
#         axarr[0,1].plot(np.arange(len(ids)), ids[:, 1], marker, alpha = 0.2) # plot median angle
#         axarr[1,1].plot(np.arange(len(ids)), ids[:, 3], marker, alpha = 0.2) # plot mean instaneous velocities
#         axarr[2,1].plot(np.arange(len(ids)), ids[:, 4], marker, alpha = 0.2) # plot median speed
#         axarr[3,1].plot(ids[:, 7], ids[:, 1], marker2, alpha = 0.1) # plot median angle vs. median speed
        
#         axarr[1,0].set_ylim(axarr[1,1].get_ylim())
#         axarr[2,0].set_ylim(axarr[2,1].get_ylim())
#         axarr[3,0].set_xlim(axarr[3,1].get_xlim())
#         axarr[3,0].set_ylim(minAngle,maxAngle)
#         axarr[3,1].set_ylim(0,200)
#         axarr[0,0].set_ylim(minAngle,maxAngle)
#         axarr[0,1].set_ylim(0,200)
#         axarr[0,1].set_xlim(axarr[1,1].get_xlim())
#     plt.show()


# ylabelsUT = ['No. of tracks', 'Distance travelled','Duration of track','Average speed','Geotactic Index']


# allGenotypePerUT_Data = []
# for _,genotype in enumerate(allGenotypeMovementData):
#     genotypePerUT_Data = []
#     for i,fly in enumerate(genotype):
#         flyDataPerUT = np.zeros((nUnitTimes, 5))
#         nTracks = 0
#         for t in xrange(nUnitTimes):
#             unitDataIndex = -1
#             if fly[unitDataIndex][t].size > 0:
#                 nTracks+=len(fly[unitDataIndex][t])
#                 flyDataPerUT[t,0] = (len(fly[unitDataIndex][t]))# nTracks
#                 flyDataPerUT[t,1] = (np.nansum(fly[unitDataIndex][t][:,1])) # Total Distance
#                 flyDataPerUT[t,2] = (getLenTrackStats(fly[unitDataIndex][t][:,3])) # duration of track
#                 flyDataPerUT[t,3] = (np.nanmean(fly[unitDataIndex][t][:,0])) #Average Speed
#                 flyDataPerUT[t,4] = (fly[unitDataIndex][t][-1, 2])/nTracks # Geotactic Index
#         genotypePerUT_Data.append(flyDataPerUT)
#     allGenotypePerUT_Data.append(np.array(genotypePerUT_Data))

# #---plot for the per unit time data ----
# fig, ax = plt.subplots(5,1)
# for c,gt in enumerate(allGenotypePerUT_Data):
#     data = np.nanmean(gt[:], axis=0)
#     std = np.nanstd(gt[:], axis=0)
#     for i in xrange(len(data[0])):
#         ax[i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=std[:,i], color=colors[c], fmt='-'+markers[c], label=genotypes[c])
#         ax[i].set_ylabel(ylabelsUT[i])
#     # plt.suptitle(genotypes[c])
#     ax[-1].set_xlabel('minutes')
#     ax[0].legend(loc='right', shadow=True, edgecolor=(0,0,0,)).draggable()
#     # plt.setp([axs for axs in ax], axs.get_yticklabels(), rotation=90)

# scatterDataWidth = 0.062

# ax0 = {'title': plotTitles[0], 'ylabel': plotYLabels[0]}
# ax1 = {'title': plotTitles[1], 'yticks': [0, 5000,10000, 15000, 20000], 'ylabel': plotYLabels[1], \
#        'yticklabels': [0, '5,000','10,000', '15,000', '20,000'] }
# ax2 = {'title': plotTitles[2], 'ylabel': plotYLabels[2], 'yticks': np.arange(0, 525, 35), \
#        'yticklabels':  np.arange(15) }
# ax3 = {'title': plotTitles[3], 'xticks': [1,2,3,4,5], 'ylabel': plotYLabels[3]}
# ax4 = {'title': plotTitles[4], 'yticks': [-1, 0, 1], 'ylabel': plotYLabels[4], 'ylim': (1.2, -1.2) }


# sSize = 20
# sMarker = 'o'
# sAlpha = 0.8
# sLinewidth = 1
# sEdgCol = (0,0,0)
# sCol = (1,1,1)

# bAlpha = 0.6
# vAlpha = 0.5

# fig, ax = plt.subplots(1,5)
# bp0 = ax[0].boxplot([d for d in genotypeNTracks], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeNTracks):
#     ax[0].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# bp1 = ax[1].boxplot([d for d in genotypeDis], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeDis):
#     ax[1].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# bp2 = ax[2].boxplot([d for d in genotypeLenTrack], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeLenTrack):
#     ax[2].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# for x, da in enumerate(genotypeAvSpeed):
#     ax[3].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )
# violPlot = ax[3].violinplot([da for da in genotypeAvSpeed], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')

# bp4 = ax[4].boxplot([d for d in genotypeGeoTacInd], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeGeoTacInd):
#     ax[4].scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# for bplot in (bp0, bp1, bp2, bp4):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(bAlpha)
# for pc, color in zip(violPlot['bodies'], colors):
#     pc.set_color(color)
#     pc.set_alpha(vAlpha)
# violPlot['cmedians'].set_color('orange')
# plt.setp(ax[0], **ax0)
# plt.setp(ax[1], **ax1)
# plt.setp(ax[2], **ax2)
# plt.setp(ax[3], **ax3)
# plt.setp(ax[4], **ax4)
# plt.setp(ax[1].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
# plt.setp(ax[4].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')

# plt.setp([axs for axs in ax], xticklabels = genotypes, xlabel = 'Genotype')
# plt.setp(ax[3], xticklabels = genotypes, xlabel = 'Genotype')
# plt.setp([axs.get_xticklabels() for axs in ax], rotation=45, horizontalalignment='center')
# plt.setp(ax[3].get_xticklabels(), rotation=45, horizontalalignment='center')
# plt.show()

# totalnTracks = []
# totalDis = []

# for nT, d in zip(genotypeNTracks, genotypeDis):
#     s, i, r, p, e = stats.linregress(nT, d)
#     print s, i, r, p ,e
#     totalnTracks.append(nT)
#     totalDis.append(d)
#     plt.plot(nT, d,  marker = 'o')
# plt.show()




# ax0 = {'title': plotTitles[0], 'ylabel': plotYLabels[0]}
# ax1 = {'title': plotTitles[1], 'yticks': [0, 5000,10000, 15000, 20000], 'ylabel': plotYLabels[1], \
#        'yticklabels': [0, '5,000','10,000', '15,000', '20,000'] }
# ax2 = {'title': plotTitles[2], 'ylabel': plotYLabels[2], 'yticks': np.arange(0, 525, 35), \
#        'yticklabels':  np.arange(15) }
# ax3 = {'title': plotTitles[3], 'ylabel': plotYLabels[3]}
# ax4 = {'title': plotTitles[4], 'yticks': [-1, 0, 1], 'ylabel': plotYLabels[4], 'ylim': (1.2, -1.2) }

# sSize = 20
# sMarker = 'o'
# sAlpha = 0.99
# sLinewidth = 1
# sEdgCol = (0,0,0)
# sCol = (1,1,1)

# bAlpha = 0.5
# vAlpha = 0.7

# fig, ax = plt.subplots(1,5)
# bp0 = ax[0].boxplot(genotypeNTracks[0], sym='', notch=True, patch_artist=True)
# for x, da in enumerate(genotypeNTracks[0]):
#     ax[0].scatter(np.linspace(scatterDataWidth+1, -scatterDataWidth+1,len(genotypeNTracks[0])), genotypeNTracks[0],\
#                 s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# bp1 = ax[1].boxplot(genotypeDis[0], sym='', notch=True, patch_artist=True)
# ax[1].scatter(np.linspace(scatterDataWidth+1, -scatterDataWidth+1,len(genotypeDis[0])), genotypeDis[0],\
#             s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# bp2 = ax[2].boxplot(genotypeLenTrack[0], sym='', notch=True, patch_artist=True)
# ax[2].scatter(np.linspace(scatterDataWidth+1, -scatterDataWidth+1,len(genotypeLenTrack[0])), genotypeLenTrack[0],\
#             s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# ax[3].scatter(np.linspace(scatterDataWidth+1, -scatterDataWidth+1,len(genotypeAvSpeed[0])), genotypeAvSpeed[0],\
#             s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )
# violPlot = ax[3].violinplot(genotypeAvSpeed[0], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')

# bp4 = ax[4].boxplot(genotypeGeoTacInd[0], sym='', notch=True, patch_artist=True)
# ax[4].scatter(np.linspace(scatterDataWidth+1, -scatterDataWidth+1,len(genotypeGeoTacInd[0])), genotypeGeoTacInd[0],\
#             s=sSize, color = sCol, marker=sMarker, alpha=sAlpha, linewidths=sLinewidth, edgecolors=sEdgCol )

# for bplot in (bp0, bp1, bp2, bp4):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(bAlpha)
# for pc, color in zip(violPlot['bodies'], colors):
#     pc.set_color(color)
#     pc.set_alpha(vAlpha)

# violPlot['cmedians'].set_color('orange')

# plt.setp([axs for axs in ax], xticklabels = genotypes, xlabel = 'Genotype')
# plt.setp(ax[3], xticks = [1], xlabel = 'Genotype')
# plt.setp([axs.get_xticklabels() for axs in ax], rotation=45, horizontalalignment='center')

# plt.setp(ax[0], **ax0)
# plt.setp(ax[1], **ax1)
# plt.setp(ax[2], **ax2)
# plt.setp(ax[3], **ax3)
# plt.setp(ax[4], **ax4)

# plt.setp(ax[1].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
# plt.setp(ax[4].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')

# plt.show()



#---plot for the per unit time data ----
# plotTitles = ['Number of Tracks',
#               'Total Distance Travelled',
#               'Duration of Tracks',
#               'Average Speed',
#               'Geotactic Index'
#               ]

# plotYLabels = ['Number of Tracks',
#                 'Body Lengths',
#                 'Seconds',
#                 'Body Length / S',
#                 'Geotactic Index'
#                 ]


# ax0 = {'title': plotTitles[0],\
#        'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], \
#        'ylabel': plotYLabels[0], 'yticks': [0, 1, 2, 3, 4, 5] }
# ax1 = {'title': plotTitles[1], \
#        'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], \
#         'ylabel': plotYLabels[1], 'yticks': [0, 1000, 2000, 3000, 4000], 'yticklabels': ['', 1000, '', 3000, ''] }
# nSecs = 8
# ax2 = {'title': plotTitles[2], \
#        'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], \
#        'yticklabels':  np.arange(nSecs), 'ylabel': plotYLabels[2], 'yticks': np.arange(0, 35*nSecs, 35) }
# ax3 = {'title': plotTitles[3], 'ylabel': plotYLabels[3], \
#        'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], \
#        'yticks': [0, 1, 2, 3, 4, 5, 6] }
# ax4 = {'title': plotTitles[4], \
#        'xticks': [0,1,2,3,4], 'xticklabels':[1,2,3,4,5], 'xlabel': 'minutes',\
#        'ylim': (1.2, -1.2), 'yticks': [-1, -0.5, 0, 0.5, 1], 'ylabel': plotYLabels[4] }


# for c,gt in enumerate(allGenotypePerUT_Data):
#     fig, ax = plt.subplots(1,5)
#     #fig.tight_layout()
#     data = np.nanmean(gt[:], axis=0)
#     std = np.nanstd(gt[:], axis=0)
#     sem = stats.sem(gt[:], axis=0)
#     for i in xrange(len(data[0])):
#         ax[i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], color=colors[c], fmt='-'+markers[c])
#         ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                        alpha=0.5)
#     plt.setp(ax[0], **ax0)
#     plt.setp(ax[1], **ax1)
#     plt.setp(ax[2], **ax2)
#     plt.setp(ax[3], **ax3)
#     plt.setp(ax[4], **ax4)
#     plt.suptitle(genotypes[c])

# for c,gt in enumerate(allGenotypePerUT_Data):
#     fig, ax = plt.subplots(1,5)
#     #fig.tight_layout()
#     data = np.nanmean(gt[:], axis=0)
#     std = np.nanstd(gt[:], axis=0)
#     sem = stats.sem(gt[:], axis=0)
#     for i in xrange(len(data[0])):
#         ax[i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], color=colors[c], fmt='-'+markers[c])
#         ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                        alpha=0.5)
#     plt.setp(ax[0], **ax0)
#     plt.setp(ax[1], **ax1)
#     plt.setp(ax[2], **ax2)
#     plt.setp(ax[3], **ax3)
#     plt.setp(ax[4], **ax4)
#     plt.suptitle(genotypes[c])

# axs = [ax0, ax1, ax2, ax3, ax4]
# w = 5
# h = 3
# figRes = 600
# figDir = '/media/aman/data/thesis/ClimbingPaper/Figures/'
# for i in xrange(5):
#     fig = plt.figure(frameon=False, figsize=(w,h))
#     ax = plt.Axes(fig, [0.15,0.15,0.66, 0.70])
#     fig.add_axes(ax)
#     for c,gt in enumerate(allGenotypePerUT_Data):
#         #fig.tight_layout()
#         data = np.nanmean(gt[:], axis=0)
#         std = np.nanstd(gt[:], axis=0)
#         sem = stats.sem(gt[:], axis=0)
#         ax.errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], color=colors[c], fmt='-'+markers[c], label=genotypes[c])
#         ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                            alpha=0.5)
#         plt.setp(ax, **axs[i])
#         plt.xlabel('minutes')
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
        
#         # Only show ticks on the left and bottom spines
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         plt.legend(bbox_to_anchor=(1.002, 1), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small').draggable()
#         plt.savefig('%sFigure%s.pdf'%(figDir, figTitles[i]), dpi=figRes)
#         plt.savefig('%sFigure%s.png'%(figDir, figTitles[i]), dpi=figRes)
        
# scatterDataWidth = 0.125

# genotypes1 = ['','CS','Dop2R','Park25','PINK1RV', 'Trp-Gamma']

# fig, ax = plt.subplots()

# violPlot = ax.violinplot([da for da in genotypeAvSpeed], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')
# for x, da in enumerate(genotypeAvSpeed):
#     ax.scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=2, color = (0.25,0,0), alpha=0.5, linewidths=1, edgecolors=(0,0,1) )
# ax.set_title('Average Speed')
# ax.set_ylabel('bodylength / s')
# ax.set_xticklabels(genotypes1, rotation=90)
# for pc, color in zip(violPlot['bodies'], colors):
#     pc.set_color(color)
# # plt.rcParams["figure.figsize"]= (4,7)
# plt.show()

# #----Total distance----------
# # ax[1].violinplot([d for d in genotypeDis])
# fig, ax = plt.subplots()

# bp1 = ax.boxplot([d for d in genotypeDis], notch=True, patch_artist=True)
# for x, da in enumerate(genotypeDis):
#     ax.scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=2, color = (0.25,0,0), alpha=0.5, linewidths=1, edgecolors=(0,0,1) )
# ax.set_xticklabels(genotypes, rotation=90)
# ax.set_title('Total Distance Travelled in 300 seconds')
# ax.set_ylabel('body length')
# ax.set_yticks([0, 5000,10000, 15000, 20000])
# ax.set_yticklabels([0, '5,000','10,000', '15,000', '20,000'],rotation='vertical')
# # ax[2].violinplot([d for d in genotypeGeoTacInd], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')
# for patch, color in zip(bp1['boxes'], colors):
#     patch.set_facecolor(color)
# # plt.rcParams["figure.figsize"]= (4,7)
# plt.show()

# fig, ax = plt.subplots()

# bp2 = ax.boxplot([d for d in genotypeGeoTacInd], notch=True, patch_artist=True)
# for x, da in enumerate(genotypeGeoTacInd):
#     ax.scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=2, color = (0.25,0,0), alpha=0.5, linewidths=1, edgecolors=(0,0,1) )
# ax.set_title('GeoTactic Index')
# ax.set_xticklabels(genotypes, rotation=90)
# ax.set_yticks([-1, 0, 1])
# for patch, color in zip(bp2['boxes'], colors):
#     patch.set_facecolor(color)
# # plt.rcParams["figure.figsize"]= (4,7)
# plt.show()


# # ax[3].violinplot([d for d in genotypeNTracks], showmeans=False, showmedians=True, showextrema=False, bw_method='silverman')
# fig, ax = plt.subplots()

# bp3 = ax.boxplot([d for d in genotypeNTracks], notch=True, patch_artist=True)
# for x, da in enumerate(genotypeNTracks):
#     ax.scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=2, color = (0.25,0,0), alpha=0.5, linewidths=1, edgecolors=(0,0,1) )
# ax.set_title('number of Tracks in 300 seconds')
# ax.set_xticklabels(genotypes, rotation=90)
# ax.set_ylabel('number of tracks')
# for patch, color in zip(bp3['boxes'], colors):
#     patch.set_facecolor(color)
# # plt.rcParams["figure.figsize"]= (4,7)
# plt.show()

# fig, ax = plt.subplots()

# bp4 = ax.boxplot([d for d in genotypeLenTrack], notch=True, patch_artist=True)
# for x, da in enumerate(genotypeLenTrack):
#     ax.scatter(np.linspace(x+scatterDataWidth+1, x-scatterDataWidth+1,len(da)), da,\
#                 s=2, color = (0.25,0,0), alpha=0.5, linewidths=1, edgecolors=(0,0,1) )
# ax.set_title('length of Tracks in 300 seconds')
# ax.set_ylabel('Number of frames (35FPS)')
# ax.set_xticklabels(genotypes, rotation=90)
# for patch, color in zip(bp4['boxes'], colors):
#     patch.set_facecolor(color)

# # plt.rcParams["figure.figsize"]= (4,7)
# plt.show()



# for pc, color in zip(violPlot['bodies'], colors):
#     pc.set_color(color)

# for bplot in (bp1, bp2, bp3, bp4):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# plt.show()







