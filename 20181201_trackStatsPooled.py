#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:03:19 2018

@author: aman

"""
#import os
#from datetime import datetime
#from datetime import timedelta
#import csv

import baseFunctions as bf
import baseFunctions_trackStats as bfTrkStats
import numpy as np
from matplotlib import pyplot as plt

headerRowId = bfTrkStats.headerRowId
csvHeader = headerRowId+1

inCsvHeader = bfTrkStats.csvHeader
colIdSex =  [inCsvHeader.index(x) for x in inCsvHeader if 'trackDetails' in x][0]
colIdBodyLen = [inCsvHeader.index(x) for x in inCsvHeader if 'median body length' in x][0]
colIdTrackLen = [inCsvHeader.index(x) for x in inCsvHeader if 'distance' in x][0]
colIdtrackTmPt = [inCsvHeader.index(x) for x in inCsvHeader if 'timeDelta' in x][0]

'''
        0)  Determine sex, male or Female
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
colIdPooledSexColor = 1
colIdPooledSexDetails = 1
colIdPooledTrackNum = 2
colIdPooledTrkDurMed = 3
colIdPooledDisTrvlTot = 4
colIdPooledSpeedAv = 5
colIdPooledBdAngStd = 6
colIdPooledBdLenAv = 7
colIdPooledStrght = 8
colIdPooledGti = 9
colIdPooledLtncyMed = 10
colIdPooledTrkDurTot = 11
colIdPooledDisTrvlPertrk = 12
colIdPooledBlu = 13
colIdPooledfps = 14


colorMales = (1,0.1,1)
colorFemales = (0,0,0)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
             }

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


#---- read all stats data from each fly folder in a genotype folder ----#
genotypeData = []
for i,d in enumerate(dirs):
    fList = bf.getFiles(d, csvExt)
    for _, f in enumerate(fList):
        trackDataOutput = bf.readCsv(f)[csvHeader:]
        flyStats = []
        blu = np.nanmean(np.array([x[colIdBodyLen] for _,x in enumerate(trackDataOutput)], dtype=np.float64))
        threshTrackLen = blu*threshTrackLenMulti
        #print d,i, blu, threshTrackLen
        for ix,x in enumerate(trackDataOutput):
            if np.float(x[colIdTrackLen])>=threshTrackLen and np.float(x[colIdtrackTmPt])<threshTrackTime:
                flyStats.append(x)
    genotypeData.append(flyStats)
#---- read all stats data from each fly folder in a genotype folder ----#


#---- convert all data into a list of unitTimePoints with wach element containing raw
#       data from the stats csv file for each fly in a genotype folder ----#
genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]
pooledUnitTimeData = [[] for x in xrange(unitTimeN)]
for d_,d in enumerate(genotypeData):
    #print d[0][0]
    for i in xrange(unitTimeN):
        unitTimeData = []
        for _,f in enumerate(d):
            if i*unitTimeDur<=np.float(f[colIdtrackTmPt])<(i+1)*unitTimeDur:
                unitTimeData.append(f)
        genotypeUnitTimeData[i].append(unitTimeData)
        if len(unitTimeData)>0:
            pooledUnitTimeData[i].append(bfTrkStats.getPooledData(unitTimeData, inCsvHeader, sexColors))

#---- convert all data into a list of unitTimePoints with wach element containing raw
#       data from the stats csv file for each fly in a genotype folder ----#


#------test plot
def getUnitTimePltData(untTmData, colId):
    '''
    get the data for plotting the timeSeried plots
    '''
    outData = []
    for i,d in enumerate(untTmData):
        outDataPerUT = [x[colId] for i_,x in enumerate(d)]
        outData.append(outDataPerUT)
    pltData = [np.average(x) for i_,x in enumerate(outData)]
    pltDataerr = [np.std(x)/np.sqrt(len(x)) for i_,x in enumerate(outData)]
    return pltData, pltDataerr


#------test plot
pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledTrackNum)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(0,5)
plt.show()

pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledDisTrvlTot)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(0,100)
plt.show()

pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledTrkDurMed)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(0,8)
plt.show()

pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledSpeedAv)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(0,10)
plt.show()

pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledStrght)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(0,1.2)
plt.show()

pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledGti)
plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
plt.ylim(1.2,-1.2)
plt.show()
#------test plot



#---- Plot the total data from behaviour from total time measured ----#
pooledTotalData = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors) for i_,x in enumerate(genotypeData) if len(x)>0]
colorSex = [x[colIdPooledSexColor] for i_,x in enumerate(pooledTotalData)]

#------test plot

#---test plot- #Tracks
fig, ax = plt.subplots()
scPlt1 = bfTrkStats.plotScatter(ax,[x[colIdPooledTrackNum] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(0,35)
plt.show()

#---test plot- Total Distance Travelled
fig, ax = plt.subplots()
scPlt2 = bfTrkStats.plotScatter(ax,[x[colIdPooledDisTrvlTot] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(0,800)
plt.show()

#---test plot- Median track Duration
fig, ax = plt.subplots()
scPlt3 = bfTrkStats.plotScatter(ax,[x[colIdPooledTrkDurMed] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(0,10)
plt.show()

#---test plot- Average Speed
fig, ax = plt.subplots()
scPlt4 = bfTrkStats.plotScatter(ax,[x[colIdPooledSpeedAv] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(0,20)
plt.show()

#---test plot- Average Straightness
fig, ax = plt.subplots()
scPlt5 = bfTrkStats.plotScatter(ax,[x[colIdPooledStrght] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(0,1.2)
plt.show()

#---test plot- Average GTI
fig, ax = plt.subplots()
scPlt6 = bfTrkStats.plotScatter(ax,[x[colIdPooledGti] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
plt.xlim(-1,5)
plt.ylim(1.2,-1.2)
plt.show()

#------test plot

#---- Plot the total data from behaviour from total time measured ----#


""" Get BLU BASED DATA
    FIX MEDIAN VALUE ALGORITHM
    MAKE FOR MULTIPLE GENOTYPES
    GET STATS FOR ALL
    MAKE GTI BASED PLOTS
    MAKE SEX BASED PLOTS
    LINE NUMBER 659 in 20180924_ClimbingPlots_MvsF.....py FOR THE DISTANCE TRAVELLED CONFUSION
"""



##---- convert all data into a list of unitTimePoints with wach element containing raw
##       data from the stats csv file for each fly in a genotype folder ----#
#genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]
#for _,d in enumerate(genotypeData):
#    for i in xrange(unitTimeN):
#        unitTimeData = []
#        for _,f in enumerate(d):
#            if i*unitTimeDur<=np.float(f[colIdtrackTmPt])<(i+1)*unitTimeDur:
#                unitTimeData.append(f)
#        genotypeUnitTimeData[i].append(unitTimeData)
##---- converted all data into a list of unitTimePoints with wach element containing raw
##       data from the stats csv file for each fly in a genotype folder ----#

##------test plot
#nTracks = []
#for i,d in enumerate(pooledUnitTimeData):
#    nTracksPerUT = [x[colIdPooledTrackNum] for i_,x in enumerate(d)]
#    nTracks.append(nTracksPerUT)
#
#pltData = [np.average(x) for i_,x in enumerate(nTracks)]
#pltDataerr = [np.std(x)/np.sqrt(len(x)) for i_,x in enumerate(nTracks)]
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.show()
#
#trkDur = []
#for i,d in enumerate(pooledUnitTimeData):
#    nTracksPerUT = [x[colIdPooledTrkDurMed] for i_,x in enumerate(d)]
#    trkDur.append(nTracksPerUT)
#
#pltData = [np.average(x) for i_,x in enumerate(trkDur)]
#pltDataerr = [np.std(x)/np.sqrt(len(x)) for i_,x in enumerate(trkDur)]
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.show()
#
#
#trkDur = []
#for i,d in enumerate(pooledUnitTimeData):
#    nTracksPerUT = [x[colIdPooledSpeedAv] for i_,x in enumerate(d)]
#    trkDur.append(nTracksPerUT)
#
#pltData = [np.average(x) for i_,x in enumerate(trkDur)]
#pltDataerr = [np.std(x)/np.sqrt(len(x)) for i_,x in enumerate(trkDur)]
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.plot([x+0.10 for i_,x in enumerate(pltData)],'-',alpha=0.6)
##plt.ylim(0,8)
#plt.show()
#
##------test plot
