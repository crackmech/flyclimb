#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:03:19 2018

@author: aman

Using INPUT: GenotypeDirectory:
    for each fly in a folder, for each track:
        if trackLen>threshTrackLen and tracktimePoint<maxTimeDuration, then,
            for that track, get everything into one list 
    OUTPUT FILE: dataFile

Using INPUT: dataFile:
    Segregate tracks for perUnitTime
    OUTPUT FILE: unitTimeData

Using INPUT: dataFile/unitTimeData:
    For each parameter, get data for plotting for each fly 
    OUTPUT FILE: pooledTotalData/pooledUnitTimeData

Using INPUT: pooledTotalData:
    Plot scatterplots for each parameter for each genotype 

Using INPUT: pooledUnitTimeData
    Get average and error data for each parameter for each timepoint
    OUTPUT FILE: plotPooledUnitTimeData

Using INPUT: plotPooledUnitTimeData:
    Plot for each parameter for each timePoint for each genotype

Segregate and ploteach genotype for:
    Positive and negative geotactic index
    Males and females data


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
"""
#from datetime import datetime
#from datetime import timedelta
#import csv

import numpy as np
import os
import baseFunctions as bf
import baseFunctions_trackStats as bfTrkStats
from matplotlib import pyplot as plt

threshTrackTime         = 300   # in seconds, maximum duration of behaviour to be analysed
threshTrackLenMulti     = 3     # multipler of BLU for minimum trackLength w.r.t BLU
unitTimeDur             = 60    # unit time in seconds for pooling data for timeSeries analysis
headerRowId = bfTrkStats.headerRowId
inCsvHeader = bfTrkStats.csvHeader

colorMales = (1,0.1,1)
colorFemales = (0,0,0)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
             }

colIdPooledDict = {'sex':         0,
                   'sexColor':    1,
                   'trackNum':    2,
                   'trackDurMed': 3,
                   'disTot':      4,
                   'speed':       5,
                   'bdAngStd':    6,
                   'bdLen':       7,
                   'straightness':8,
                   'gti':         9,
                   'latency':     10,
                   'trkDurTot':   11,
                   'disPerTrk':   12,
                   'blu':         13,
                   'fps':         14,
                   }

pltParamList = ['trackNum', 'trackDurMed', 
                'disTot', 'speed',
                'bdAngStd', 'bdLen',
                'straightness', 'gti',
                'latency', 'trkDurTot',
                'disPerTrk', 'blu', 'fps',
                ]

#============================================================================================================
#baseDir = '/media/pointgrey/data/flywalk/'
baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
#baseDir = bf.getFolder(baseDir)
csvExt = ['*trackStats*.csv']

print ("=============== Processing for all genotypes =============")

csvDirs = bf.getDirList(baseDir)

totalData  = {}
totalDataTmSrs = {}
pooledTotalData  = {}
pooledTotalDataTmSrs = {}
pltTotalData = {}
pltTmSrsData = {}
for i_,d in enumerate(csvDirs):
    genotype = d.split(os.sep)[-1]
    totData,untTmData,pldTotalData,pldUntData,pltDataTotal,pltDataUnitTime = \
                            bfTrkStats.pooledData(d, csvExt, unitTimeDur, threshTrackTime,
                                          threshTrackLenMulti, inCsvHeader, headerRowId,
                                          colIdPooledDict, sexColors, pltParamList)
    totalData[genotype] = totData
    totalDataTmSrs[genotype] = untTmData
    pooledTotalData[genotype] = pldTotalData
    pooledTotalDataTmSrs[genotype] = pldUntData
    pltTotalData[genotype] = pltDataTotal
    pltTmSrsData[genotype] = pltDataUnitTime
#============================================================================================================

#------test plot
#---- Plot the total data from behaviour from total time measured ----#
genotypes = totalData.keys()
genotypes = ['CS', 'W1118', 'PINK1RV', 'Park25xW1118', 'W1118xLrrk-ex1', 'Park25xLrrk-ex1', 'Trp-Gamma']
gtype = genotypes[0]

colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]

sWidth = 0.5
vPlots = []
for i in xrange(len(pltTotalData[gtype])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(genotypes):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalData[gtype][i], g, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalData[gtype][i], [g], showextrema=False)
        vPlots.append(vp)
    plt.xlim(-1,len(genotypes))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

#------test plot
#---- Plot the timeSEries data from behaviour from total time measured ----#
for i in xrange(len(pltTmSrsData[gtype])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(genotypes):
        pltData, pltDataErr = pltTmSrsData[gtype][i]
        ax.errorbar(np.arange(len(pltData)), pltData, pltDataErr, alpha=0.7)
        plt.title(str(i)+'_'+pltParamList[i])
plt.show()

#============================================================================================================

'''    MAKE GTI BASED PLOTS   '''

def segregateGti(genotypeData, colIdgti):
    '''
    returns two lists, containing data segregated based on 
            Postive and Negative geotactic Index respectively from input genotypeData
    '''
    gtineg = []
    gtipos = []
    for i,fly in enumerate(genotypeData):
        flyGtineg = []
        flyGtipos = []
        for i_,trk in enumerate(fly):
            if int(trk[colIdgti])==-1:
                flyGtineg.append(trk)
            elif int(trk[colIdgti])==1:
                flyGtipos.append(trk)
        gtineg.append(flyGtineg)
        gtipos.append(flyGtipos)
    return gtineg, gtipos


colIdGti = [inCsvHeader.index(x) for x in inCsvHeader if 'geotactic' in x][0]
data = totalData['Trp-Gamma']
data = totalData['CS']
gtiPos, gtiNeg = segregateGti(data, colIdGti)

gtiPldNeg = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors) for i_,x in enumerate(gtiNeg) if len(x)>0]
gtiPldPos = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors) for i_,x in enumerate(gtiPos) if len(x)>0]

gtiPltPldDataPos = []
for i in xrange(len(pltParamList)):
    gtiPltPldDataPos.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(gtiPldPos)])
gtiPltPldDataNeg = []
for i in xrange(len(pltParamList)):
    gtiPltPldDataNeg.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(gtiPldNeg)])

gtiLabels = ['pos', 'neg']
gtype = gtiLabels[0]

pooledTotalGtiData = {'pos':gtiPldPos,
                      'neg':gtiPldNeg}
pltTotalGtiData = {'pos':gtiPltPldDataPos,
                   'neg':gtiPltPldDataNeg}

sWidth = 0.5
vPlots = []
for i in xrange(len(pltTotalGtiData[gtype])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(gtiLabels):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalGtiData[gtype])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalGtiData[gtype][i], g, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalGtiData[gtype][i], [g], showextrema=False)
        vPlots.append(vp)
    plt.xlim(-1,len(gtiLabels))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

'''    MADE GTI BASED PLOTS   '''

#============================================================================================================

'''    MAKE SEX BASED PLOTS   '''

def segregateSex(pooledgenotypedata, colIdsex):
    '''
    returns two lists, containing data segregated based on 
            sex from input genotypeData
    '''
    males = []
    females = []
    for i,fly in enumerate(pooledgenotypedata):
        if int(fly[colIdsex])==1:
            males.append(fly)
        elif int(fly[colIdsex])==0:
            females.append(fly)
    return males, females


colIdSex = [inCsvHeader.index(x) for x in inCsvHeader if 'trackDetails' in x][0]
data = pooledTotalData['Trp-Gamma']
data = pooledTotalData['CS']
sortedMales, sortedFemales = segregateSex(data, colIdSex)

gtiPltPldDataPos = []
for i in xrange(len(pltParamList)):
    gtiPltPldDataPos.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(sortedMales)])
gtiPltPldDataNeg = []
for i in xrange(len(pltParamList)):
    gtiPltPldDataNeg.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(sortedFemales)])

gtiLabels = ['males', 'females']
gtype = gtiLabels[0]

pooledTotalGtiData = {'males':sortedMales,
                      'females':sortedFemales}
pltTotalGtiData = {'males':gtiPltPldDataPos,
                   'females':gtiPltPldDataNeg}

sWidth = 0.5
vPlots = []
for i in xrange(len(pltTotalGtiData[gtype])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(gtiLabels):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalGtiData[gtype])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalGtiData[gtype][i], g, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalGtiData[gtype][i], [g], showextrema=False)
        vPlots.append(vp)
    plt.xlim(-1,len(gtiLabels))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

'''    MADE SEX BASED PLOTS   '''

#============================================================================================================

