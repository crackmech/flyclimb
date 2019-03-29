#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:46:28 2019

@author: aman
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:03:19 2018

@author: aman

ToDo List START------------------------------
    ---Get BLU BASED DATA---
    ---MAKE FOR MULTIPLE GENOTYPES---
    ---MAKE GTI BASED PLOTS---
    ---MAKE SEX BASED PLOTS---
    GET STATS FOR ALL
    FIX MEDIAN VALUE ALGORITHM
    LINE NUMBER 659 in 20180924_ClimbingPlots_MvsF.....py FOR THE DISTANCE TRAVELLED CONFUSION
    MAKE GENOTYPE KEYS DYNAMIC, SET THEIR COLORS
ToDo List END------------------------------


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

#import csv
#from matplotlib import pyplot as plt
#import numpy as np

import os
import baseFunctions as bf
import baseFunctions_trackStats as bfTrkStats

threshTrackTime         = 300   # in seconds, maximum duration of behaviour to be analysed
threshTrackLenMulti     = 3     # multipler of BLU for minimum trackLength w.r.t BLU
unitTimeDur             = 60    # unit time in seconds for pooling data for timeSeries analysis


headerRowId = bfTrkStats.headerRowId
inCsvHeader = bfTrkStats.csvHeader

colorFemales = (0,0,0)
colorMales = (1,0.1,1)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
             }

pltParamList = ['trackNum', 'trackDurMed', 
                'disTot', 'speed','trkDurTot',
                'straightness', 'gti',
                ]
#============================================================================================================

def readFigFolderFile(figFolderFName, figFolderList):
    figFoldersDict = {}
    with open(figFolderFName, 'r') as f:
        lines = f.readlines()
    for figFold in figFolderList:
        figFoldersDict[figFold] = [line for line in lines if figFold in line]
    return figFoldersDict

#baseDir = '/media/pointgrey/data/flywalk/'
#baseDir = bf.getFolder(baseDir)
baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
fig = 'fig5'
statsFName = baseDir+fig+'_stats_Total5min_Sex_GTI_'+bf.present_time()+'.csv'

csvExt = ['*trackStats*.csv']
figDataFName = baseDir+'figDataFiles.txt'
figFoldersList = readFigFolderFile(figDataFName, [fig])
figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))

pMin = 0.05

print ("=============== Processing for all genotypes =============")
csvDirs = bf.getDirList(baseDir)

totalData  = {}
pooledTotalData  = {}
for genotype in figGenotypes:
    for i_,d in enumerate(csvDirs):
        if genotype == d.split(os.sep)[-1]:
            print ('---Processing for Genotype: %s'%genotype)
            figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
            totData, totUntTmData, pldTotalData, pldTotUntData, pltDataTotal, pltDataUnitTime = \
                                    bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
                                                  threshTrackLenMulti, inCsvHeader, headerRowId,
                                                  colIdPooledDict, sexColors, pltParamList)
            totalData[genotype] = totData
            pooledTotalData[genotype] = pldTotalData

genotypes = totalData.keys()

for gtype in genotypes:
    data = pooledTotalData[gtype]
    sexLabels = ['males', 'females']
    colIdSex = [inCsvHeader.index(x) for x in inCsvHeader if 'trackDetails' in x][0]
    pooledTotalSexData, pltTotalSexData = bfTrkStats.getPooledSexData(data, colIdSex, pltParamList, colIdPooledDict)
    label = 'Sex Comparison of, %s'%gtype
    statsListSex = bfTrkStats.getStats2Grps(pooledTotalSexData, sexLabels, pltParamList, colIdPooledDict, pMin, statsFName, label)
    
    data = totalData[gtype]
    colIdGti = [inCsvHeader.index(x) for x in inCsvHeader if 'geotactic' in x][0]
    gtiLabels = ['posGti', 'negGti']
    pooledTotalGtiData, pltTotalGtiData = bfTrkStats.getPooledGTIData(data, colIdGti, pltParamList,
                                                           colIdPooledDict, inCsvHeader, sexColors)
    label = 'GTI Comparison of, %s'%gtype
    statsListGti = bfTrkStats.getStats2Grps(pooledTotalGtiData, gtiLabels, pltParamList, colIdPooledDict, pMin, statsFName, label)


"""
'''    MAKE GTI BASED PLOTS   '''


colIdGti = [inCsvHeader.index(x) for x in inCsvHeader if 'geotactic' in x][0]
gtiLabels = ['posGti', 'negGti']
data = totalData['CS']
pooledTotalGtiData, pltTotalGtiData = bfTrkStats.getPooledGTIData(data, colIdGti, pltParamList,
                                                       colIdPooledDict, inCsvHeader, sexColors)


sWidth = 0.5
vPlots = []
for i in xrange(len(pltTotalGtiData[gtiLabels[0]])):
    fig, ax = plt.subplots()
    for g, gtiType in enumerate(gtiLabels):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalGtiData[gtiType])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalGtiData[gtiType][i], g, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalGtiData[gtiType][i], [g], showextrema=True)
        bp = plt.boxplot(pltTotalGtiData[gtiType][i], positions=[g])
        vPlots.append(vp)
    plt.xlim(-1,len(gtiLabels))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

'''    MADE GTI BASED PLOTS   '''

#============================================================================================================



'''    MAKE SEX BASED PLOTS   '''



sexLabels = ['males', 'females']
colIdSex = [inCsvHeader.index(x) for x in inCsvHeader if 'trackDetails' in x][0]
data = pooledTotalData['Trp-Gamma']
data = pooledTotalData['CS']
pooledTotalSexData, pltTotalSexData = bfTrkStats.getPooledSexData(data, colIdSex, pltParamList, colIdPooledDict)


sWidth = 0.15
vPlots = []
for i in xrange(len(pltTotalSexData[sexLabels[0]])):
    fig, ax = plt.subplots()
    for s, sexType in enumerate(sexLabels):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalSexData[sexType])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalSexData[sexType][i], s, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.45, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalSexData[sexType][i], [s], showextrema=True)
        bp = plt.boxplot(pltTotalSexData[sexType][i], positions=[s])
        vPlots.append(vp)
    plt.xlim(-1,len(sexLabels))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

'''    MADE SEX BASED PLOTS   '''

#============================================================================================================
"""














