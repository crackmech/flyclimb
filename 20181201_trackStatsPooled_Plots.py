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

colorFemales = (0,0,0)
colorMales = (1,0.1,1)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
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

gtiLabels = ['posGti', 'negGti']
gtype = gtiLabels[0]

pooledTotalGtiData  = {'posGti':gtiPldPos,
                       'negGti':gtiPldNeg}
pltTotalGtiData     = {'posGti':gtiPltPldDataPos,
                       'negGti':gtiPltPldDataNeg}

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

sxdPltPldDataMales = []
for i in xrange(len(pltParamList)):
    sxdPltPldDataMales.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(sortedMales)])
sxdPltPldDataFemales = []
for i in xrange(len(pltParamList)):
    sxdPltPldDataFemales.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(sortedFemales)])

sexLabels = ['males', 'females']
sextype = sexLabels[0]

pooledTotalGtiData  = {'males':sortedMales,
                       'females':sortedFemales}
pltTotalGtiData     = {'males':sxdPltPldDataMales,
                       'females':sxdPltPldDataFemales}

sWidth = 0.15
vPlots = []
for i in xrange(len(pltTotalGtiData[sextype])):
    fig, ax = plt.subplots()
    for s, sextype in enumerate(sexLabels):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalGtiData[sextype])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalGtiData[sextype][i], s, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.45, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalGtiData[sextype][i], [s], showextrema=False)
        bp = plt.boxplot(pltTotalGtiData[sextype][i], positions=[s])
        vPlots.append(vp)
    plt.xlim(-1,len(sexLabels))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

'''    MADE SEX BASED PLOTS   '''

#============================================================================================================


""" ---Get BLU BASED DATA---
    ---MAKE FOR MULTIPLE GENOTYPES---
    ---MAKE GTI BASED PLOTS---
    ---MAKE SEX BASED PLOTS---
    GET STATS FOR ALL
    FIX MEDIAN VALUE ALGORITHM
    LINE NUMBER 659 in 20180924_ClimbingPlots_MvsF.....py FOR THE DISTANCE TRAVELLED CONFUSION
    MAKE GENOTYPE KEYS DYNAMIC, SET THEIR COLORS
"""

























#unitTimeN = threshTrackTime/unitTimeDur
#csvHeader = headerRowId+1
#colIdSex =  [inCsvHeader.index(x) for x in inCsvHeader if 'trackDetails' in x][0]
#colIdBodyLen = [inCsvHeader.index(x) for x in inCsvHeader if 'median body length' in x][0]
#colIdTrackLen = [inCsvHeader.index(x) for x in inCsvHeader if 'distance' in x][0]
#colIdtrackTmPt = [inCsvHeader.index(x) for x in inCsvHeader if 'timeDelta' in x][0]





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




#colIdPooledSexDetails = 0
#colIdPooledSexColor = 1
#colIdPooledTrackNum = 2
#colIdPooledTrkDurMed = 3
#colIdPooledDisTrvlTot = 4
#colIdPooledSpeedAv = 5
#colIdPooledBdAngStd = 6
#colIdPooledBdLenAv = 7
#colIdPooledStrght = 8
#colIdPooledGti = 9
#colIdPooledLtncyMed = 10
#colIdPooledTrkDurTot = 11
#colIdPooledDisTrvlPertrk = 12
#colIdPooledBlu = 13
#colIdPooledfps = 14
##------test plot
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledTrackNum)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(0,5)
#plt.show()
#
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledDisTrvlTot)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(0,100)
#plt.show()
#
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledTrkDurMed)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(0,8)
#plt.show()
#
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledSpeedAv)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(0,10)
#plt.show()
#
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledStrght)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(0,1.2)
#plt.show()
#
#pltData, pltDataerr = getUnitTimePltData(pooledUnitTimeData, colIdPooledGti)
#plt.errorbar(np.arange(len(pltData)), pltData, pltDataerr, alpha=0.7)
#plt.ylim(1.2,-1.2)
#plt.show()
##------test plot
#
#
#
#
##------test plot
#
##---test plot- #Tracks
#fig, ax = plt.subplots()
#scPlt1 = bf.plotScatter(ax,[x[colIdPooledTrackNum] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(0,35)
#plt.show()
#
##---test plot- Total Distance Travelled
#fig, ax = plt.subplots()
#scPlt2 = bf.plotScatter(ax,[x[colIdPooledDisTrvlTot] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(0,800)
#plt.show()
#
##---test plot- Median track Duration
#fig, ax = plt.subplots()
#scPlt3 = bf.plotScatter(ax,[x[colIdPooledTrkDurMed] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(0,10)
#plt.show()
#
##---test plot- Average Speed
#fig, ax = plt.subplots()
#scPlt4 = bf.plotScatter(ax,[x[colIdPooledSpeedAv] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(0,20)
#plt.show()
#
##---test plot- Average Straightness
#fig, ax = plt.subplots()
#scPlt5 = bf.plotScatter(ax,[x[colIdPooledStrght] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(0,1.2)
#plt.show()
#
##---test plot- Average GTI
#fig, ax = plt.subplots()
#scPlt6 = bf.plotScatter(ax,[x[colIdPooledGti] for i_,x in enumerate(pooledTotalData)], np.arange(1), 
#                           scatterRadius=20, scatterColor=colorSex, scatterEdgeColor=(1,1,1),scatterAlpha=0.9)
#plt.xlim(-1,5)
#plt.ylim(1.2,-1.2)
#plt.show()
#
##------test plot






##---- read all stats data from each fly folder in a genotype folder ----#
#dirs = bf.getDirList(baseDir)
#genotypeData = []
#for i,d in enumerate(dirs):
#    fList = bf.getFiles(d, csvExt)
#    for _, f in enumerate(fList):
#        trackDataOutput = bf.readCsv(f)[csvHeader:]
#        flyStats = []
#        blu = np.nanmean(np.array([x[colIdBodyLen] for _,x in enumerate(trackDataOutput)], dtype=np.float64))
#        threshTrackLen = blu*threshTrackLenMulti
#        #print d,i, blu, threshTrackLen
#        for ix,x in enumerate(trackDataOutput):
#            if np.float(x[colIdTrackLen])>=threshTrackLen and np.float(x[colIdtrackTmPt])<threshTrackTime:
#                flyStats.append(x)
#    genotypeData.append(flyStats)
##---- read all stats data from each fly folder in a genotype folder ----#
#
#
##---- convert all data into a list of unitTimePoints with wach element containing raw
##       data from the stats csv file for each fly in a genotype folder ----#
#genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]
#pooledUnitTimeData = [[] for x in xrange(unitTimeN)]
#for d_,d in enumerate(genotypeData):
#    #print d[0][0]
#    for i in xrange(unitTimeN):
#        unitTimeData = []
#        for _,f in enumerate(d):
#            if i*unitTimeDur<=np.float(f[colIdtrackTmPt])<(i+1)*unitTimeDur:
#                unitTimeData.append(f)
#        genotypeUnitTimeData[i].append(unitTimeData)
#        if len(unitTimeData)>0:
#            pooledUnitTimeData[i].append(bfTrkStats.getPooledData(unitTimeData, inCsvHeader, sexColors))
#
##---- convert all data into a list of unitTimePoints with wach element containing raw
##       data from the stats csv file for each fly in a genotype folder ----#
#
#
#
##---- Plot the total data from behaviour from total time measured ----#
#pooledTotalData = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors) for i_,x in enumerate(genotypeData) if len(x)>0]
#colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData)]
#
#pltDataUnitTime = []
#pltDataTotal = []
#for i in xrange(len(pltParamList)):
#    print pltParamList[i], colIdPooledDict[pltParamList[i]]
#    pltDataUnitTime.append(getUnitTimePltData(pooledUnitTimeData, colIdPooledDict[pltParamList[i]]))
#    pltDataTotal.append([x[colIdPooledDict[pltParamList[i]]] for i_,x in enumerate(pooledTotalData)])





#sWidth = 0.5
#vPlots = []
#for i in xrange(len(pltTotalData[gtype])):
#    fig, ax = plt.subplots()
#    scPlt1 = bf.plotScatterCentrd(ax,pltTotalData[gtype][i], np.arange(1), \
#                                  scatterRadius=10, scatterColor=colorSex, \
#                                  scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
#                                  scatterWidth = sWidth)
#    vp = plt.violinplot(pltTotalData[gtype][i], np.arange(1), showextrema=False)
#    vPlots.append(vp)
#    plt.xlim(-1,5)
#    plt.title(str(i)+'_'+pltParamList[i])
#plt.show()
#
#


##---- Plot the total data from behaviour from total time measured ----#
#gtype = 'CS'
##------test plot
#colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
#for i in xrange(len(pltTotalData[gtype])):
#    fig, ax = plt.subplots()
#    scPlt1 = bf.plotScatter(
#                ax,pltTotalData[gtype][i], np.arange(1), 
#                scatterRadius=20, scatterColor=colorSex,\
#                scatterEdgeColor=(1,1,1),scatterAlpha=0.9
#                )
#    plt.xlim(-1,5)
#    plt.title(str(i)+'_'+pltParamList[i])
#plt.show()
#
##---- Plot the timeSeries data from behaviour from total time measured ----#
#for i in xrange(len(pltTmSrsData[gtype])):
#    pltData, pltDataErr = pltTmSrsData[gtype][i]
#    fig, ax = plt.subplots()
#    ax.errorbar(np.arange(len(pltData)), pltData, pltDataErr, alpha=0.7)
#    plt.title(str(i)+'_'+pltParamList[i])
#plt.show()
#
##------test plot


