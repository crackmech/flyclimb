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
from matplotlib import pyplot as plt
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
                'disTot', 'speed',
                'bdAngStd', 'bdLen',
                'straightness', 'gti',
                'latency', 'trkDurTot',
                'disPerTrk', 'blu', 'fps',
                ]
#============================================================================================================

def readFigFolderFile(figFolderFName, figFolderList):
    figFoldersDict = {}
    with open(figFolderFName, 'r') as f:
        lines = f.readlines()
    for figFold in figFolderList:
        figFoldersDict[figFold] = [line for line in lines if figFold in line]
    return figFoldersDict

def createDF(data, labels, genotypes):
    '''
    returns the pandas DataFrame for stats proessing in R
    '''
    prmValDict = {labels[0]:[], labels[1]:[]}
    for i,x in enumerate(data):
        flyLabls = [genotypes[i] for y in xrange(len(x))]
        dfData = [x, flyLabls]
        for j,y in enumerate(labels):
            prmValDict[y].extend(dfData[j])
    return pd.DataFrame(prmValDict, columns=labels)

def r_matrix_to_data_frame(r_matrix):
    """Convert an R matrix into a Pandas DataFrame"""
    array = pandas2ri.ri2py(r_matrix)
    return pd.DataFrame(array,
                        index=r_matrix.names[0],
                        columns=r_matrix.names[1])
                        
def getRKrusWall(formula, data):
    '''
    returns the data analysed by Kruskal wallis in 'R' using rpy2 module
    '''
    krsWall = statsR.kruskal_test(formula=formula, data=data)
    krsWallPd = pd.DataFrame(pandas2ri.ri2py(krsWall.rx2('p.value')))
    pVal = krsWallPd[0]
    postHocDunn = fsa.dunnTest(formula, data=data, method='bh')
    postHoc = pd.DataFrame(pandas2ri.ri2py(postHocDunn.rx2('res')))
    chiSq = pd.DataFrame(pandas2ri.ri2py(krsWall.rx2('statistic')))
    return {'pvalue': pVal, 'chi-squared': chiSq,'posthoc': postHoc.sort_values(by=['Comparison'])}, krsWallPd

def getRAnoval(formula, data):
    '''
    returns the data analysed by Kruskal wallis in 'R' using rpy2 module
    '''
    model1 = robjects.r.lm(formula=frmla, data=data)
    anv = robjects.r.anova(model1)
    postHocHSD = agr.HSD_test(model1, 'genotype', group=False, console=False)
    postHoc = pd.DataFrame(pandas2ri.ri2py(postHocHSD.rx2('comparison')))
    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
    pVal= smry1['Pr(>F)']['genotype']
    fValue = smry1['F value']['genotype']
    return {'pvalue': pVal, 'fvalue': fValue, 'posthoc': postHoc}



#baseDir = '/media/pointgrey/data/flywalk/'
#baseDir = bf.getFolder(baseDir)
baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
statsFName = baseDir+'stats_Total5min.csv'


csvExt = ['*trackStats*.csv']
fig = 'fig5'
figDataFName = baseDir+'figDataFiles.txt'
figFoldersList = readFigFolderFile(figDataFName, [fig])
figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))

print ("=============== Processing for all genotypes =============")
csvDirs = bf.getDirList(baseDir)

totalData  = {}
totalDataTmSrs = {}
pooledTotalData  = {}
pooledTotalDataTmSrs = {}
pltTotalData = {}
pltTmSrsData = {}
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
            totalDataTmSrs[genotype] = totUntTmData
            pooledTotalData[genotype] = pldTotalData
            pooledTotalDataTmSrs[genotype] = pldTotUntData
            pltTotalData[genotype] = pltDataTotal
            pltTmSrsData[genotype] = pltDataUnitTime

genotypes = totalData.keys()
"""
print ('\n#=============== **** PERFORMING STATISTICS **** ===================\n')

import scipy.stats as stats
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from matplotlib import pyplot as plt

pandas2ri.activate()
nlme = importr('nlme')
statsR = importr('stats')
base = importr('base')
multcomp = importr('multcomp')
agr = importr('agricolae')
fsa = importr('FSA')


pMin = 0.05
import copy
gtypes = copy.deepcopy(genotypes)
gtypes.sort()
statsList = {}

grpLbls = ['paramVal','genotype']
frmla = robjects.Formula('paramVal ~ genotype')
for param in pltParamList:
    label = '====='+param+'====='
    print label
    colId = colIdPooledDict[param]
    dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
    df = createDF(dSets, grpLbls, gtypes)
    dfr1 = pandas2ri.py2ri(df)
    normP = [stats.normaltest(dSet)[1] for dSet in dSets]
    print 'Min Normal Dist Value: ',min(normP)
    descStats = fsa.Summarize(frmla, data = dfr1)
    #print descStats
    if min(normP)<pMin:
        print ('Stats: Kruskal-Wallis')
        statsData,a = getRKrusWall(frmla, dfr1)
    else:
        print ('Stats: One Way ANOVA')
        statsData = getRAnoval(frmla, dfr1)
    statsList[param] = statsData
    f = open(statsFName, 'a')
    f.write('\n\n---\n\n')
    f.write(param+'\n')
    f.close()
    statsKeys = statsData.keys()
    statsKeys.sort()
    statsKeys.remove('posthoc')
    statsKeys.insert(len(statsKeys),'posthoc')
    for key in statsKeys:
        f = open(statsFName, 'a')
        f.write(key+': ')
        if key=='posthoc':
            f.write('\n')
            f.close()
            statsData[key].to_csv(statsFName, mode='a', header=True)
        elif key=='chi-squared':
            f.write(', '+str(statsData[key][0][0])+'\n')
        else:
            f.write(', '+str(statsData[key][0])+'\n')
        f.close()
"""



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

gtiPldNeg = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors, x) for i_,x in enumerate(gtiNeg) if len(x)>0]
gtiPldPos = [bfTrkStats.getPooledData(x, inCsvHeader, sexColors, x) for i_,x in enumerate(gtiPos) if len(x)>0]

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

















