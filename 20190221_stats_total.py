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

import numpy as np
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

#baseDir = '/media/pointgrey/data/flywalk/'
baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
#baseDir = bf.getFolder(baseDir)
csvExt = ['*trackStats*.csv']
fig = 'fig5'
figDataFName = baseDir+'figDataFiles.txt'
figFoldersList = readFigFolderFile(figDataFName, [fig])
figGenotypes = list(set([f.split(os.sep)[1] for f in figFoldersList[fig]]))

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
#genotypes = ['CS', 'W1118', 'PINK1RV', 'Park25xW1118', 'W1118xLrrk-ex1', 'Park25xLrrk-ex1', 'Trp-Gamma']



print ('\n#=============== **** PERFORMING STATISTICS **** ===================\n')

import scikit_posthocs as sp
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.anova import AnovaRM
import pandas as pd


def restrucDataForRMA(dataSet):
    restrucData = [[] for x in dataSet]
    minLen = min([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(minLen):
            restrucData[i].append(np.float(dataSet[i][j]))
    return restrucData

def getKWmultiComp(data, labels, verbose=False):
    pVals = sp.posthoc_dunn(data, p_adjust='bonferroni')
    if verbose:
        print pVals#np.hstack((np.transpose([0]+labels).reshape(4,1),np.vstack((labels,pVals))))
    return pVals#[pVals[1,0], pVals[2,0], pVals[2,1]]

def getOWANOVAmultiComp(data, labels, verbose=False):
    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(data)])
    res = pairwise_tukeyhsd(np.concatenate(data), tlabels)
    if verbose:
        print (res.summary())
    return psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total)
    
def getRMAnova(dataSet, labels, verbose=False):
    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(dataSet)])
    concatData = np.concatenate(dataSet)
    ids = np.concatenate([np.arange(len(x)) for _,x in enumerate(dataSet)])
    d = {'id':ids, 'rt':concatData, 'cond':tlabels}
    df = pd.DataFrame(d)
    anovarm = AnovaRM(df, 'rt', 'id', within=['cond'])
    res = anovarm.fit()
    if verbose:
        print (res.summary())
    return res

def getConcStats(tData, datasetLabels, param, labels, pNormMin, verbose=False):
    for i,x in enumerate(tData):
        label = '---'+param+'_'+labels[i]+'---'
        rma = getRMAnova(x, datasetLabels, verbose)
        if i==0:
            df = rma.anova_table
            df.insert(loc = 0, column = 'state', value = label)
        else:
            print df
            currDf = rma.anova_table
            currDf.insert(loc = 0, column = 'state', value = label)
            df = pd.concat([df, currDf])
    return df

def getStats(tData, datasetLabels, param, labels, pmin, verbose=False):
    #c = datasetLabels[0]
    #e1 = datasetLabels[1]
    #e2 = datasetLabels[2]
    statsData = []
    statsData.append(['Test and Parameter', 'p-Value', 'p-Value', 'p-Value'])
    #statsData.append(['',c+' vs. '+e1, c+' vs. '+e2, e1+' vs. '+e2])
    for i in xrange(len(tData)):
        label = '---'+param+'_'+labels[i]+'---'
        print label
        normP = []
        multiCompP = []
        for j in xrange(len(tData[i])):
            _, pValue = stats.normaltest(tData[i][j])
            normP.append(pValue)
        print 'Min Normal Dist Value: ',min(normP)
        if min(normP)<pmin:
            testUsed = 'Kruskal-Wallis test'
            _, statsP = stats.kruskal(*tData[i])
            print testUsed+' pValue:', statsP,'---'
            if statsP<pmin:
                multiCompP = getKWmultiComp(tData[i], datasetLabels, verbose)
        else:
            testUsed = 'One Way ANOVA'
            _, statsP = stats.f_oneway(*tData[i])
            print testUsed+' pValue:', statsP
            if statsP<pmin:
                multiCompP = getKWmultiComp(tData[i], datasetLabels, verbose)
            multiCompP = list(getOWANOVAmultiComp(tData[i], datasetLabels, verbose))
        statsData.append([label])
        statsData.append(['normalityTestStats']+normP)
        statsData.append([testUsed,statsP])
        if len(multiCompP)>0:
            statsData.append(['MultipleComparisons p-Value'])
            statsData.append(multiCompP)
        statsData.append([])
    return statsData


import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri #http://pandas.pydata.org/pandas-docs/version/0.19/r_interface.html

pandas2ri.activate()
nlme = importr('nlme')
statsR = importr('stats')
base = importr('base')
multcomp = importr('multcomp')
agr = importr('agricolae')


grpLbls = ['paramVal','genotype']
frmla = robjects.Formula('paramVal ~ genotype')

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




pMin = 0.05
import copy
gtypes = copy.deepcopy(genotypes)
gtypes.sort()
statsList = []
#for param in pltParamList:
#    colId = colIdPooledDict[param]
#    dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
#    statsVals = getStats([dSets], gtypes, param, [param], pMin, verbose=True)
#    statsList.append(statsVals)

for param in pltParamList:
    label = '====='+param+'====='
    print label
    colId = colIdPooledDict[param]
    dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
    
    statsVals = getStats([dSets], gtypes, param, [param], pMin, verbose=True)
    statsList.append(statsVals)
    
    df = createDF(dSets, grpLbls, gtypes)
    dfr1 = pandas2ri.py2ri(df)
    krsWall = statsR.kruskal_test(formula=frmla, data=dfr1)
    krsWallPd = pd.DataFrame(pandas2ri.ri2py(krsWall.rx2('p.value')))
    print ('Kruskal-Wallis p-value: %0.5f'%krsWallPd[0])
    model1 = robjects.r.lm(formula=frmla, data=dfr1)
    anv = robjects.r.anova(model1)
    #print anv
    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
    #print smry1
    multiComp1 = agr.HSD_test(model1, 'genotype', group=False, console=False)
    dfw1 = pd.DataFrame(pandas2ri.ri2py(multiComp1.rx2('comparison')))
    pVal= smry1['Pr(>F)']['genotype']
    if pVal<=pMin:
        print ('F value: %0.5f'%smry1['F value']['genotype'])
        print ('p-value: %0.5f'%pVal)
        print dfw1['pvalue']
    else:
        print('==> Not significantly different, p-value: %0.5f'%pVal)




#https://rcompanion.org/rcompanion/d_06.html




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
"""

















