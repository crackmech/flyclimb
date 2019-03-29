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
    totData, totUntTmData, pldTotalData, pldTotUntData, pltDataTotal, pltDataUnitTime = \
                            bfTrkStats.pooledData(d, csvExt, unitTimeDur, threshTrackTime,
                                          threshTrackLenMulti, inCsvHeader, headerRowId,
                                          colIdPooledDict, sexColors, pltParamList)
    totalData[genotype] = totData
    totalDataTmSrs[genotype] = totUntTmData
    pooledTotalData[genotype] = pldTotalData
    pooledTotalDataTmSrs[genotype] = pldTotUntData
    pltTotalData[genotype] = pltDataTotal
    pltTmSrsData[genotype] = pltDataUnitTime

genotypes = totalData.keys()
genotypes = ['CS', 'W1118', 'PINK1RV', 'Park25xW1118', 'W1118xLrrk-ex1', 'Park25xLrrk-ex1', 'Trp-Gamma']


##============================================================================================================
#
##------test plots
##---- Plot the total data from behaviour from total time measured ----#
#genotypes = totalData.keys()
#genotypes = ['CS', 'W1118', 'PINK1RV', 'Park25xW1118', 'W1118xLrrk-ex1', 'Park25xLrrk-ex1', 'Trp-Gamma']
#gtype = genotypes[0]
#
#colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
#
#sWidth = 0.5
#vPlots = []
#for i in xrange(len(pltTotalData[gtype])):
#    fig, ax = plt.subplots()
#    for g, gtype in enumerate(genotypes):
#        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
#        scPlt1 = bf.plotScatterCentrd(ax,pltTotalData[gtype][i], g, \
#                                      scatterRadius=10, scatterColor=colorSex, \
#                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
#                                      scatterWidth = sWidth)
#        vp = plt.violinplot(pltTotalData[gtype][i], [g], showextrema=False)
#        vPlots.append(vp)
#    plt.xlim(-1,len(genotypes))
#    plt.title(str(i)+'_'+pltParamList[i])
#plt.show()
#
##------test plots
##---- Plot the timeSEries data from behaviour from total time measured ----#
#for i in xrange(len(pltTmSrsData[gtype])):
#    fig, ax = plt.subplots()
#    for g, gtype in enumerate(genotypes):
#        pltData, pltDataErr = pltTmSrsData[gtype][i]
#        ax.errorbar(np.arange(len(pltData)), pltData, pltDataErr, alpha=0.7)
#        plt.title(str(i)+'_'+pltParamList[i])
#plt.show()
#
##============================================================================================================


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
    c = datasetLabels[0]
    e1 = datasetLabels[1]
    e2 = datasetLabels[2]
    statsData = []
    statsData.append(['Test and Parameter', 'p-Value', 'p-Value', 'p-Value'])
    statsData.append(['',c+' vs. '+e1, c+' vs. '+e2, e1+' vs. '+e2])
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



pMin = 0.05
import copy
gtypes = copy.deepcopy(genotypes)
gtypes.sort()
statsList = []
for param in pltParamList:
    colId = colIdPooledDict[param]
    dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
    statsVals = getStats([dSets], gtypes, param, [param], pMin, verbose=True)
    statsList.append(statsVals)

###---------Compare SCIPY ANOVA with R- ANOVA---------
colId = colIdPooledDict['fps']

colId = colIdPooledDict['disTot']
#dSets1 = {gtype:[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)}
dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
_, statsP = stats.f_oneway(*dSets)
multiCompP = getKWmultiComp(dSets, gtypes, verbose=True)
print statsP
#dSets = [[x[colId] for i_,x in enumerate(pooledTotalData[gtype])] for g_,gtype in enumerate(gtypes)]
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
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

##---- Create a Pandas DataFrame for all genotypes at a particular timepoint in the tSeries data ---
#grpLbls = ['paramVal','genotype']#['param','timePt','fly']
#prmValDict = {'paramVal':[], 'genotype':[]}
#for i,x in enumerate(dSets):
#    flyLabls = [gtypes[i] for y in xrange(len(x))]
#    dfData = [x, flyLabls]
#    for j,y in enumerate(grpLbls):
#        prmValDict[y].extend(dfData[j])
#df = pd.DataFrame(prmValDict, columns=grpLbls)
##---- Created a Pandas DataFrame forfor all genotypes at a particular timepoint in the tSeries data ---

df = createDF(dSets, grpLbls, gtypes)
dfr1 = pandas2ri.py2ri(df)
model1 = robjects.r.lm(formula=frmla, data=dfr1)
anv = robjects.r.anova(model1)
print anv
smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
print smry1
multiComp1 = agr.HSD_test(model1, 'genotype', group=False, console=False)
dfw1 = pd.DataFrame(pandas2ri.ri2py(multiComp1.rx2('comparison')))
print dfw1['pvalue']



#print multiComp1
#print multiComp1.names
#print multiComp1.rx2(multiComp1.names[0])
#print multiComp1.rx2(multiComp1.names[1])
#print multiComp1.rx2(multiComp1.names[2])
#print multiComp1.rx2(multiComp1.names[3])




#model = statsR.aov(formula=frmla, data=dfr1)
#multiComp = statsR.TukeyHSD(model)
#dfw = r_matrix_to_data_frame(multiComp.rx2('genotype'))
#print dfw['p adj']
#smry= robjects.r['summary'](model)
#print smry
#print(base.summary(model))
#
#
#model1 = robjects.r.lm(formula=frmla, data=dfr1)
#anv = robjects.r.anova(model1)
#aa = pd.DataFrame(pandas2ri.ri2py(anv))
#aa.axes[1]
#
#
#print(base.summary(model1).rx2('coefficients'))
#dfS = r_matrix_to_data_frame(base.summary(model1).rx2('coefficients'))
#
#print model1.names
#
#
#

###---------Compare SCIPY ANOVA with R- ANOVA---------



#    rmaData = restrucDataForRMA(dSets)
#    res = getRMAnova(rmaData, gtypes, verbose=False)
#    print res.anova_table

#
#df = pd.DataFrame(columns=['groups','values'])
#for i,g in enumerate(gtypes):
#    for j,d in enumerate(dSets[i]):
#        df = df.append(pd.DataFrame([[g,d]], columns=['groups','values']), ignore_index=True)
#
#pVals = sp.posthoc_dunn(df, val_col = 'values', group_col = 'groups', p_adjust='bonferroni')
#print pVals

'''
for Stats:
    For total Behaviour data:
        Check Normality:
            if normal distribution: 
                OneWayANOVA, postHoc pairwaiseTukeysHSD
            else:
                KruskalWallis, postHoc dunn with p_adjust='bonferroni'
    
    For timeSeries Behaviour data:
        1)  Do a repeated-measures ANOVA (or Friedman's test) for a genotype,
                to compare between the behaviour of the same genotype at different timepoint 
                PostHoc:    Tukey's HSD for rmANOVA or
                            Conover, Nemenyi, Siegel, and Miller tests for Friedman test
        2)  For comparing multiple genotypes for a single timepoint:
                if normal distribution: 
                        OneWayANOVA, postHoc pairwaiseTukeysHSD
                    else:
                        KruskalWallis, postHoc dunn with p_adjust='bonferroni'

For no data at a timepoint, use zero track number instead of ignoring that timepoint

Save all the data for total Behavior or each timepoint in an excel sheet with :
    a)  All data of each genotype
    b)  Means, median, SD, SEM, "n" size for each genotype
    c)  Normality value for each genotype
    d)  Stats applied for each behavioural parameter
Create a separate worksheet for each parameter in a sheet.

'''

def getDescStats(array):
    '''
    generate descriptive statistics for the input 1-D array
    
    returns:
        "n" Size, Mean, Median, SD, SEM and Normality, skewness, kurtoisis
    '''
    nSize = len(array)
    mean = np.mean(array)
    median = np.median(array)
    sd = np.std(array)
    sem = sd/np.sqrt(nSize)
    if nSize>=8: #minimum sample size required for calculating below mentioned stats
        _, normP = stats.normaltest(array)
        skewness = stats.skewtest(array)
        kurtoisis = stats.kurtosis(array)
    else:
        normP, skewness, kurtoisis = np.nan
    return nSize, mean, median, sd, sem, normP, skewness, kurtoisis

formula1 = robjects.Formula('result~timePoint*genotype')
formula2 = robjects.Formula('~1|flyNumber')

grpLbls = ['result','timePoint','flyNumber']
grpLbls_gtypes = ['result','timePoint','flyNumber','genotype']
for param in pltParamList:
    colId = colIdPooledDict[param]
    df1 = pd.DataFrame(columns=grpLbls_gtypes)
    for g_,gtype in enumerate(gtypes):
        dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for tmPt in xrange(threshTrackTime/unitTimeDur)]
        prmValDict = {x:[] for i,x in enumerate(grpLbls)}
        for i,x in enumerate(dSets):
            flyLabls = list(np.arange(len(x)))
            tPts =  list(np.zeros(len(x))+i)
            dfData = [x, tPts, flyLabls]
            for j,y in enumerate(grpLbls):
                prmValDict[y].extend(dfData[j])
        df = pd.DataFrame(prmValDict, columns=grpLbls)
        df['genotype'] = gtype
        df1 = df1.append(df, ignore_index=True)
    dfr1 = pandas2ri.py2ri(df1)
    model = nlme.lme(fixed=formula1, data=dfr1, random=formula2, **{'na.action':statsR.na_omit})
    smry= robjects.r['summary'](model)
    print smry.rx2("tTable")
    print smry
    print robjects.r.anova(model)


#    df1.to_csv('/media/aman/data/tmpData_Srivatsan/'+param+'.csv', header = grpLbls_gtypes)



'''comparing single genotype across all timepoints'''
#formula1 = robjects.Formula('param~timePt')
formula1 = robjects.Formula('param~timePt*genotype')
formula2 = robjects.Formula('~1|fly')

#formula1 = robjects.Formula('param~fly')
#formula2 = robjects.Formula('~1|timePt')

grpLbls = ['param','timePt','fly']
statsList = []
for g_,gtype in enumerate(gtypes):
    for param in pltParamList[:5]:
        colId = colIdPooledDict[param]
        print "++++++++++++++++++++++++ tSeries ",colId, param, "++++++++++++++++++++++++"
        dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for tmPt in xrange(threshTrackTime/unitTimeDur)]
        rmaData = restrucDataForRMA(dSets)
        res = getRMAnova(rmaData, gtypes, verbose=False)
        #print res.anova_table
        statsList.append(res.anova_table)
        #---- Create a Pandas DataFrame for a parameter in UnitTimeData ---
        prmValDict = {x:[] for i,x in enumerate(grpLbls)}
        for i,x in enumerate(rmaData):
            flyLabls = list(np.arange(len(x)))
            tPts =  list(np.zeros(len(x))+i)
            dfData = [x, tPts, flyLabls]
            for j,y in enumerate(grpLbls):
                prmValDict[y].extend(dfData[j])
        df = pd.DataFrame(prmValDict, columns=grpLbls)
        #---- Created a Pandas DataFrame for a parameter in UnitTimeData ---
        dfr1 = pandas2ri.py2ri(df)
        model = nlme.lme(fixed=formula1, data=dfr1, random=formula2, **{'na.action':statsR.na_omit})
        smry= robjects.r['summary'](model)
        print smry.rx2("tTable")
        print smry
        print robjects.r.anova(model)

#---- Create a Pandas DataFrame for a parameter in UnitTimeData ---
prmValDict = {x:[] for i,x in enumerate(grpLbls)}
for i,x in enumerate(rmaData):
    flyLabls = list(np.arange(len(x)))
    tPts =  list(np.zeros(len(x))+i)
    dfData = [x, tPts, flyLabls]
    for j,y in enumerate(grpLbls):
        prmValDict[y].extend(dfData[j])
df = pd.DataFrame(prmValDict, columns=grpLbls)
#---- Created a Pandas DataFrame for a parameter in UnitTimeData ---
dfr1 = pandas2ri.py2ri(df)
model = nlme.lme(fixed=formula1, data=dfr1, random=formula2, **{'na.action':statsR.na_omit})
smry= robjects.r['summary'](model)
print smry.rx2("tTable")



getOmitsR = smry.rx2('na.action')
groups = pandas2ri.ri2py(smry.rx2('groups'))
omitsP = pandas2ri.ri2py(getOmitsR)

print groups
print omitsP
print len(omitsP), type(omitsP)
'''comparing single genotype across all timepoints'''




'''comparing multiple genotypes at a single timepoint'''
grpLbls = ['paramVal','genotype']
statsList = []
for tmPt in xrange(threshTrackTime/unitTimeDur):
    for param in pltParamList[:5]:
        colId = colIdPooledDict[param]
        print "++++++++++++++++++++++++ Total ",colId, param, "++++++++++++++++++++++++"
        dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for g_,gtype in enumerate(gtypes)]
        statsVals = getStats([dSets], gtypes, param, [param], pMin, verbose=False)
        statsList.append(statsVals)


for tmPt in xrange(threshTrackTime/unitTimeDur):
    for param in pltParamList[:5]:
        colId = colIdPooledDict[param]
        print "++++++++++++++++++++++++ tSeries",colId, param, '@tmPt', tmPt, "++++++++++++++++++++++++"
        dSets = [[x[colId] for i_, x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for g_, gtype in enumerate(gtypes)]
        dfR = pandas2ri.py2ri(createDF(dSets, grpLbls, gtypes))
        model1 = robjects.r.lm(formula=frmla, data=dfR, **{'na.action': statsR.na_exclude})
        anv = robjects.r.anova(model1)
        smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
        #print smry1
        multiComp1 = agr.HSD_test(model1, 'genotype', group=False, console=False)
        dfw1 = pd.DataFrame(pandas2ri.ri2py(multiComp1.rx2('comparison')))
        print dfw1['pvalue']


model = statsR.aov(formula=frmla, data=dfR, **{'na.action':statsR.na_omit})
multiComp = statsR.TukeyHSD(model)
print pandas2ri.ri2py(multiComp.rx2('genotype'))#('p adj')
smry= robjects.r['summary'](model)
print smry

dfw = r_matrix_to_data_frame(multiComp.rx2('genotype'))
print dfw['p adj']

'''comparing multiple genotypes at a single timepoint'''








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

















