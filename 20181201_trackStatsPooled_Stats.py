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
#from matplotlib import pyplot as plt

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

#
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
#

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

'''comparing multiple genotypes at a single timepoint'''
statsList = []
for tmPt in xrange(threshTrackTime/unitTimeDur):
    for param in pltParamList[:5]:
        colId = colIdPooledDict[param]
        print "++++++++++++++++++++++++ Total ",colId, param, "++++++++++++++++++++++++"
        dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for g_,gtype in enumerate(gtypes)]
        statsVals = getStats([dSets], gtypes, param, [param], pMin, verbose=False)
        statsList.append(statsVals)
'''comparing multiple genotypes at a single timepoint'''

'''comparing single genotype across all timepoints'''
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
'''comparing single genotype across all timepoints'''



#---- Create a Pandas DataFrame for a parameter in UnitTimeData ---
grpLbls = ['param','timePt','fly']
prmValDict = {x:[] for i,x in enumerate(grpLbls)}
for i,x in enumerate(rmaData):
    flyLabls = list(np.arange(len(x)))
    tPts =  list(np.zeros(len(x))+i)
    dfData = [x, tPts, flyLabls]
    for j,y in enumerate(grpLbls):
        prmValDict[y].extend(dfData[j])
df = pd.DataFrame(prmValDict, columns=grpLbls)
#---- Created a Pandas DataFrame for a parameter in UnitTimeData ---


from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri #http://pandas.pydata.org/pandas-docs/version/0.19/r_interface.html
pandas2ri.activate()
nlme = importr('nlme')
statsR = importr('stats')

formula1 = robjects.Formula('param~timePt')
formula2 = robjects.Formula('~1|fly')

dfr1 = pandas2ri.py2ri(df)
model = nlme.lme(fixed=formula1, data=dfr1, random=formula2, **{'na.action':statsR.na_omit})

#summary = robjects.r['summary']
#smry= summary(model)
smry= robjects.r['summary'](model)
getOmitsR = smry.rx2('na.action')
groups = pandas2ri.ri2py(smry.rx2('groups'))
omitsP = pandas2ri.ri2py(getOmitsR)

print groups
print omitsP
print len(omitsP), type(omitsP)












#---- Create a Pandas DataFrame for all genotypes at a particular timepoint in the tSeries data ---
grpLbls = ['paramVal','genotype']#['param','timePt','fly']
prmValDict = {'paramVal':[], 'genotype':[]}
for i,x in enumerate(dSets):
    flyLabls = [gtypes[i] for y in xrange(len(x))]
    dfData = [x, flyLabls]
    for j,y in enumerate(grpLbls):
        prmValDict[y].extend(dfData[j])
df = pd.DataFrame(prmValDict, columns=grpLbls)
#---- Created a Pandas DataFrame forfor all genotypes at a particular timepoint in the tSeries data ---

frmla = robjects.Formula('paramVal ~ genotype')
dfr1 = pandas2ri.py2ri(df)
model = statsR.aov(formula=frmla, data=dfr1)
multiComp = statsR.TukeyHSD(model)
print pandas2ri.ri2py(multiComp.rx2('genotype'))#('p adj')
smry= robjects.r['summary'](model)
print smry



















#for i, v in enumerate(list(smry.names)):
#    print v
#    if v not in ['call', 'residuals', 'data', 'fitted']:
#        print '%s====================='%i, v
#        print smry.rx2(v)

#pandas2ri.deactivate()

#
##---- Create a Pandas DataFrame for a parameter in UnitTimeData ---
#grpLbls = ['param','timePt','fly']
#prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#for i,x in enumerate(rmaData):
#    flyLabls = list(np.arange(len(x)))
#    tPts =  list(np.zeros(len(x))+i)
#    dfData = [x, tPts, flyLabls]
#    for j,y in enumerate(grpLbls):
#        prmValDict[y].extend(dfData[j])
#df = pd.DataFrame(prmValDict, columns=grpLbls)

##---- Create a Pandas DataFrame for a parameter in UnitTimeData ---
#groupLabels = ['param','timePt','fly']
#df = pd.DataFrame(columns=groupLabels)
#for i,x in enumerate(rmaData):
#    flyLabls = list(np.arange(len(x)))
#    tPts =  list(np.zeros(len(x))+i)
#    dfData = [x, tPts, flyLabls]
#    dfi =  pd.concat([pd.Series(dfData[0], name=groupLabels[0]),\
#                      pd.Series(dfData[1], name=groupLabels[1]),\
#                      pd.Series(dfData[2], name=groupLabels[2])], axis=1)
#    df = pd.concat([df, dfi], ignore_index=True) 
#df.to_csv('/media/aman/data/tmpdf.csv')
##---- Created a Pandas DataFrame for a parameter in UnitTimeData ---
##pandas2ri.activate()
##dfr = conversion.py2ri(df)  # convert from pandas to R and make string columns factors
#formula1 = robjects.Formula('fixed= param~timePt')
#formula2 = robjects.Formula('~1|fly')
#dfr = robjects.r('''mydata <- read.csv('/media/aman/data/tmpdf.csv',sep=',')''')
#type(dfr)
##print dfr
#
#model = nlme.lme(fixed=formula1, data=df, \
#                random=formula2, **{'na.action':statsR.na_omit})
#test2_sum= robjects.r.summary(model)
##for i, v in enumerate(list(test2_sum.names)):
##    print v
##    if v not in ['call', 'residuals']:
##        print '%s========================================================='%i, v
##        print test2_sum.rx2(v)
#
#print test2_sum.rx2('na.action')
#print test2_sum
#getOmitsR = test2_sum.rx2('na.action')
#omitsP = conversion.ri2py(getOmitsR)




#dfr = robjects.r('''mydata2 <- read.csv('/media/aman/data/tmpdf.csv',sep=',')''')
#aov2 = ro.r('''aov2 <- lme(fixed= param~timePt, random=~1|fly, na.action=na.omit, data=mydata2)''')
#aov2Sum = ro.r('summary(aov2)')
#print aov2Sum.names
#print aov2Sum.rx2('tTable')



##---https://stackoverflow.com/questions/22693059/migrating-a-logistic-regression-from-r-to-rpy2
#import rpy2.robjects as ro
#mydata = ro.r['data.frame']
#read = ro.r['read.csv']
#head = ro.r['head']
#summary = ro.r['summary']
#
#mydata = read("http://www.ats.ucla.edu/stat/data/binary.csv")
##cabecalho = head(mydata)
#formula = 'admit ~ gre + gpa + rank'
#mylogit = ro.r.glm(formula=ro.r(formula), data=mydata,family=ro.r('binomial(link="logit")'))
##What NEXT?
#





'''
https://www.r-bloggers.com/linear-models-anova-glms-and-mixed-effects-models-in-r/
https://stats.stackexchange.com/questions/65656/how-to-account-for-repeated-measures-in-glmer
https://stat.ethz.ch/pipermail/r-help/2010-February/228627.html




#---- To install R package from python---
#from rpy2.robjects.packages import importr
#utils = importr('utils')
#utils.install_packages('DirichletReg')

#https://stackoverflow.com/questions/19928662/call-anova-on-lme4-lmer-output-via-rpy
#http://nbviewer.ipython.org/urls/gist.github.com/TheChymera/7396334/raw/20c7878456fea4c05e1889d566ab1e5b20bc90bb/nlme-aov
#http://nbviewer.jupyter.org/urls/gist.github.com/TheChymera/7669971/raw/e35bc10d5b6443d28eca44bce8cf17eaa7bf1a8a/TC_lme4
#---https://www.statsmodels.org/dev/mixed_linear.html---
#http://nbviewer.jupyter.org/urls/umich.box.com/shared/static/6tfc1e0q6jincsv5pgfa.ipynb



import rpy2.robjects.conversion as conversion
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#https://github.com/pandas-dev/pandas/issues/10062
import pandas.rpy.common as com
pydf = com.load_data('iris')
r_matrix = com.convert_to_r_matrix(df)

from rpy2.robjects import r
r.data(name)
robj = r[name]
conversion.ri2py(robj)

#https://stackoverflow.com/questions/35395476/extract-coefficients-from-r-lme-model-in-rpy2
The method rx2 corresponds to R's [[, which I understand to be identical to $.


https://stackoverflow.com/questions/24216036/get-a-clean-summary-of-nlme-lme-or-lme4-lmer-in-rpy


#https://stackoverflow.com/questions/9505952/omit-na-values-from-prcomp-in-rpy2
**{'na.action' : stats.na_omit}


#http://www.bodowinter.com/tutorial/bw_LME_tutorial.pdf
#https://datascienceplus.com/linear-mixed-model-workflow/

'''




#
#df = pd.DataFrame(columns=['groups','values'])
#for i,g in enumerate(gtypes):
#    for j,d in enumerate(dSets[i]):
#        df = df.append(pd.DataFrame([[g,d]], columns=['groups','values']), ignore_index=True)
#
#pVals = sp.posthoc_dunn(df, val_col = 'values', group_col = 'groups', p_adjust='bonferroni')
#print pVals

'''
http://r.789695.n4.nabble.com/Repeated-Measures-ANOVA-and-Missing-Values-in-the-data-set-td4708855.html

require(nlme)
subject <- c(1,2,3,4,5,6,7,8,9,10)
time1 <- c(5040,3637,6384,5309,5420,3549,NA,5140,3890,3910)
time2 <- c(5067, 3668, NA, 6489, NA, 3922, 3408, 6613, 4063, 3937)
time3 <- c( 3278, 3814, 8745, 4760, 4911, 5716, 5547, 5844, 4914, 4390)
time4 <- c(   0, 2971,    0, 2776, 2128, 1208, 2935, 2739, 3054, 3363)
time5 <- c(4161, 3483, 6728, 5008, 5562, 4380, 4006, 7536, 3805, 3923)
time6 <- c( 3604, 3411, 2523, 3264, 3578, 2941, 2939,   NA, 3612, 3604)
mydata <- data.frame(time1, time2, time3, time4, time5, time6)
mydata2 = stack(mydata)
subject  = factor(rep(subject,6))
mydata2[3] = subject
colnames(mydata2) = c("values", "time", "subject")

require(nlme)
aov2 <- lme(fixed= values~time, random=~1|subject, na.action=na.omit, data=mydata2)
summary(aov2)
anova(aov2)

'''


#gt1 = genotypes[0]
#gt2 = genotypes[1]
#gt3 = genotypes[4]
#gt4 = genotypes[5]
#dataSetLbls  = [gt1, gt2, gt3, gt4]
#
#paramLbl = ['totalDistanceTravelled']
#param = 'disTot'
#colId = colIdPooledDict[param]
#dCS = [x[colId] for i_,x in enumerate(pooledTotalData[gt1])]
#dEx1 = [x[colId] for i_,x in enumerate(pooledTotalData[gt2])]
#dEx2 = [x[colId] for i_,x in enumerate(pooledTotalData[gt3])]
#dEx3 = [x[colId] for i_,x in enumerate(pooledTotalData[gt4])]
#dSets = [dCS,dEx1,dEx2]
#
#aa = getStats([dSets], dataSetLbls, param, [param], pNormMin, verbose=False)
#
#rmaData = restrucDataForRMA(dSets)
#res = getRMAnova(rmaData, dataSetLbls, verbose=False)
#print (res.summary())
#print res.anova_table
#
#statsList = []
#for param in pltParamList:
#    colId = colIdPooledDict[param]
#    dCS = [x[colId] for i_,x in enumerate(pooledTotalData[gt1])]
#    dEx1 = [x[colId] for i_,x in enumerate(pooledTotalData[gt2])]
#    dEx2 = [x[colId] for i_,x in enumerate(pooledTotalData[gt3])]
#    dEx3 = [x[colId] for i_,x in enumerate(pooledTotalData[gt4])]
#    dSets = [dCS,dEx1,dEx2,dEx3]
#    statsVals = getStats([dSets], dataSetLbls, param, [param], pNormMin, verbose=True)
#    statsList.append(statsVals)
#
#    rmaData = restrucDataForRMA(dSets)
#    res = getRMAnova(rmaData, dataSetLbls, verbose=False)
#    print (res.summary())
#    print res.anova_table






































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


