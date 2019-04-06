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
import itertools

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri #http://pandas.pydata.org/pandas-docs/version/0.19/r_interface.html
import pandas as pd
pandas2ri.activate()
nparLD = importr('nparLD')
fsa = importr('FSA')


threshTrackTime         = 300   # in seconds, maximum duration of behaviour to be analysed
threshTrackLenMulti     = 3     # multipler of BLU for minimum trackLength w.r.t BLU
unitTimeDur             = 60    # unit time in seconds for pooling data for timeSeries analysis


nTmpts = threshTrackTime/unitTimeDur
headerRowId = bfTrkStats.headerRowId
inCsvHeader = bfTrkStats.csvHeader

colorFemales = (0,0,0)
colorMales = (1,0.1,1)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
             }

##============================================================================================================
def r_matrix_to_data_frame(r_matrix, getLabels = True):
    '''Convert an R matrix into a Pandas DataFrame'''
    array = pandas2ri.ri2py(r_matrix)
    if getLabels:
        return pd.DataFrame(array,
                            index=r_matrix.names[0],
                            columns=r_matrix.names[1])
    else:
        return pd.DataFrame(array)
def extendParamDict(paramValDict, data, gtype, timePoint):
    '''
    returns the paramValDict extended on the basis of the given inputs
    '''
    lenDat = len(data)
    paramValDict['result'].extend(data)
    paramValDict['timePoint'].extend([timePoint for x in xrange(lenDat)])
    paramValDict['flyNumber'].extend([x for x in xrange(lenDat)])
    paramValDict['genotype'].extend([gtype for x in xrange(lenDat)])
    return paramValDict

pltParamList = ['trackNum', 'trackDurMed', 
                'disTot', 'speed','trkDurTot',
                'straightness', 'gti',
                ]

baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
fig = 'fig5'
figList = ['fig2', 'fig3', 'fig4', 'fig5']
dfLbls = ['paramVal','genotype']
frmla = dfLbls[0]+'~'+dfLbls[1]
csvExt = ['*trackStats*.csv']
figDataFName = baseDir+'figDataFiles.txt'
figFoldersList = bfTrkStats.readFigFolderFile(figDataFName, figList)
csvDirs = bf.getDirList(baseDir)
currTime = bf.present_time()
print ("=============== Processing for all genotypes =============")
grpLbls = ['result','timePoint','flyNumber', 'genotype']
statsFormula = robjects.Formula('result~timePoint+genotype')
pMin = 0.05

for fig in figList:
    statsFName = baseDir+fig+'_stats_perMin_'+currTime+'.csv'
    figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))
    pooledTotalDataTmSrs = {}
    for genotype in figGenotypes:
        for i_,d in enumerate(csvDirs):
            if genotype == d.split(os.sep)[-1]:
                print ('---Processing for Genotype: %s'%genotype)
                figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
                pldTotalData = bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
                                                      threshTrackLenMulti, inCsvHeader, headerRowId,
                                                      colIdPooledDict, sexColors, pltParamList)
                pooledTotalDataTmSrs[genotype] = pldTotalData[3]
    
    genotypes = pooledTotalDataTmSrs.keys()
    gtTmptList = []
    tmPts = [str(i+1) for i in xrange(nTmpts)]
    for r in itertools.product(genotypes, tmPts):
        gtTmptList.append(r[0] + '_tmPt' + r[1])
    gtTmptMatrix = []
    gtTmptMatrix.extend(itertools.combinations(gtTmptList,2))

    for param in pltParamList:
        colId = colIdPooledDict[param]
        print "++++++++++++++++++++++++ tSeries ",colId, param, "++++++++++++++++++++++++"
        df1 = pd.DataFrame(columns=grpLbls)
        prmValDict = {x:[] for i,x in enumerate(grpLbls)}
        nFly=0
        for g_,gtype in enumerate(genotypes):
            dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for tmPt in xrange(nTmpts)]
            for i,d in enumerate(dSets):
                flyNum = list(np.arange(nFly,nFly+len(d)))
                tPts =  list(np.zeros(len(d))+i)
                gtypeList = [gtype for x in xrange(len(d))]
                dfData = {'result':d, 'timePoint': tPts,'flyNumber': flyNum, 'genotype': gtypeList}
                for j,l in enumerate(grpLbls):
                    prmValDict[l].extend(dfData[l])
            nFly+=len(d)
        df = pd.DataFrame(prmValDict, columns=grpLbls)
        descStats = pd.DataFrame(pandas2ri.ri2py(fsa.Summarize(statsFormula, data = pandas2ri.py2ri(df))))
        ll = nparLD.f1_ld_f1(pandas2ri.py2ri(df['result']),
                             pandas2ri.py2ri(df['timePoint']),\
                             pandas2ri.py2ri(df['genotype']),\
                             pandas2ri.py2ri(df['flyNumber']),
                             **{'description': 'FALSE',\
                                 'plot_RTE':'FALSE',
                                 'order.warning':'FALSE',
                                 })
        pdWald = r_matrix_to_data_frame(ll.rx2('Wald.test'), getLabels = True)
        pdAnova = r_matrix_to_data_frame(ll.rx2('ANOVA.test'), getLabels = True)
        pdPairComp = r_matrix_to_data_frame(ll.rx2('pair.comparison'), getLabels = False)
        #print ('Wald test\n%r'%pdWald)
        #print ('ANOVA test\n%r'%pdAnova)
        #print ('Pariwise Comparison\n%r'%pdPairComp)
        tmPts = [str(i) for i in xrange(nTmpts)]
        gtTmptMatrix = []
        gtTmptMatrix.extend(itertools.combinations(itertools.product(genotypes, tmPts),2))
        multiCompsList = []
        for gtM, multiComp in enumerate(gtTmptMatrix):
            prmValDict = {x:[] for i,x in enumerate(grpLbls)}
            dSet = []
            for mC in multiComp:
                gType = mC[0]
                tmPt = int(mC[1])
                dat = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
                dSet.append([dat, gType, tmPt])
            dataLen = [len(d[0]) for d in dSet]
            lenDiff =  max(dataLen)-min(dataLen)
            for d in dSet:
                dat, gType, tmPt = d
                prmValDict = extendParamDict(prmValDict, dat[:min(dataLen)], gType, str(tmPt)+gType)
            dat = pd.DataFrame(prmValDict, columns=grpLbls)
            ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
                               pandas2ri.py2ri(dat['timePoint']),\
                               pandas2ri.py2ri(dat['flyNumber']),\
                               **{'description': 'FALSE',\
                               'plot_RTE':'FALSE',
                               'order.warning':'FALSE',
                               })
            pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
            multiCompsList.append([gtM, lenDiff, dataLen, multiComp, pVal])
        with open(statsFName, 'a') as f:
            f.write('\nComparing Genotype or/and Time effect for:,%s\n\nDescriptive Stats:\n'%param)
        descStats.to_csv(statsFName, mode='a', header=True)
        with open(statsFName, 'a') as f:
            f.write('\nWald test statistics (WTS) Output\n')
        pdWald.to_csv(statsFName, mode='a', header=True)
        with open(statsFName, 'a') as f:
            f.write('\nANOVA test statistics (ATS) Output\n')
        pdAnova.to_csv(statsFName, mode='a', header=True)
        with open(statsFName, 'a') as f:
            f.write('\nPairwise Comparison\n')
        pdPairComp.to_csv(statsFName, mode='a', header=True)
        with open(statsFName, 'a') as f:
            f.write('\nPost-hoc analysis:')
            f.write('\n%s,%s,%s,%s,%s\n'%('group #', 'diff in Group "n\'s"',\
                    'Group1 n , Group2 n', 'Group1 genotype ,Group1 timePoint,\
                    Group2 genotype ,Group2 timePoint', 'p-value'))
            for mC in multiCompsList:
                gtM, lenDiff, dataLen, multiComp, pVal = mC
                f.write('%d,%d,%r,%r,%0.5f'%(gtM, lenDiff, dataLen, multiComp, pVal))
                f.write('\n')
            f.write('\n')
        with open(statsFName, 'a') as f:
            for x in xrange(10):
                f.write('-=-=-=-=-=-=-=-=-=-,')
            f.write('\n\n')

    #dfR = pandas2ri.py2ri(df)
    #formula1 = robjects.Formula('result ~ genotype * timePoint')
    #ll = nparLD.nparLD(formula1, data=dfR, subject = 'flyNumber', description = 'FALSE')
    #https://community.rstudio.com/t/arguments-imply-differing-number-of-rows/11479/3
    #Since there are 'NA' values, we will use the implicit function nparLD.f1_ld_f1
#    print ll.rx2('pair.comparison')
#    smry = robjects.r['summary'](ll)
#    print smry
#    df.to_csv(baseDir+'tempDF1.csv')








#for fig in figList:
#    statsFName = baseDir+fig+'_stats_perMin_'+currTime+'.csv'
#    figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))
#    pooledTotalDataTmSrs = {}
#    for genotype in figGenotypes:
#        for i_,d in enumerate(csvDirs):
#            if genotype == d.split(os.sep)[-1]:
#                print ('---Processing for Genotype: %s'%genotype)
#                figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
#                pldTotalData = bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
#                                                      threshTrackLenMulti, inCsvHeader, headerRowId,
#                                                      colIdPooledDict, sexColors, pltParamList)
#                pooledTotalDataTmSrs[genotype] = pldTotalData[3]
#    genotypes = pooledTotalDataTmSrs.keys()
#    tmPts = [str(i) for i in xrange(nTmpts)]
#    gtTmptMatrix = []
#    gtTmptMatrix.extend(itertools.combinations(itertools.product(genotypes, tmPts),2))
#    multiCompsList = []
#    for gtM, multiComp in enumerate(gtTmptMatrix):
#        prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#        dSet = []
#        for mC in multiComp:
#            gType = mC[0]
#            tmPt = int(mC[1])
#            dat = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
#            dSet.append([dat, gType, tmPt])
#        dataLen = [len(d[0]) for d in dSet]
#        lenDiff =  max(dataLen)-min(dataLen)
#        for d in dSet:
#            dat, gType, tmPt = d
#            prmValDict = extendParamDict(prmValDict, dat[:min(dataLen)], gType, str(tmPt)+gType)
#        dat = pd.DataFrame(prmValDict, columns=grpLbls)
#        ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
#                           pandas2ri.py2ri(dat['timePoint']),\
#                           pandas2ri.py2ri(dat['flyNumber']),\
#                           **{'description': 'FALSE',\
#                           'plot_RTE':'FALSE',
#                           'order.warning':'FALSE',
#                           })
#        pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
#        print gtM, lenDiff, dataLen, multiComp, pVal
#        multiCompsList.append([gtM, lenDiff, dataLen, multiComp, pVal])
#        with open(statsFName, 'a') as f:
#            f.write('%d,%d,%r,%r,%0.3f'%(gtM, lenDiff, dataLen, multiComp, pVal))
#            f.write('\n')
#
#
#
#

#prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#gType1 = gtTmptMatrix[0][0][0]
#tmPt1 = int(gtTmptMatrix[0][0][1])
#dat1 = [x[colId] for x in pooledTotalDataTmSrs[gType1][tmPt1]]
#prmValDict = extendParamDict(prmValDict, dat1, gType1, tmPt1)
#
#gType2 = gtTmptMatrix[0][1][0]
#tmPt2 = int(gtTmptMatrix[0][1][1])
#dat2 = [x[colId] for x in pooledTotalDataTmSrs[gType2][tmPt2]]
#prmValDict = extendParamDict(prmValDict, dat2, gType2+'_'+str(tmPt2), tmPt2)
#dat = pd.DataFrame(prmValDict, columns=grpLbls)
#ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
#                   pandas2ri.py2ri(dat['timePoint']),\
#                   pandas2ri.py2ri(dat['flyNumber']),\
#                   **{'description': 'FALSE',\
#                   'plot_RTE':'FALSE',
#                   'order.warning':'TRUE',
#                   })
#print ll1.rx2('ANOVA.test').rx2('p-value')[0]
#print ll1.names
#
#from scipy import stats
#statsR = importr('stats')
#
#frmla = robjects.Formula('result ~ timePoint')
#
#tmPts = [str(i) for i in xrange(nTmpts)]
#gtTmptMatrix = []
#gtTmptMatrix.extend(itertools.combinations(itertools.product(genotypes, tmPts),2))
#multiCompsList = []
#for multiComp in gtTmptMatrix:
#    print multiComp
#    prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#    dSet = []
#    for combinations in multiComp:
#        gType = combinations[0]
#        tmPt = int(combinations[1])
#        dat = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
#        dSet.append(dat)
#        prmValDict = extendParamDict(prmValDict, dat, gType, tmPt)
#    dat = pd.DataFrame(prmValDict, columns=grpLbls)
#    #print dat
#    ll1 = nparLD.f1_ld_f1(pandas2ri.py2ri(dat['result']),
#                         pandas2ri.py2ri(dat['timePoint']),\
#                         pandas2ri.py2ri(dat['genotype']),\
#                         pandas2ri.py2ri(dat['flyNumber']),
#                         **{'description': 'FALSE',\
#                             'plot_RTE':'FALSE',
#                             'order.warning':'FALSE',
#                             })
#    #    ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
#    #                       pandas2ri.py2ri(dat['timePoint']),\
#    #                       pandas2ri.py2ri(dat['flyNumber']),\
#    #                       **{'description': 'FALSE',\
#    #                       'plot_RTE':'FALSE',
#    #                       'order.warning':'TRUE',
#    #                       })
#    pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
#    print multiComp, pVal
#    
#    #print ll1.names
#    multiCompsList.append([multiComp, pVal])
#    model1 = robjects.r.lm(formula=frmla, data=pandas2ri.py2ri(dat))
#    anv = robjects.r.anova(model1)
#    print anv.rx2("Pr(>F)")
#    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
#    print stats.mannwhitneyu(dSet[0], dSet[1]).pvalue
#    print statsR.t_test(pandas2ri.py2ri(pd.DataFrame(dSet[0])), pandas2ri.py2ri(pd.DataFrame(dSet[1]))).rx2('p.value')[0]
#
#
#
#
#model1 = robjects.r.lm(formula=frmla, data=dfr1)
#anv = robjects.r.anova(model1)
#print anv
#smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
#
#
#aa = df['genotype']=='W1118'
#aa
#ab = df['timePoint']=='0.0'
#
#tmPts = [str(i) for i in xrange(nTmpts)]
#multiCompsList = []
#for multiComp in gtTmptMatrix:
#    print multiComp
#    dat = pd.DataFrame(columns=grpLbls)
#    for mC in multiComp:
#        dfGt = df['genotype']==mC[0]
#        dfTp = df['timePoint']==int(mC[1])
#        dat = dat.append(df[dfGt*dfTp])
#    #    ll1 = nparLD.f1_ld_f1(pandas2ri.py2ri(dat['result']),
#    #                         pandas2ri.py2ri(dat['timePoint']),\
#    #                         pandas2ri.py2ri(dat['genotype']),\
#    #                         pandas2ri.py2ri(dat['flyNumber']),
#    #                         **{'description': 'FALSE',\
#    #                             'plot_RTE':'FALSE',
#    #                             'order.warning':'FALSE',
#    #                             })
#    ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
#                       pandas2ri.py2ri(dat['timePoint']),\
#                       pandas2ri.py2ri(dat['flyNumber']),\
#                       **{'description': 'FALSE',\
#                       'plot_RTE':'FALSE',
#                       'order.warning':'TRUE',
#                       })
#    pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
#    print multiComp, pVal
#    
#    #print ll1.names
#    multiCompsList.append([multiComp, pVal])
#    model1 = robjects.r.lm(formula=frmla, data=pandas2ri.py2ri(dat))
#    anv = robjects.r.anova(model1)
#    print anv.rx2("Pr(>F)")
#    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
#    print stats.mannwhitneyu(dSet[0], dSet[1]).pvalue
#    print statsR.t_test(pandas2ri.py2ri(pd.DataFrame(dSet[0])), pandas2ri.py2ri(pd.DataFrame(dSet[1]))).rx2('p.value')[0]
#


"""
tmPts = [str(i) for i in xrange(nTmpts)]
gtTmptMatrix = []
gtTmptMatrix.extend(itertools.combinations(itertools.product(genotypes, tmPts),2))
multiCompsList = []
for gtM, multiComp in enumerate(gtTmptMatrix):
    prmValDict = {x:[] for i,x in enumerate(grpLbls)}
    dSet = []
    for mC in multiComp:
        gType = mC[0]
        tmPt = int(mC[1])
        dat = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
        dSet.append([dat, gType, tmPt])
    dataLen = [len(d[0]) for d in dSet]
    lenDiff =  max(dataLen)-min(dataLen)
    for d in dSet:
        dat, gType, tmPt = d
        prmValDict = extendParamDict(prmValDict, dat[:min(dataLen)], gType, str(tmPt)+gType)
    dat = pd.DataFrame(prmValDict, columns=grpLbls)
    ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
                       pandas2ri.py2ri(dat['timePoint']),\
                       pandas2ri.py2ri(dat['flyNumber']),\
                       **{'description': 'FALSE',\
                       'plot_RTE':'FALSE',
                       'order.warning':'FALSE',
                       })
    pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
    print gtM, lenDiff, dataLen, multiComp, pVal
    multiCompsList.append([gtM, lenDiff, dataLen, multiComp, pVal])











tmPts = [str(i) for i in xrange(nTmpts)]
gtTmptMatrix = []
gtTmptMatrix.extend(itertools.combinations(itertools.product(genotypes, tmPts),2))
multiCompsList = []
for multiComp in gtTmptMatrix:
    print multiComp
    prmValDict = {x:[] for i,x in enumerate(grpLbls)}
    dSet = []
    for combinations in multiComp:
        gType = combinations[0]
        tmPt = int(combinations[1])
        dat = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
        dSet.append(dat)
        prmValDict = extendParamDict(prmValDict, dat, gType, tmPt)
    dat = pd.DataFrame(prmValDict, columns=grpLbls)
    print dat
    ll1 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
                       pandas2ri.py2ri(dat['timePoint']),\
                       pandas2ri.py2ri(dat['flyNumber']),\
                       **{'description': 'FALSE',\
                       'plot_RTE':'FALSE',
                       'order.warning':'TRUE',
                       })
    pVal = ll1.rx2('ANOVA.test').rx2('p-value')[0]
    print multiComp, pVal
    
    #print ll1.names
    multiCompsList.append([multiComp, pVal])
    model1 = robjects.r.lm(formula=frmla, data=pandas2ri.py2ri(dat))
    anv = robjects.r.anova(model1)
    print anv.rx2("Pr(>F)")
    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
    print stats.mannwhitneyu(dSet[0], dSet[1]).pvalue
    print statsR.t_test(pandas2ri.py2ri(pd.DataFrame(dSet[0])), pandas2ri.py2ri(pd.DataFrame(dSet[1]))).rx2('p.value')[0]

"""