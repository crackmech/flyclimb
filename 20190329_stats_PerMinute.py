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


within group: single group, across timepoints: nparcomp.mctp_rm (VALID only for One group)
nparcomp.mctp_rm(resp~time, data=datanew, asy.method = "fisher",
     type = "Tukey", alternative = "two.sided", plot.simci = FALSE,
     info = FALSE))

within group: single group, across timepoints: 
nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
            pandas2ri.py2ri(dat['genotype']),\
            pandas2ri.py2ri(dat['flyNumber']),\
            **{'description': 'FALSE',\
            'plot_RTE':'FALSE',
            'order.warning':'TRUE',
            })


https://rpsychologist.com/r-guide-longitudinal-lme-lmer

#http://r.789695.n4.nabble.com/Repeated-Measures-ANOVA-and-Missing-Values-in-the-data-set-td4708855.html
http://dwoll.de/rexrepos/posts/anovaMixed.html
library(nlme)
lmeFit <- lme(Y ~ Xw1, random=~1 | id, method="ML", data=d1))
library(multcomp)
contr <- glht(lmeFit, linfct=mcp(Xw1="Tukey"))
summary(contr)

dataP=subset(data, genotype=='Park25xLrrk-ex1')
dataP$timePoint <- as.factor(dataP$timePoint)
statModel = lmer(result ~ timePoint + (1 | flyNumber), data=dataP)
contr <- glht(statModel, linfct=mcp(timePoint="Tukey"))
summary(contr)

between groups, single timepoint: nparcomp.mctp(weight ~dosage, data=liver, asy.method = "fisher",
        type = "Tukey", alternative = "two.sided", plot.simci = FALSE,
        info = FALSE)
between groups, single timepoint: 
(f1.ld.f1(y=datanew$resp, time=datanew$time, group=datanew$group, subject=datanew$subject, description=FALSE))$pair.comparison
summary(mctp(resp~group, data=datanew, asy.method = "fisher",
     type = "Tukey", alternative = "two.sided", plot.simci = FALSE,
     info = FALSE))
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
nparcomp = importr('nparcomp')

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
    tmPts = [i for i in xrange(nTmpts)]
    tmptMatrix = [x for x in itertools.product([tmPts[0]],tmPts)][1:]#[x for x in itertools.product([0],tmPts)]
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
        multiComps = {'between': [],
                      'within': []\
                      }
        
        for tmPt in xrange(nTmpts):
            prmValDict = {x:[] for i,x in enumerate(grpLbls)}
            nFly=0
            for gType in genotypes:
                data1 = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
                nFlies = len(data1)
                prmValDict['result'].extend(data1)
                prmValDict['timePoint'].extend([tmPt for x in xrange(nFlies)])
                prmValDict['genotype'].extend([gType for x in xrange(nFlies)])
                prmValDict['flyNumber'].extend(np.arange(nFlies)+nFly)
                nFly+=nFlies
            dat = pd.DataFrame(prmValDict, columns=grpLbls)
            multCompFormula = robjects.Formula('result~genotype')
            if len(genotypes)>2:
                multiCmp = nparcomp.mctp(multCompFormula, data=pandas2ri.py2ri(dat),
                                    **{'asy.method': "mult.t", 'type' : "Tukey",
                                    'alternative' : "two.sided", 'plot.simci' : 'FALSE', 
                                    'info' : 'FALSE'})
            else:
                multiCmp = nparcomp.npar_t_test(multCompFormula, 
                                           data=pandas2ri.py2ri(dat),\
                                           **{'method' : "permu",
                                              'alternative' : "two.sided",
                                              'info' : 'FALSE'})
            multComp = pandas2ri.ri2py(multiCmp.rx2('Analysis'))
            multiComps['between'].append([param, tmPt, multComp])
        for gType in genotypes:
            for _,tmPtComb in enumerate(tmptMatrix):
                prmValDict = {x:[] for i,x in enumerate(grpLbls)}
                print tmPtComb
                for tmPt in tmPtComb:
                    data1 = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
                    nFlies = len(data1)
                    prmValDict['result'].extend(data1)
                    prmValDict['timePoint'].extend([tmPt for x in xrange(nFlies)])
                    prmValDict['genotype'].extend([gType for x in xrange(nFlies)])
                    prmValDict['flyNumber'].extend(np.arange(nFlies))
                dat = pd.DataFrame(prmValDict, columns=grpLbls)
                ll2 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
                                   pandas2ri.py2ri(dat['timePoint']),\
                                   pandas2ri.py2ri(dat['flyNumber']),\
                                   **{'description': 'FALSE',\
                                   'plot_RTE':'FALSE','order.warning':'FALSE','time.name': 'timePoint'
                                   })
                multcompAnova = pandas2ri.ri2py(ll2.rx2('ANOVA.test').rx2('p-value'))
                print param, gType, tmPtComb[0], tmPtComb[1], bfTrkStats.statsR.p_adjust(multcompAnova, 'bonferroni', len(tmptMatrix))[0]
                multiComps['within'].append([param, gType, tmPtComb[0], tmPtComb[1], multcompAnova[0],
                           bfTrkStats.statsR.p_adjust(multcompAnova, 'bonferroni', len(tmptMatrix))[0]])
            print "++++++++++++++++++++++++ tSeries ", fig, colId, param, tmPt,"++++++++++++++++++++++++"
            print pdAnova['p-value']['Group']
            print multComp

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
            f.write('\nPosthoc Comparison\n')
        for mC in multiComps['between']:
            with open(statsFName, 'a') as f:
                f.write('\nParameter: %s, timePoint: %d\n'%(mC[0], mC[1]))
            mC[-1].to_csv(statsFName, mode='a', header=True)
        for mC in multiComps['within']:
            with open(statsFName, 'a') as f:
                f.write('Parameter: %s, genotype:, %s, comparison timepoints:, %d - %d, p-Value:, %0.5f, p-Value (Bonferroni adjusted):, %0.5f\n'\
                        %(mC[0], mC[1], mC[2], mC[3], mC[4], mC[5]))
        with open(statsFName, 'a') as f:
            for x in xrange(12):
                f.write('-=-=-=-=-=-=-=-=-=-,')
            f.write('\n\n')





#print bfTrkStats.statsR.p_adjust(1.800611e-02, 'bonferroni', len(multComp))
#print multiCmp                
        
        

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
#    
#    genotypes = pooledTotalDataTmSrs.keys()
#    gtTmptList = []
#    tmPts = [str(i+1) for i in xrange(nTmpts)]
#    for r in itertools.product(genotypes, tmPts):
#        gtTmptList.append(r[0] + '_tmPt' + r[1])
#    gtTmptMatrix = []
#    gtTmptMatrix.extend(itertools.combinations(gtTmptList,2))
#
#    for param in pltParamList:
#        colId = colIdPooledDict[param]
#        print "++++++++++++++++++++++++ tSeries ",colId, param, "++++++++++++++++++++++++"
#        df1 = pd.DataFrame(columns=grpLbls)
#        prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#        nFly=0
#        for g_,gtype in enumerate(genotypes):
#            dSets = [[x[colId] for i_,x in enumerate(pooledTotalDataTmSrs[gtype][tmPt])] for tmPt in xrange(nTmpts)]
#            for i,d in enumerate(dSets):
#                flyNum = list(np.arange(nFly,nFly+len(d)))
#                tPts =  list(np.zeros(len(d))+i)
#                gtypeList = [gtype for x in xrange(len(d))]
#                dfData = {'result':d, 'timePoint': tPts,'flyNumber': flyNum, 'genotype': gtypeList}
#                for j,l in enumerate(grpLbls):
#                    prmValDict[l].extend(dfData[l])
#            nFly+=len(d)
#        df = pd.DataFrame(prmValDict, columns=grpLbls)
#        descStats = pd.DataFrame(pandas2ri.ri2py(fsa.Summarize(statsFormula, data = pandas2ri.py2ri(df))))
#        ll = nparLD.f1_ld_f1(pandas2ri.py2ri(df['result']),
#                             pandas2ri.py2ri(df['timePoint']),\
#                             pandas2ri.py2ri(df['genotype']),\
#                             pandas2ri.py2ri(df['flyNumber']),
#                             **{'description': 'FALSE',\
#                                 'plot_RTE':'FALSE',
#                                 'order.warning':'FALSE',
#                                 })
#        pdWald = r_matrix_to_data_frame(ll.rx2('Wald.test'), getLabels = True)
#        pdAnova = r_matrix_to_data_frame(ll.rx2('ANOVA.test'), getLabels = True)
#        pdPairComp = r_matrix_to_data_frame(ll.rx2('pair.comparison'), getLabels = False)
#        #print ('Wald test\n%r'%pdWald)
#        #print ('ANOVA test\n%r'%pdAnova)
#        #print ('Pariwise Comparison\n%r'%pdPairComp)
#        tmPts = [str(i) for i in xrange(nTmpts)]
#        multiCompsList = []
#        for tmPt in xrange(nTmpts):
#            prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#            nFly=0
#            for gType in genotypes:
#                data1 = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
#                nFlies = len(data1)
#                prmValDict['result'].extend(data1)
#                prmValDict['timePoint'].extend([tmPt for x in xrange(nFlies)])
#                prmValDict['genotype'].extend([gType for x in xrange(nFlies)])
#                prmValDict['flyNumber'].extend(np.arange(nFlies)+nFly)
#                nFly+=nFlies
#            dat = pd.DataFrame(prmValDict, columns=grpLbls)
#            multCompFormula = robjects.Formula('result~genotype')
#            if len(genotypes)>2:
#                multiCmp = nparcomp.mctp(multCompFormula, data=pandas2ri.py2ri(dat),
#                                    **{'asy.method': "mult.t", 'type' : "Tukey",
#                                    'alternative' : "two.sided", 'plot.simci' : 'FALSE', 
#                                    'info' : 'FALSE'})
#            else:
#                multiCmp = nparcomp.npar_t_test(multCompFormula, 
#                                           data=pandas2ri.py2ri(dat),\
#                                           **{'method' : "permu",
#                                              'alternative' : "two.sided",
#                                              'info' : 'FALSE'})
#            multComp = pandas2ri.ri2py(multiCmp.rx2('Analysis'))
#            print "++++++++++++++++++++++++ tSeries ", fig, colId, param, tmPt,"++++++++++++++++++++++++"
#            print pdAnova['p-value']['Group']
#            print multComp
#
#            multiCompsList.append([param, tmPt, multComp])
#        with open(statsFName, 'a') as f:
#            f.write('\nComparing Genotype or/and Time effect for:,%s\n\nDescriptive Stats:\n'%param)
#        descStats.to_csv(statsFName, mode='a', header=True)
#        with open(statsFName, 'a') as f:
#            f.write('\nWald test statistics (WTS) Output\n')
#        pdWald.to_csv(statsFName, mode='a', header=True)
#        with open(statsFName, 'a') as f:
#            f.write('\nANOVA test statistics (ATS) Output\n')
#        pdAnova.to_csv(statsFName, mode='a', header=True)
#        with open(statsFName, 'a') as f:
#            f.write('\nPairwise Comparison\n')
#        pdPairComp.to_csv(statsFName, mode='a', header=True)
#        with open(statsFName, 'a') as f:
#            f.write('\nPosthoc Comparison\n')
#        for mC in multiCompsList:
#            with open(statsFName, 'a') as f:
#                f.write('\nParameter: %s, timePoint: %d\n'%(mC[0], mC[1]))
#            mC[-1].to_csv(statsFName, mode='a', header=True)
#        with open(statsFName, 'a') as f:
#            for x in xrange(10):
#                f.write('-=-=-=-=-=-=-=-=-=-,')
#            f.write('\n\n')
#        for gType in genotypes:
#            nFly=0
#            prmValDict = {x:[] for i,x in enumerate(grpLbls)}
#            for tmPt in xrange(nTmpts):
#                data1 = [x[colId] for x in pooledTotalDataTmSrs[gType][tmPt]]
#                nFlies = len(data1)
#                prmValDict['result'].extend(data1)
#                prmValDict['timePoint'].extend([tmPt for x in xrange(nFlies)])
#                prmValDict['genotype'].extend([gType for x in xrange(nFlies)])
#                prmValDict['flyNumber'].extend(np.arange(nFlies)+nFly)
#            dat = pd.DataFrame(prmValDict, columns=grpLbls)
#            ll2 = nparLD.ld_f1(pandas2ri.py2ri(dat['result']),
#                               pandas2ri.py2ri(dat['timePoint']),\
#                               pandas2ri.py2ri(dat['flyNumber']),\
#                               **{'description': 'FALSE',\
#                               'plot_RTE':'FALSE','order.warning':'TRUE','time.name': 'timePoint'
#                               })
#            multcompAnova = r_matrix_to_data_frame(ll.rx2('ANOVA.test'), getLabels = True)
#            print gType, multcompAnova
##            multCompFormula = robjects.Formula('result~timePoint')
##            multiCmp = nparcomp.mctp_rm(multCompFormula, data=pandas2ri.py2ri(dat),
##                                **{'asy.method': "mult.t", 'type' : "Tukey",
##                                'alternative' : "two.sided", 'plot.simci' : 'FALSE', 
##                                'info' : 'TRUE'})
##            multComp = pandas2ri.ri2py(multiCmp.rx2('Analysis'))
##            print "++++++++++++++++++++++++ tSeries ", fig, colId, param, gType,"++++++++++++++++++++++++"
##            print pdAnova['p-value']['Group']
##            print multComp
##print bfTrkStats.statsR.p_adjust(1.800611e-02, 'bonferroni', len(multComp))
##print multiCmp                
#        
#        

"""



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