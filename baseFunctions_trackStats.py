#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:18:03 2018

@author: aman

functions file for generating a single track statistics file for each fly
"""

import baseFunctions as bf
import numpy as np
from datetime import datetime
from scipy import stats
import os

def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S_%f')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S_%f')
    return (time2-time1).total_seconds()

def getFPS(csvname, logFilePath, FPSsep, FPSIndex):
    '''
    returns the fps of current image folder by looking up the folder details in
    the camera log file in the parent folder
    '''
    csvdetails = csvname.split('_')
    folder = ('_').join([csvdetails[0],csvdetails[1]])
    with open(logFilePath) as f:
        lines = f.readlines()
    fpsValues = []
    for line in lines:
        if all(x in line for x in [folder, 'FPS']):
            fpsValues.append(line.split(FPSsep))
    fpsFrames = [x[FPSIndex-2] for x in fpsValues] # get all lines with required imFolder
    #return the fps from the camloop line with highest number of frames
    return fpsValues[fpsFrames.index(max(fpsFrames))][FPSIndex] 

def getTrackDirection(trackData, minDis):
    '''
    returns a +1 or -1 based on direction of fly movement.
    If the fly walks from left to right  it returns -1 (equivalent to bottom to top for climbing)
    if the fly walks from right to left, it returns +1 (equivalent to top to bottom for climbing)
    Value is calculated purely on the basis of a line fit on the track based on change of X-coordinate w.r.t frames
    '''
    dataLen = len(trackData)
    m,c,r,_,_ = stats.linregress(np.arange(dataLen), trackData[:,0])
    delta = (m*(9*(dataLen/10))+c)-(m*(dataLen/10)+c)
    if delta>=minDis:
        return -1, r
    elif delta<=-minDis:
        return 1, r
    else:
        return 0, r

def getEuDisCenter(pt1, pt2):
    return np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))

def getTotEuDis(xyArr):
    xyArr = np.array(xyArr)
    n = xyArr.shape[0]
    totDis = np.zeros((n-1))
    for i in xrange(0, n-1):
        totDis[i] = getEuDisCenter(xyArr[i], xyArr[i+1])
    return totDis

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points between p1 and p2
    https://stackoverflow.com/questions/43594646/how-to-calculate-the-coordinates-of-the-line-between-two-points-in-python
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])

def diff(x):
    return x[1]-x[0]

def getTrackBreaksPts(xyArray):
    '''
    returns the breakpoints in the XYarray of the centroids of the fly (determined by the '0' in the array)
    '''
    trackStop = 0
    nTracks = []
    for f,x in enumerate(xyArray):
        if x[0]==0 and x[1]==0:
            if trackStop == 0:
                trackStop = f
                nTracks.append([trackStop])
        else:
            if trackStop != 0:
                nTracks[-1].extend([f])
                trackStop = 0
    if trackStop != 0:
        nTracks[-1].extend([f])
    return nTracks

def extrapolateTrack(xyArray, breakPoints, skippedFrThresh, verbose=True):
    '''
    fills the gaps in the xyArray determined by breakPoints, if the gap is less
        than skippedFrThresh
    
    '''
    splitTracks = []
    trackStart = 0
    arrCopy = xyArray.copy()
    for i,x in enumerate(breakPoints):
        if len(x)>1 and diff(x)>0:
            trackBrkLen = diff(x)
            if verbose:
                print trackBrkLen, x
            if trackBrkLen >= skippedFrThresh:
                splitTracks.append([trackStart, x[0]])
                trackStart = x[1]
            else:
                arrCopy[x[0]:x[1],:] = intermediates(xyArray[x[0]-1,:], xyArray[x[1],:], trackBrkLen)
    if (x[0] - trackStart)>=skippedFrThresh:
        splitTracks.append([trackStart, x[0]])
    return splitTracks, arrCopy

pxSize = 70.00/1280     # in mm/pixel
imDataFolder = 'imageData'
headerRowId = 0         # index of the header in the CSV file
fpsSep = ' ' 
fpsIndex = -2
csvHeader = ['trackDetails',
                'track duration (frames)',
                'distance travelled (px)',
                'average instantaneuos speed (px/s)',
                'median instantaneuos speed (px/s)',
                'STD instantaneuos speed (px/s)',
                'average body angle (degrees)',
                'median body angle (degrees)',
                'STD body angle (degrees)',
                'average body length (px)',
                'median body length (px)',
                'STD body length (px)',
                'path straightness (r^2)',
                'geotactic Index',
                'latency (seconds)',
                'timeDelta (seconds)',
                'FPS',
                'Bin Size (frames)',
                'Pixel Size (um/px)',
                'skipped frames threshold (#frames)',
                'track duration threshold (#frames)',
                ]







def getMedVal(array):
    '''
    return the "median" value of the array to be used as median value of array
    '''
    array = np.array(array, dtype=np.float64)
    return np.median(array)

def getTotalNTracks(array, colIdtracknumber):
    '''
    returns the total number of tracks in the given array
    '''
    return len(array)

def getMedTrackDur(array, colIdtrackdur):
    '''
    returns the median track duration from the input array of tracks for a single fly
    '''
    trackDurs = [x[colIdtrackdur] for _,x in enumerate(array)]
    return getMedVal(trackDurs)

def getTotDisTrvld(array, colIdtracklen):
    '''
    returns the total distance travelled by a single fly
    '''
    return np.sum(np.array([x[colIdtracklen] for _,x in enumerate(array)], dtype=np.float64))

def getAvFps(array, colIdfps):
    '''
    returns the average standard deviation of body angle of a single fly 
    '''
    return np.average(np.array([x[colIdfps] for _,x in enumerate(array)], dtype=np.float64))

def getAvSpeed(array, colIdspeedmedian, colIdfps, colIdbodylenmedian):
    '''
    returns the average speed of a single fly by taking the average of the
        median speed of each track
    '''
    return np.average(np.array(
            [(np.float(x[colIdspeedmedian])*np.float(x[colIdfps]))/np.float(x[colIdbodylenmedian])\
            for _,x in enumerate(array)], dtype=np.float64))

def getBodyAngleStd(array, colIdbodyangstd):
    '''
    returns the average standard deviation of body angle of a single fly 
    '''
    return np.average(np.array([x[colIdbodyangstd] for _,x in enumerate(array)], dtype=np.float64))

def getAvBodyLen(array, colIdbodylenmedian):
    '''
    returns the average body length of a single fly by taking the average of the
        median body length of each track
    '''
    return np.average(np.array([x[colIdbodylenmedian] for _,x in enumerate(array)], dtype=np.float64))

def getAvStraightness(array, colIdstraightness):
    '''
    returns the average straightness of a single fly 
    '''
    return np.average(np.array([np.square(np.float(x[colIdstraightness])) for _,x in enumerate(array)], dtype=np.float64))

def getAvGti(array, colIdgti):
    '''
    returns the average geotactic Index of a single fly 
    '''
    return np.average(np.array([x[colIdgti] for _,x in enumerate(array)], dtype=np.float64))

def getMedlatency(array, colIdlatency):
    '''
    returns the median track duration from the input array of tracks for a single fly
    '''
    trackLatencies = [x[colIdlatency] for _,x in enumerate(array)]
    return getMedVal(trackLatencies)

def getTotDur(array, colIdtrackdur):
    '''
    returns the total distance travelled by a single fly
    '''
    return np.sum(np.array([x[colIdtrackdur] for _,x in enumerate(array)], dtype=np.float64))

def getAvDisPerTrk(array, colIdtracklen):
    '''
    returns the average distance covered per track by a single fly
    '''
    return np.average(np.array([x[colIdtracklen] for _,x in enumerate(array)], dtype=np.float64))

def getSex(array, colIdtrackdetails, colors):
    '''
    determine if the fly is male or female, based on the name of the track,
        name of the track contains the sex, either the fly is male or female
    If the sex data is not present, then the fly is determined as unknown
    Returns:
        0 for female
        1 for male
        2 for unknown
    '''
    if len(array)>0:
        trackDetails = array[0][colIdtrackdetails]
        if '_female' in trackDetails:
            return 0, colors['female']
        elif '_male' in trackDetails:
            return 1, colors['male']
        else:
            return 2, colors['unknownSex']
    else:
        return None

def getPooledData(trackStatsData, csvHeader, sexcolors, allTrackData):
    '''
    converts the raw trackStatsData to data for stats and final plotting
        0)  Determine sex, male or female
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
    """FIX THE INPUT colIds and OUTPUT FOR THE DICT OF TITLES"""
    colIdTrackDetails = [csvHeader.index(x) for x in csvHeader if 'trackDetails' in x][0]
    colIdTrackDur = [csvHeader.index(x) for x in csvHeader if 'track duration' in x][0]
    colIdTrackLen = [csvHeader.index(x) for x in csvHeader if 'distance travelled' in x][0]
    colIdTrackSpeed = [csvHeader.index(x) for x in csvHeader if 'median instantaneuos speed ' in x][0]
    colIdBdAngStd = [csvHeader.index(x) for x in csvHeader if 'STD body angle' in x][0]
    colIdTrackBdLen = [csvHeader.index(x) for x in csvHeader if 'median body length' in x][0]
    colIdTrackStrght = [csvHeader.index(x) for x in csvHeader if 'path straightness' in x][0]
    colIdTrackGti = [csvHeader.index(x) for x in csvHeader if 'geotactic Index' in x][0]
    colIdTrackLtncy = [csvHeader.index(x) for x in csvHeader if 'latency' in x][0]
    colIdTrackFps = [csvHeader.index(x) for x in csvHeader if 'FPS' in x][0]
    colIdPxSize = [csvHeader.index(x) for x in csvHeader if 'Pixel Size' in x][0]
    if len(trackStatsData)>0:
        avBdLen = getAvBodyLen(trackStatsData, colIdTrackBdLen)
        fps = getAvFps(trackStatsData, colIdTrackFps)
        pxSize = np.float(trackStatsData[0][colIdPxSize])
        blu = avBdLen*pxSize
        sex, sexColor = getSex(trackStatsData, colIdTrackDetails, sexcolors)
        trackTotNum = getTotalNTracks(trackStatsData, None)
        trackDurMed = getMedTrackDur(trackStatsData, colIdTrackDur)/fps         # in seconds
        totDis = getTotDisTrvld(trackStatsData, colIdTrackLen)/avBdLen          # in BLUs
        avSpeed = getAvSpeed(trackStatsData, colIdTrackSpeed, colIdTrackFps, \
                            colIdTrackBdLen)                                    # in BLU/sec
        avStdBdAng = getBodyAngleStd(trackStatsData, colIdBdAngStd)
        avTrackStrght = getAvStraightness(trackStatsData, colIdTrackStrght)
        avGti = getAvGti(trackStatsData, colIdTrackGti)
        medLatency = getMedlatency(trackStatsData, colIdTrackLtncy)
        totTime = getTotDur(trackStatsData, colIdTrackDur)/fps                  # in seconds
        avDis = getAvDisPerTrk(trackStatsData, colIdTrackLen)/avBdLen           # in BLUs
    else:
        avBdLen = getAvBodyLen(allTrackData, colIdTrackBdLen)
        fps = getAvFps(allTrackData, colIdTrackFps)
        pxSize = np.float(allTrackData[0][colIdPxSize])
        blu = avBdLen*pxSize
        sex, sexColor = getSex(allTrackData, colIdTrackDetails, sexcolors)
        trackTotNum = 0#getTotalNTracks(trackStatsData, None)
        trackDurMed = 0#getMedTrackDur(trackStatsData, colIdTrackDur)/fps         # in seconds
        totDis = 0#getTotDisTrvld(trackStatsData, colIdTrackLen)/avBdLen          # in BLUs
        avSpeed = 0#getAvSpeed(trackStatsData, colIdTrackSpeed, colIdTrackFps, \
                   #         colIdTrackBdLen)                                    # in BLU/sec
        avStdBdAng = np.nan#getBodyAngleStd(trackStatsData, colIdBdAngStd)
        avTrackStrght = np.nan#getAvStraightness(trackStatsData, colIdTrackStrght)
        avGti = np.nan#getAvGti(trackStatsData, colIdTrackGti)
        medLatency = np.nan#getMedlatency(trackStatsData, colIdTrackLtncy)
        totTime = 0#getTotDur(trackStatsData, colIdTrackDur)/fps                  # in seconds
        avDis = 0#getAvDisPerTrk(trackStatsData, colIdTrackLen)/avBdLen           # in BLUs
        
    return [sex, sexColor, trackTotNum, trackDurMed, totDis, avSpeed, \
            avStdBdAng, avBdLen, avTrackStrght, avGti, medLatency, \
            totTime, avDis, blu, fps]

def getUnitTimePltData(untTmData, colId):
    '''
    get the data for plotting the timeSeries plots
    '''
    outData = [[x[colId] for i_,x in enumerate(d)] for i,d in enumerate(untTmData)]
    dataAv = [np.nanmean(x) for i_,x in enumerate(outData)]
    dataerr = [np.nanstd(x)/np.sqrt(len(x)) for i_,x in enumerate(outData)]
    return dataAv, dataerr


def pooledData(dirName, folderList, csvext, unittimedur, threshtotbehavtime, threshtracklenmultiplier,
                  csvheader, csvheaderrow, colIdpooleddict, sexcolors, pltparamlist):
    '''
    returns:
        1)  a list of all data from the input dirName (directory with data from all flies of a single genotype)
        2)  a list of timeSeries data, with time difference of unitTime between two time points, upto threshtotbehavtime
        3)  a list of data for plotting behaviour from total time of behaviour, deteremined by threshtotbehavtime
        4)  a list of data for plotting timeSeries data, with DataAverage and DataError for each parameter for each timepoint
    '''
    #---- read all stats data from each fly folder in a genotype folder ----#
    dirs = folderList
    genotype = dirName.split(os.sep)[-1]
    print('Total fly data present in %s : %d'%(genotype, len(dirs)))
    colIdbodylen = [csvheader.index(x) for x in csvheader if 'median body length' in x][0]
    colIdtracklen = [csvheader.index(x) for x in csvheader if 'distance' in x][0]
    colIdtracktmpt = [csvheader.index(x) for x in csvheader if 'timeDelta' in x][0]
    genotypedata = []
    for i,d in enumerate(dirs):
        fList = bf.getFiles(d, csvext)
        for _, f in enumerate(fList):
            trackDataOutput = bf.readCsv(f)[csvheaderrow+1:]
            if len(trackDataOutput)>0:
                flyStats = []
                blu = np.nanmean(np.array([x[colIdbodylen] for _,x in enumerate(trackDataOutput)], dtype=np.float64))
                threshTrackLen = blu*threshtracklenmultiplier
                #print d,i, blu, threshTrackLen
                for ix,x in enumerate(trackDataOutput):
                    if np.float(x[colIdtracklen])>=threshTrackLen and np.float(x[colIdtracktmpt])<threshtotbehavtime:
                        flyStats.append(x)
                genotypedata.append(flyStats)
            else:
                print('No track data found for :\n%s'%f.split(dirName)[-1])
    #---- read all stats data from each fly folder in a genotype folder ----#
    allData = x
    #---- get plot data of the total data from behaviour from total time measured ----#
    pooledtotaldata = [getPooledData(x, csvheader, sexcolors, allData) for i_,x in enumerate(genotypedata) if len(x)>0]
    
    
    unitTimeN = threshtotbehavtime/unittimedur
    #---- convert all data into a list of unitTimePoints with wach element containing raw
    #       data from the stats csv file for each fly in a genotype folder ----#
    genotypeUnitTimeData = [[] for x in xrange(unitTimeN)]
    pooledUnitTimeData = [[] for x in xrange(unitTimeN)]
    for i_,d in enumerate(genotypedata):
        #print d[0][0]
        for i in xrange(unitTimeN):
            unitTimeData = []
            for i_,f in enumerate(d):
                if i*unittimedur<=np.float(f[colIdtracktmpt])<(i+1)*unittimedur:
                    unitTimeData.append(f)
            genotypeUnitTimeData[i].append(unitTimeData)
            pooledUnitTimeData[i].append(getPooledData(unitTimeData, csvheader, sexcolors, d))
    #---- get plot data of the timeSeries data from behaviour from total time measured ----#
    pltDataUnitTime = []
    pltDataTotal = []
    for i in xrange(len(pltparamlist)):
        #print pltParamList[i], colIdPooledDict[pltParamList[i]]
        pltDataUnitTime.append(getUnitTimePltData(pooledUnitTimeData, colIdpooleddict[pltparamlist[i]]))
        pltDataTotal.append([x[colIdpooleddict[pltparamlist[i]]] for i_,x in enumerate(pooledtotaldata)])
    
    return genotypedata, genotypeUnitTimeData, pooledtotaldata, pooledUnitTimeData, pltDataTotal, pltDataUnitTime


def readFigFolderFile(figFolderFName, figFolderList):
    figFoldersDict = {}
    with open(figFolderFName, 'r') as f:
        lines = f.readlines()
    for figFold in figFolderList:
        figFoldersDict[figFold] = [line for line in lines if figFold in line]
    return figFoldersDict


import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
nlme = importr('nlme')
statsR = importr('stats')
base = importr('base')
multcomp = importr('multcomp')
agr = importr('agricolae')
fsa = importr('FSA')

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
    pVal = krsWallPd[0][0]
    postHocDunn = fsa.dunnTest(formula, data=data, method='bh')
    postHoc = pd.DataFrame(pandas2ri.ri2py(postHocDunn.rx2('res')))
    chiSq = pd.DataFrame(pandas2ri.ri2py(krsWall.rx2('statistic')))
    return {'pvalue': pVal, 'chi-squared': chiSq,'posthoc': postHoc.sort_values(by=['Comparison'])}

def getRAnoval(formula, data):
    '''
    returns the data analysed by Kruskal wallis in 'R' using rpy2 module
    '''
    model1 = robjects.r.lm(formula=formula, data=data)
    anv = robjects.r.anova(model1)
    postHocHSD = agr.HSD_test(model1, 'genotype', group=False, console=False)
    postHoc = pd.DataFrame(pandas2ri.ri2py(postHocHSD.rx2('comparison')))
    smry1 = pd.DataFrame(pandas2ri.ri2py(anv))
    pVal= smry1['Pr(>F)']['genotype']
    fValue = smry1['F value']['genotype']
    return {'pvalue': pVal, 'fvalue': fValue, 'posthoc': postHoc}

def getStatsMultiGrps(pooledData, pooledDataLabels, pltPrmList, colIdPooledDict, statsFormula, dfLbls, pmin, statsOutFName):
    statsList = {}
    statsFormula = robjects.Formula(statsFormula)
    for param in pltPrmList:
        colId = colIdPooledDict[param]
        print ('===== Comparing for %s ====='%(param))
        dSets = [[x[colId] for i_,x in enumerate(pooledData[lbl])] for g_,lbl in enumerate(pooledDataLabels)]
        normP = [stats.normaltest(dSet)[1] for dSet in dSets]
        
        df = createDF(dSets, dfLbls, pooledDataLabels)
        dfr1 = pandas2ri.py2ri(df)
        normP = [stats.normaltest(dSet)[1] for dSet in dSets]
        print 'Min Normal Dist Value: ',min(normP)
        if min(normP)<pmin:
            statsTest = ('Kruskal-Wallis')
            statsData = getRKrusWall(statsFormula, dfr1)
        else:
            statsTest = ('One Way ANOVA')
            statsData = getRAnoval(statsFormula, dfr1)
            print statsData['fvalue']
        statsList[param] = statsData
        f = open(statsOutFName, 'a')
        f.write('\n\nComparing for parameter: , %s\n\n'%(param))
        f.close()
        descStats = pd.DataFrame(pandas2ri.ri2py(fsa.Summarize(statsFormula, data = dfr1)))
        descStats.to_csv(statsOutFName, mode='a', header=True)
        statsKeys = statsData.keys()
        statsKeys.sort()
        statsKeys.remove('posthoc')
        statsKeys.insert(len(statsKeys),'posthoc')
        f = open(statsOutFName, 'a')
        f.write('\nStats:, %s\n'%(statsTest))
        f.close()
        for key in statsKeys:
            f = open(statsOutFName, 'a')
            f.write('%s:'%(key))
            if key=='posthoc':
                f.write('\n')
                f.close()
                statsData[key].to_csv(statsOutFName, mode='a', header=True)
            elif key=='chi-squared':
                f.write(', '+str(statsData[key][0][0])+'\n')
            elif key=='fvalue':
                da = '%0.3f'%(statsData[key])
                print param, key, statsData[key], da, type(da)
                f.write(', '+str(statsData[key])+'\n')
            else:
                f.write(', '+str(statsData[key])+'\n')
        f = open(statsOutFName, 'a')
        [f.write('-=-=-=-=-=-=-=-=-=-,') for x in xrange(10)]
        f.close()

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

def getPooledGTIData(genotypeData, colIdGti, pltPrmList, colIdPooledDict, inCsvHeader, sexColors):
    '''
    returns two lists, each for pooled data for 
            positive and negative GTI data for all parameters in pltPrmList
    '''
    gtiPos, gtiNeg = segregateGti(genotypeData, colIdGti)
    
    gtiPldNeg = [getPooledData(x, inCsvHeader, sexColors, x) for i_,x in enumerate(gtiNeg) if len(x)>0]
    gtiPltPldDataNeg = []
    for i in xrange(len(pltPrmList)):
        gtiPltPldDataNeg.append([x[colIdPooledDict[pltPrmList[i]]] for i_,x in enumerate(gtiPldNeg)])

    gtiPldPos = [getPooledData(x, inCsvHeader, sexColors, x) for i_,x in enumerate(gtiPos) if len(x)>0]
    gtiPltPldDataPos = []
    for i in xrange(len(pltPrmList)):
        gtiPltPldDataPos.append([x[colIdPooledDict[pltPrmList[i]]] for i_,x in enumerate(gtiPldPos)])
    
    pooledTotalGtiData  = {'posGti':gtiPldPos,
                           'negGti':gtiPldNeg}
    pltTotalGtiData     = {'posGti':gtiPltPldDataPos,
                           'negGti':gtiPltPldDataNeg}
    return pooledTotalGtiData, pltTotalGtiData

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

def getPooledSexData(genotypeData, colIdSex, pltPrmList, colIdPooledDict):
    '''
    returns two lists, each for pooled data for 
            positive and negative GTI data for all parameters in pltPrmList
    '''
    sortedMales, sortedFemales = segregateSex(genotypeData, colIdSex)
    
    sxdPltPldDataMales = []
    for i in xrange(len(pltPrmList)):
        sxdPltPldDataMales.append([x[colIdPooledDict[pltPrmList[i]]] for i_,x in enumerate(sortedMales)])
    sxdPltPldDataFemales = []
    for i in xrange(len(pltPrmList)):
        sxdPltPldDataFemales.append([x[colIdPooledDict[pltPrmList[i]]] for i_,x in enumerate(sortedFemales)])
    
    pooledTotalSexData  = {'males':sortedMales,
                           'females':sortedFemales}
    pltTotalSexData     = {'males':sxdPltPldDataMales,
                           'females':sxdPltPldDataFemales}
    return pooledTotalSexData, pltTotalSexData

def getStats2Grps(pooledData, pooledDataLabels, pltPrmList, colIdPooledDict, pmin, statsOutFName, label):
    statsList = {}
    for param in pltPrmList:
        colId = colIdPooledDict[param]
        print ('=====%s for %s ====='%(label, param))
        dSets = [[x[colId] for i_,x in enumerate(pooledData[lbl])] for g_,lbl in enumerate(pooledDataLabels)]
        normP = [stats.normaltest(dSet)[1] for dSet in dSets]
        if min(normP)<pmin:
            statsTest = 'Mann-Whitney'
            statsData = stats.mannwhitneyu(dSets[0], dSets[1])
        else:
            statsTest = 't-test'
            statsData = stats.ttest_ind(dSets[0], dSets[1])
        print ('Min Normal Dist Value: %0.2f, test used %s'%(min(normP), statsTest))
        f = open(statsOutFName, 'a')
        f.write('\n\n%s\n\nComparing for parameter:, %s\n\n'%(label, param))
        f.close()
        descStats =  pd.DataFrame(dSets).transpose().describe()
        descStats.columns = pooledDataLabels
        descStats.transpose().to_csv(statsOutFName, mode='a')
        f = open(statsOutFName, 'a')
        f.write('\nStats:, %s\np-value:,%0.3f\n'%(statsTest,statsData.pvalue))
        [f.write('-=-=-=-=-=-=-=-=-=-,') for x in xrange(10)]
        f.close()
        statsList[param] = statsData.pvalue
    return statsList
    








