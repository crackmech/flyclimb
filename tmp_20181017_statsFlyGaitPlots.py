#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 01:22:47 2018

@author: aman
"""

import scikit_posthocs as sp
import csv
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

import os
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import glob
import random
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.anova import AnovaRM
import pandas as pd



markers = ['^','s','v','d','o', 'P']

alfa = 0.71
div = 255.0

colors = [(0/div,0/div,0/div,alfa),#gray
             (200/div,129/div,0/div,alfa),#orange
             (86/div,180/div,233/div,alfa),#Light blue
             (204/div,121/div,167/div,alfa),#pink
             (0/div,158/div,115/div,alfa),#greenish
             (0/div,114/div,178/div,alfa),#blue
             (213/div,94/div,0/div,alfa),#orange
             (240/div,228/div,66/div,alfa),#yellow
             (220/div,198/div,66/div,alfa)#dark yellowish
             ]

def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def random_color():
    levels = range(32,256,2)
    return tuple(random.choice(levels) for _ in range(3))

#colors = [random_color() for i in xrange(20)]
def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row)
    return rows
    
def readConcCsv(ConcCsvFname):
    rows = []
    with open(ConcCsvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append([x.strip('[]') for _,x in enumerate(row)])
    return rows
    
def restrucDataForStats(dataSet, dataSetLabels):
    restrucData = [[[] for y in dataSetLabels] for x in dataSet[0][0]]
    
    maxLen = max([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(1, maxLen):
            try:
                temp_data = dataSet[i][j]
            except:
                temp_data = ''
            for k,d in enumerate(temp_data):
                if d!='':
                    restrucData[k][i].append(np.float(d))
    return restrucData, dataSet[0][0]

def restrucDataForRMA(dataSet, dataSetLabels):
    restrucData = [[[] for y in dataSetLabels] for x in dataSet[0][0]]
    
    maxLen = min([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(1, maxLen):
            try:
                temp_data = dataSet[i][j]
            except:
                temp_data = ''
            for k,d in enumerate(temp_data):
                if d!='':
                    restrucData[k][i].append(np.float(d))
    return restrucData

def getKWmultiComp(data, labels, verbose=False):
    pVals = sp.posthoc_dunn(data, p_adjust='bonferroni')
    if verbose:
        print np.hstack((np.transpose([0]+labels).reshape(4,1),np.vstack((labels,pVals))))
    return [pVals[1,0], pVals[2,0], pVals[2,1]]

def getOWANOVAmultiComp(data, labels, verbose=False):
    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(data)])
    res = pairwise_tukeyhsd(np.concatenate(data), tlabels)
    if verbose:
        print (res.summary())
    return psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total)
    

ctrl = 'W1118_'
exp1 = 'W1118xLrrk-ex1'
exp2 = 'Park25xLrrk-ex1'

dataSets = [ctrl,exp1,exp2]
 
baseDir = '/media/aman/data/flyWalk_data/climbingData/gait/allData/copied/analyzed/'

paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']

csvsCtrl = getFiles(baseDir, [ctrl+'*.csv'])
csvsExp1 = getFiles(baseDir, [exp1+'*.csv'])
csvsExp2 = getFiles(baseDir, [exp2+'*.csv'])

csvs1Ctrl = [x for _,x in enumerate(csvsCtrl) if 'CONCURRENCY' not in x]
csvs1Exp1 = [x for _,x in enumerate(csvsExp1) if 'CONCURRENCY' not in x]
csvs1Exp2 = [x for _,x in enumerate(csvsExp2) if 'CONCURRENCY' not in x]

concCsvCtrl = [x for _,x in enumerate(csvsCtrl) if 'CONCURRENCY' in x]
concCsvExp1 = [x for _,x in enumerate(csvsExp1) if 'CONCURRENCY' in x]
concCsvExp2 = [x for _,x in enumerate(csvsExp2) if 'CONCURRENCY' in x]


dataCtrl = [readCsv(x) for _,x in enumerate(csvs1Ctrl)]
dataExp1 = [readCsv(x) for _,x in enumerate(csvs1Exp1)]
dataExp2 = [readCsv(x) for _,x in enumerate(csvs1Exp2)]

concCtrl = [readConcCsv(x) for _,x in enumerate(concCsvCtrl)]
concExp1 = [readConcCsv(x) for _,x in enumerate(concCsvExp1)]
concExp2 = [readConcCsv(x) for _,x in enumerate(concCsvExp2)]

dC  = concCtrl+dataCtrl
dE1 = concExp1+dataExp1
dE2 = concExp2+dataExp2

pNormMin = 0.05
paramIdx = [0,8,9,11,12,13]

paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']


allStats = []
allStats.append(['Test and Parameter', 'p-Value', 'p-Value', 'p-Value'])
allStats.append(['',ctrl+' vs. '+exp1, ctrl+' vs. '+exp2, exp1+' vs. '+exp2])




plotTitles = ['Number of Tracks\nin 5 minutes',
              'Duration of Tracks',
              'Total Distance Travelled\nin 5 minutes',
              'Average Speed',
              'Path Straightness',
              'Geotactic Index',
              ]

plotTitlesPerUT = ['Number of Tracks',
              'Duration of Tracks',
              'Total Distance Travelled',
              'Average Speed',
              'Path Straightness',
              'Geotactic Index',
              ]

plotYLabels = ['Number of Tracks',
                'duration of Tracks\n(s)',
                'Distance Traveled\n'+r'(BLU x10$^3$)',
                'Average Speed\n(BLU/S)',
                'Path Straightness\n'+r'(R$^2$ Value)',
                'Geotactic Index',
                ]

sWidth = 0.012
sSize = 5
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
sCol = (0,0,0)


#vPlotPos = np.arange(len(genotypes))

def plotScatter(axis, data, scatterX, swidth = sWidth, \
                sradius = sSize , scolor = sCol,\
                smarker = sMarker, salpha = sAlpha, \
                slineWidth = sLinewidth, sedgeColor = sEdgCol, zOrder=0):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot
    '''
    return axis.scatter(np.linspace(swidth+scatterX, -swidth+scatterX,len(data)), data,\
            s=sradius, color = scolor, marker=smarker,\
            alpha=salpha, linewidths=slineWidth, edgecolors=sedgeColor, zorder=zOrder )


#---get the per unit time data ----

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)+0))
    ax.set_xticklabels(labels)
    ax.set_xlim(-1, len(labels))


pltCols = 3
pltRows = len(paramIdx)/pltCols
figWidth = 1.4*5
figHeight = figWidth/1.618
fontSize = (8/7.0)*figWidth
tightLayout = False
wSpace = 0.4
hSpace = 0.15
marginLeft = 0.05
marginRight = 0.99
marginTop = 0.97
marginBottom = 0.082

showMeans = False
showMedians = True
showExtrema = False
medianColor = 'Orange'
vPlotLineShow = 'cmedians'


bwMethod = 'silverman'
boxLineWidth = 0.5
boxprops = dict(linestyle='--', linewidth=boxLineWidth)
whiskerprops = dict(linestyle='--', linewidth=boxLineWidth)
capprops = dict(linestyle='--', linewidth=boxLineWidth)
medianprops = dict(linestyle = None, linewidth=0)
boxPro = dict(boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)

#
#legendHorPos = 0.32
#legendVerPos = 1.058
#legendAxesRowSet = total5MinPlotIndex
#legendAxesRowGet = tSeriesPlotIndex
#legendAxesColSet = 4
#legendAxesColGet = 4
#
#
#nParamsToPlot = paramIdx
#
#
#
#ax00 = {'yticks': np.arange(nTrackPerMin+1) }
##ax10 = {'yticks': np.arange(0,36,5), 'ylim':(0,36)}
#ax10 = {'yticks': np.arange(0,nTrackPerMin*nUnitTimes,nUnitTimes), 'ylim':(0,nTrackPerMin*nUnitTimes+1)}
#ax01 = {'yticks': np.arange(0, trackFPS*nSecsDurPerMin, 2*trackFPS) , 'yticklabels':  np.arange(0,nSecsDurPerMin,2), 'ylim':(0,trackFPS*nSecsDurPerMin)}
#ax11 = {'yticks': np.arange(0, trackFPS*nSecsDurTotal, 2*trackFPS),'yticklabels':  np.arange(0,nSecsDurTotal,2), 'ylim':(0,trackFPS*nSecsDurTotal) }
#ax02 = {'yticks': np.arange(0,disPerMin,distickStep), 'yticklabels': np.arange(disticks) }
#ax12 = {'yticks': np.arange(0,disTotal,disTotalStep), 'yticklabels': np.arange(0,disTotalTicks,disTotalTicksStep), 'ylim':(0,disTotal) }
#ax03 = {'yticks': np.arange(0,avSpeedVal,2)}
#ax13 = {'yticks': np.arange(0,avSpeedVal,2)}
#ax04 = {'ylim': (0, 1.1), 'yticks': [0, 0.5, 1]}
#ax14 = {'ylim': (0, 1.5), 'yticks': [0, 0.5, 1]}
#ax05 = {'ylim': (1.2, -1.5), 'yticks': [-1, 0, 1]}
#ax15 = {'ylim': (1.2, -1.5), 'yticks': [-1, 0, 1]}
#axP = [
#        [ax10, ax11, ax12, ax13, ax14, ax15],
#        [ax00, ax01, ax02, ax03, ax04, ax05]
#      ]
#
#
#
#
#ptime = present_time()
#figDir = baseDir+'../'
#dpi = 300
#
#sMarkers  = ['o' for x in markers]
#fig, ax = plt.subplots(pltRows,pltCols, figsize=(figWidth, figHeight), tight_layout = tightLayout)
#fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight, top=marginTop, wspace = wSpace, hspace = hSpace)
#legendHandles, legendLabels = ax[legendAxesRowGet, legendAxesColGet].get_legend_handles_labels()
#ax[legendAxesRowSet, legendAxesColSet].legend(handles=legendHandles,labels=legendLabels, bbox_to_anchor=(legendHorPos, legendVerPos), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=1).draggable()
#bPlots = []
#vPlots = []
#for i in xrange(0, nParamsToPlot):
#    plotData = dataToPlot[i]
#    vp = ax[total5MinPlotIndex, i].violinplot([da for da in plotData], vPlotPos+1, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
#    bp = ax[total5MinPlotIndex, i].boxplot([da for da in plotData], sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
#    for s,scatterPlotData in enumerate(plotData):
#        plotScatter(ax[total5MinPlotIndex, i], scatterPlotData, scatterX = s+1, scatterMarker = sMarkers[s], scatterColor = genotypeMarker[s], zOrder=2)
#    vPlots.append(vp)
#    bPlots.append(bp)
#for vplot in vPlots:
#    vplot[vPlotLineShow].set_color(medianColor)
#    for patch, color in zip(vplot['bodies'], colors):
#        patch.set_color(color)
#        patch.set_edgecolor(None)
#        patch.set_alpha(vAlpha)
#
#for i in xrange(0, len(axP)):
#    for j in xrange(0, nParamsToPlot):
#        plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
#        plt.setp(ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
#        plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
#        plt.setp(ax[i,j], ylabel = plotYLabels[j])
#        plt.setp(ax[i,j], **axP[i][j])
#        if i==tSeriesPlotIndex:
#                plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep), xlabel = 'minutes')
#plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
#plt.savefig(combinedFigNamePng, dpi=dpi, format='png')
#plt.savefig(combinedFigNameSvg, format='svg')




def toMatrix(l, n):
    return [l[i:i+n] for i in xrange(0, len(l), n)]


paramIdx = [0,8,9,11,12,13]
paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']

plotParamIdx = toMatrix(paramIdx, pltCols)
plotParamTitles = toMatrix(paramTitles, pltCols)

sWidth = 0.12
sSize = 5

sSizes = [[3,0.1,1],[0.1,1,8]]
sAlphas = [[0.5, 0.3, 0.1], [0.3, 0.1, 0.5]]


fig, ax = plt.subplots(pltRows,pltCols, figsize=(figWidth, figHeight), tight_layout = tightLayout)
bPlots = []
vPlots = []
step = 4#2*len(dataSets)
for i in xrange(pltRows):
    for j in xrange(pltCols):
        n = plotParamIdx[i][j]
        print '\n--------',plotParamTitles[i][j],'--------'
        testData, xLabels = restrucDataForStats([dC[n],dE1[n],dE2[n]], dataSets)
        for d in xrange(len(dataSets)):
            tData = [x[d] for _,x in enumerate(testData)]
            pos = np.arange(0, len(tData)*step, step)+d
            print pos
            plotData = tData
            vp = ax[i][j].violinplot(plotData, positions = pos, showmeans=showMeans,\
                                                        showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
            bp = ax[i][j].boxplot(plotData, sym='', medianprops = medianprops,\
                               positions = pos, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
            for s,scatterPlotData in enumerate(plotData):
                plotScatter(ax[i][j], scatterPlotData, scatterX = d+(s*step),\
                            swidth = sWidth, sradius = sSizes[i][j], salpha=sAlphas[i][j], zOrder=2)
            ax[i][j].set_xlim(-1, pos[-1]+1)
            ax[i][j].set_title(plotParamTitles[i][j])
        ax[i][j].set_xticks(np.arange(1,pos[-1], step))
        ax[i][j].set_xticklabels(xLabels)
        print np.arange(1,pos[-1], step), xLabels
        vPlots.append(vp)
        bPlots.append(bp)

plt.show()







n=0
testData = restrucDataForStats([dC[n],dE1[n],dE2[n]], dataSets)
for d in xrange(len(dataSets)):
    tData = [x[d] for _,x in enumerate(testData)]
    pos = np.arange(0, len(tData)*len(dataSets),len(dataSets) )+d
    print pos
    plotData = tData
    vp = plt.violinplot(plotData, positions = pos, showmeans=showMeans,\
                                                showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
    bp = plt.boxplot(plotData, sym='', medianprops = medianprops,\
                       positions = pos, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
    for s,scatterPlotData in enumerate(plotData):
        plotScatter(plt, scatterPlotData, scatterX = d+(s*len(dataSets)), scatterMarker = 'o',\
                    scatterColor = (0,0,0), zOrder=2)
plt.xlim(-1, pos[-1]+2)
plt.show()

#        label = '---'+paramTitles[p]+'_'+dC[n][0][i]+'---'
#        print label
#        normP = []
#        for j in xrange(len(testData[i])):
#            _, pValue = stats.normaltest(testData[i][j])
#            normP.append(pValue)
#        if min(normP)<pNormMin:
#            testUsed = 'Kruskal-Wallis test'
#            _, statsP = stats.kruskal(*testData[i])
#            print testUsed+' pValue:', statsP,'---'
#            multiCompP = getKWmultiComp(testData[i], dataSets, verbose=False)
#        else:
#            testUsed = 'One Way ANOVA'
#            _, statsP = stats.f_oneway(*testData[i])
#            print testUsed+' pValue:', statsP
#            multiCompP = list(getOWANOVAmultiComp(testData[i], dataSets, verbose=False))
#        allStats.append([label])
#        allStats.append(['normalityTestStats']+normP)
#        allStats.append([testUsed,statsP])
#        allStats.append(['MultipleComparisons p-Value']+multiCompP)
#        allStats.append([])


#for p,n in enumerate(paramIdx):
#    print '\n--------',paramTitles[p],'--------'
#    testData = restrucDataForStats([dC[n],dE1[n],dE2[n]], dataSets)
#    for i in xrange(len(testData)):
#        label = '---'+paramTitles[p]+'_'+dC[n][0][i]+'---'
#        print label
#        normP = []
#        for j in xrange(len(testData[i])):
#            _, pValue = stats.normaltest(testData[i][j])
#            normP.append(pValue)
#        if min(normP)<pNormMin:
#            testUsed = 'Kruskal-Wallis test'
#            _, statsP = stats.kruskal(*testData[i])
#            print testUsed+' pValue:', statsP,'---'
#            multiCompP = getKWmultiComp(testData[i], dataSets, verbose=False)
#        else:
#            testUsed = 'One Way ANOVA'
#            _, statsP = stats.f_oneway(*testData[i])
#            print testUsed+' pValue:', statsP
#            multiCompP = list(getOWANOVAmultiComp(testData[i], dataSets, verbose=False))
#        allStats.append([label])
#        allStats.append(['normalityTestStats']+normP)
#        allStats.append([testUsed,statsP])
#        allStats.append(['MultipleComparisons p-Value']+multiCompP)
#        allStats.append([])
#
#
#csvOutFile = baseDir+'stats_gait.csv'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(allStats)
#
#
#
#
#concData = [concCtrl[0], concExp1[0], concExp2[0]]
#concRestrucData = restrucDataForRMA(concData, dataSets)
#
#def getRMAnova(dataSet, labels, verbose=False):
#    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(dataSet)])
#    concatData = np.concatenate(dataSet)
#    ids = np.concatenate([np.arange(len(x)) for _,x in enumerate(dataSet)])
#    d = {'id':ids, 'rt':concatData, 'cond':tlabels}
#    df = pd.DataFrame(d)
#    anovarm = AnovaRM(df, 'rt', 'id', within=['cond'])
#    res = anovarm.fit()
#    if verbose:
#        print (res.summary())
#    return res
#
#
#rmAnovaStats = [['Repeated Measures ANOVA for Concurrency states (S0, S1, S2, S3)']]
#for i,x in enumerate(concRestrucData):
#    rma = getRMAnova(x, dataSets, True)
#    rmAnovaStats.append([rma.anova_table])
#    rmAnovaStats.append([rma.summary()])
#
#
#rmAnovaStats
#csvOutFile = baseDir+'stats_conc.txt'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(rmAnovaStats)
#







 


