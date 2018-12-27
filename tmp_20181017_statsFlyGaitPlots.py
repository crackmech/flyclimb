#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 01:22:47 2018

@author: aman
"""

import scikit_posthocs as sp
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.anova import AnovaRM
import pandas as pd

import numpy as np
import os
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import glob
import random
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
col=1
plt.rcParams['axes.facecolor'] = (col,col,col)
fontSize = 14
plt.rc('axes', titlesize = fontSize)

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

sWidth = 0.012
sSize = 5
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
sCol = (0,0,0)


# Leg order
legs = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3']
# Contra pair order
contralateral_pairs = map(str, [1, 2, 3])
# Ipsi pair order
ipsilateral_pairs = map(str, [1, 2, 3, 4])
concurrency_states = map(str, [3, 2, 1, 0])
concurrency_states = ['S0', 'S1', 'S2', 'S3']

specs = {
          'LEG_BODY_ANGLE'    :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(-220, 220), 'YLABEL':'Degrees', 'TITLE':'Leg-Body Angle', 'sSize': 1, 'sAlpha':0.3},
          'SWING_AMPLITUDE'   :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(0, 2500), 'YLABEL':'um', 'TITLE':'Swing Amplitude', 'sSize': 0.5, 'sAlpha':0.3},
          'SWING_DURATION'    :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(0, 125), 'YLABEL':'ms', 'TITLE':'Swing Duration', 'sSize': 1, 'sAlpha':0.1},
          'STANCE_AMPLITUDE'   :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(0, 2500), 'YLABEL':'um', 'TITLE':'Stance Amplitude', 'sSize': 0.5, 'sAlpha':0.3},
          'STANCE_DURATION'   :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(0, 200), 'YLABEL':'ms', 'TITLE':'Stance Duration', 'sSize': 1, 'sAlpha':0.1},
          'AEPx'              :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(-5000, 5000), 'YLABEL':'um', 'TITLE':'Anterior Extreme Position w.r.t. X', 'sSize': 1, 'sAlpha':0.3},
          'PEPx'              :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(-5000, 5000), 'YLABEL':'um', 'TITLE':'Posterior Extreme Position w.r.t. X', 'sSize': 1, 'sAlpha':0.3},
          'AEA'               :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(-220, 220), 'YLABEL':'Degrees', 'TITLE':'Anterior Extreme Angle', 'sSize': 1, 'sAlpha':0.3},
          'PEA'               :   {'PLOT':'BOX', 'ENTITIES':legs, 'YLIMS':(-220, 220), 'YLABEL':'Degrees', 'TITLE':'Posterior Extreme Angle', 'sSize': 1, 'sAlpha':0.3},
          'CCI'               :   {'PLOT':'BOX', 'ENTITIES':contralateral_pairs, 'YLIMS':(0.75, 1.02), 'YLABEL':'/s', 'TITLE':'Contra-lateral Coordination Index', 'sSize': 1, 'sAlpha':0.3},
          'ICI'               :   {'PLOT':'BOX', 'ENTITIES':ipsilateral_pairs, 'YLIMS':(0.75, 1.02), 'YLABEL':'/s', 'TITLE':'Ipsi-lateral Coordination Index', 'sSize': 1, 'sAlpha':0.3},
          'WALK_SPEED'        :   {'PLOT':'BOX', 'ENTITIES':['Speed'], 'YLIMS':(0, 40), 'YLABEL':'mm/s', 'TITLE':'Average Walking Speed', 'sSize': 8, 'sAlpha':0.5},
          'STOLEN_SWINGS'     :   {'PLOT':'BOX', 'ENTITIES':['Swings/cycle'], 'YLIMS':(0, 1.2), 'YLABEL':'#/cycle', 'TITLE':'Stolen Swings Per Cycle', 'sSize': 1, 'sAlpha':0.3},
          'CONCURRENCY'       :   {'PLOT':'PIE', 'ENTITIES':concurrency_states, 'YLIMS':(0, 100.0), 'YLABEL':'%', 'TITLE':'Proportional Concurrency States', 'sSize': 3, 'sAlpha':0.5},
          }

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

colorsRandom = [random_color() for c in xrange(1000)]
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
    minLen = min([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(1, minLen):
            for k,d in enumerate(dataSet[i][j]):
                restrucData[k][i].append(np.float(d))
    return restrucData, dataSet[0][0]

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

def getStats(tData, datasetLabels, param, labels, pNormMin, verbose=False):
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
        for j in xrange(len(tData[i])):
            _, pValue = stats.normaltest(tData[i][j])
            normP.append(pValue)
        if min(normP)<pNormMin:
            testUsed = 'Kruskal-Wallis test'
            _, statsP = stats.kruskal(*tData[i])
            print testUsed+' pValue:', statsP,'---'
            multiCompP = getKWmultiComp(tData[i], datasetLabels, verbose)
        else:
            testUsed = 'One Way ANOVA'
            _, statsP = stats.f_oneway(*tData[i])
            print testUsed+' pValue:', statsP
            multiCompP = list(getOWANOVAmultiComp(tData[i], datasetLabels, verbose))
        statsData.append([label])
        statsData.append(['normalityTestStats']+normP)
        statsData.append([testUsed,statsP])
        statsData.append(['MultipleComparisons p-Value']+multiCompP)
        statsData.append([])
    return statsData


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


ctrl = 'W1118_'
exp1 = 'W1118xLrrk-ex1'
exp2 = 'Park25xLrrk-ex1'

dataSets = [ctrl,exp1,exp2]
 
baseDir = '/media/aman/data/flyWalk_data/climbingData/gait/allData/copied/analyzed/'

paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']

csvsCtrl = getFiles(baseDir, [ctrl+'*.csv'])
csvsExp1 = getFiles(baseDir, [exp1+'*.csv'])
csvsExp2 = getFiles(baseDir, [exp2+'*.csv'])

pNormMin = 0.05
paramIdx = [0,8,9,11,12,13]

pltCols = 3
pltRows = len(paramIdx)/pltCols
figWidth = 1.4*5
figHeight = figWidth/1.618
fontSize = (8/7.0)*figWidth
tightLayout = True
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

sWidth = 0.12
legendHorPos = -0.2
legendVerPos = -0.1
legendAxesRowSet = pltRows-1
legendAxesColSet = (pltCols/2)+1
step = len(dataSets)+1

bwMethod = 'silverman'
boxLineWidth = 0.5
boxprops = dict(linestyle='--', linewidth=boxLineWidth)
whiskerprops = dict(linestyle='--', linewidth=boxLineWidth)
capprops = dict(linestyle='--', linewidth=boxLineWidth)
medianprops = dict(linestyle = None, linewidth=0)
boxPro = dict(boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)

paramTitles1 = [['CONCURRENCY', 'SWING_AMPLITUDE', 'STANCE_AMPLITUDE'], ['WALK_SPEED', 'SWING_DURATION', 'STANCE_DURATION']]

genotypes = []
colors = []
markers = []
for i, gt in enumerate(dataSets):
    if gt in ('W1118_', 'w1118'):
        genotypes.append(r'W$^1$$^1$$^1$$^8$')
        colors.append((230/div,218/div,66/div,alfa))
        markers.append('P')
    elif gt in ('PARK25xLrrk-ex1', 'Park25xLrrk-ex1'):
        genotypes.append(r'Park$^2$$^5$/Lrrk$^e$$^x$$^1$')
        colors.append((86/div,180/div,233/div,alfa))
        markers.append('s')
    elif gt in ('W1118xLrrk-ex1', 'w1118xLrrk-ex1'):
        genotypes.append(r'Lrrk$^e$$^x$$^1$/W$^1$$^1$$^1$$^8$')
        colors.append((180/div,109/div,0/div,alfa))
        markers.append('v')
    else:
        genotypes.append(gt)
        colors.append(random.choice(colorsRandom))
        markers.append('8')
    print i, gt, len(colors), colors

fig, ax = plt.subplots(pltRows,pltCols, figsize=(figWidth, figHeight))#, tight_layout = tightLayout)
fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight, top=marginTop, wspace = wSpace, hspace = hSpace)
legendPatches = [mpatches.Patch(color=c, label=genotypes[i]) for i,c in enumerate(colors)]
allStats = []
concStats = []
csvOutFname = 'stats_gait.csv'
for i in xrange(pltRows):
    for j in xrange(pltCols):
        param = paramTitles1[i][j]
        if param=='CONCURRENCY':
            getData = readConcCsv
            doStats = getConcStats
            restrucData  = restrucDataForRMA
            statsList = concStats
        else:
            getData = readCsv
            doStats = getStats
            restrucData  = restrucDataForStats
            statsList = allStats
        dC = [getData(x) for _,x in enumerate(csvsCtrl) if param in x][0]
        dEx1 = [getData(x) for _,x in enumerate(csvsExp1) if param in x][0]
        dEx2 = [getData(x) for _,x in enumerate(csvsExp2) if param in x][0]
        print '\n--------',param,'--------'
        testData, lbls = restrucData([dC,dEx1,dEx2], dataSets)
        statsList.append(doStats(testData, dataSets, param, lbls, pNormMin, verbose=False))
        pltTitle = specs[param]['TITLE']
        yLabel = specs[param]['YLABEL']
        yLim = specs[param]['YLIMS']
        xLabels = specs[param]['ENTITIES']
        sSize = specs[param]['sSize']
        sAlpha = specs[param]['sAlpha']
        for d in xrange(len(dataSets)):
            tData = [x[d] for _,x in enumerate(testData)]
            pos = np.arange(0, len(tData)*step, step)+d
            plotData = tData
            vp = ax[i][j].violinplot(plotData, positions = pos, showmeans=showMeans,\
                                     showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
            bp = ax[i][j].boxplot(plotData, sym='', medianprops = medianprops, labels =[genotypes[d] for x in plotData],\
                                    positions = pos, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
            for s,scatterPlotData in enumerate(plotData):
                plotScatter(ax[i][j], scatterPlotData, scatterX = d+(s*step),\
                            swidth = sWidth, sradius = sSize, salpha=sAlpha, zOrder=2)
            vp[vPlotLineShow].set_color(medianColor)
            for patch in vp['bodies']:
                patch.set_color(colors[d])
                patch.set_edgecolor(None)
        ax[i][j].set_xlim(-1, pos[-1]+1)
        ax[i][j].set_ylim(yLim)
        ax[i][j].set_ylabel(yLabel)
        ax[i][j].set_title(pltTitle)
        ax[i][j].set_xticks(np.arange(1,pos[-1], step))
        ax[i][j].set_xticklabels(xLabels)
csvOutFile = baseDir+csvOutFname
concStats[0].to_csv(csvOutFile, sep=',')
with open(csvOutFile, "a") as f:
    writer = csv.writer(f)
    writer.writerows([[' ',' '] for x in xrange(3)])
    for _,row in enumerate(allStats):
        writer.writerows(row)

ax[legendAxesRowSet,legendAxesColSet].legend(handles=legendPatches,bbox_to_anchor=(legendHorPos, legendVerPos), ncol=3).draggable()
#plt.show()
























