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


nTmpts = threshTrackTime/unitTimeDur
headerRowId = bfTrkStats.headerRowId
inCsvHeader = bfTrkStats.csvHeader

pltGTOrder = ['CS','W1118','PINK1RV',
              'Park25xW1118','Lrrk-ex1xW1118',
              'Park25xLrrk-ex1','Trp-Gamma']

pltGTLabels = {'CS':'CS',
               'W1118': r'w$^1$$^1$$^1$$^8$',
               'PINK1RV': r'PINK1$^R$$^V$', 
               'Park25xW1118': r'$\it{park}$$^2$$^5$/w$^1$$^1$$^1$$^8$', 
               'Lrrk-ex1xW1118': 'LRRK$^e$$^x$$^1$/w$^1$$^1$$^1$$^8$', 
               'Park25xLrrk-ex1': r'$\it{park}$$^2$$^5$/LRRK$^e$$^x$$^1$', 
               'Trp-Gamma': r'Trp-$\gamma$'}

colorFemales = (0,0,0)
colorMales = (1,1,1)
colorSexUnknown = (0,0.5,0.5)

sexColors = {'male':        colorMales,
             'female':      colorFemales,
             'unknownSex':  colorSexUnknown
             }

pltParamList = ['trackNum', 'trackDurMed', 
                'disTot', 'speed',
                'straightness', 'gti',
                ]

baseDir = '/media/aman/data/flyWalk_data/climbingData/climbingData_20181201/csvDir/'
fig = 'fig5'
figList = ['fig1','fig2', 'fig3', 'fig4', 'fig5']
#dfLbls = ['paramVal','genotype']
#frmla = dfLbls[0]+'~'+dfLbls[1]
csvExt = ['*trackStats*.csv']
figDataFName = baseDir+'figDataFiles.txt'
figFoldersList = bfTrkStats.readFigFolderFile(figDataFName, figList)
csvDirs = bf.getDirList(baseDir)
currTime = bf.present_time()
print ("=============== Processing for all genotypes =============")
#grpLbls = ['result','timePoint','flyNumber', 'genotype']
#pMin = 0.05


import random
def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))
colorsRandom = [random_color() for c in xrange(1000)]


alfa = 0.71
div = 255.0

def getGTColMarkers(gtypes, div, alfa):
    '''
    returns two lists, 
        one with colors of the specific genotype,
        other with the markers for the genotype
    '''
    colors = []
    markers = []
    for i, gt in enumerate(genotypes):
        print i, gt
        if gt in ('CS', 'cs'):
            colors.append((0/div,0/div,0/div,alfa))
            markers.append('^')
        elif gt in ('CS_males', 'cs'):
            colors.append((0/div,0/div,0/div,alfa))
            markers.append('^')
        elif gt in ('CS_females', 'cs'):
            colors.append((0/div,0/div,0/div,alfa))
            markers.append('^')
        elif gt in ('W1118'):
            colors.append((230/div,218/div,66/div,alfa))
            markers.append('P')
        elif gt in ('Trp-Gamma'):
            colors.append((0/div,158/div,115/div,alfa))
            markers.append('8')
        elif gt in ('Park25xW1118'):
            colors.append((70/div,0/div,10/div,alfa))
            markers.append('o')
        elif gt in ('PINK1RV'):
            colors.append((204/div,121/div,167/div,alfa))
            markers.append('d')
        elif gt in ('Park25xLrrk-ex1'):
            colors.append((86/div,180/div,233/div,alfa))
            markers.append('s')
        elif gt in ('Lrrk-ex1xW1118' ):
            colors.append((180/div,109/div,0/div,alfa))
            markers.append('v')
        else:
            colors.append(random.choice(colorsRandom))
            markers.append('8')
    return colors, markers

nParamsToPlot = len(pltParamList)-1
maxTimeThresh   = 300 # time in seconds for calculation of data from tracks under this much seconds
markerSize = 4.0       #4.0 for 5 minutes, 18 for 30 minutes
lineWidth = 0.95           #0.95 for 5 minutes plot
sWidth = 0.5           #0.012
sSize = 5              #5 for 5 minutes, 300 for 30 minutes
#
disPerMin       = 150
disTotal        = 1000
#csTotalTracks   = 21
##----------- Set Values for setting X-Y limits on the plots -------
nTrackPerMin    = 6
nTrackTotal     = 35
nSecsDurPerMin  = 5
nSecsDurTotal   = 9
avSpeedPerMin   = 10 #7 for others, 5 for KCNJ10
avSpeedTotal    = 10 #8 for others, 4 for KCNJ10
disTotal        = 800
#
unitTime = 60   #seconds
nUnitTimes = maxTimeThresh/unitTime # number of minutes
figWidth = 1.4*nUnitTimes
figHeight = figWidth/1.618
figHeight = figWidth/1.4
#fontSize = (8/7.0)*figWidth
figLabelSize = 12
titleFontSize = 10
yLabelSize = 10
#
nNumTracksTotalTicks    = 6
nDisYTicks              = 6
nSecsDurTotalTicks      = 6

ntSeriesXTicks          = 5

nNumTracksTotalStep = nTrackTotal/nNumTracksTotalTicks#5
nSecsDurTotalStep = nSecsDurTotal/nSecsDurTotalTicks#2

tSeriesXtickStep = nUnitTimes/ntSeriesXTicks

distickStep = 50
disTotalTicks = disTotal/distickStep
disticks = disPerMin/distickStep
disTickScale = 100
disTotalStep = (disTotal/(nDisYTicks*disTickScale))*disTickScale
disTotalTicksStep = disTotalStep/disTickScale

#
#
#
tSeriesPlotIndex = 1
total5MinPlotIndex = 0
#
nPlotStacks = 2
figRatio = [3,1]
tightLayout = False
wSpace = 0.4
hSpace = 0.15
marginLeft = 0.06
marginRight = 0.99
marginTop = 0.90
marginBottom = 0.15
medianWidth = 0.25
meanWidth = 0.25
#

figLabelXoffset = 0.04
xFactor = 0.198
yUp = 0.914
yDown = 0.33
figLabels = [["A","B","C","D","E"],["A'","B'","C'","D'","E'"]]
figLabelPositions = [[[i*xFactor,yUp] for i in xrange(5)], [[i*xFactor,yDown] for i in xrange(5)]]

gtiYUp = 0.914
gtiFigLabelXoffset = 0.128
gtiYDown = 0.33
gtiFigLabels = ["C","C'"]
gtiFigLabelPositions = [[gtiFigLabelXoffset,gtiYUp],[gtiFigLabelXoffset,gtiYDown]]



sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
scatterDataWidth = 0.012
#sCol = (1,1,1)
#
legendHorPos = 0.25
legendVerPos = 1.058
#legendAxesRowSet = total5MinPlotIndex
#legendAxesColSet = nParamsToPlot-1
legendAxesRowGet = tSeriesPlotIndex
legendAxesColGet = nParamsToPlot-1
#
#
#
#bAlpha = 0.5
vAlpha = 0.5
#vAlphaCS = 0.5
#



plotTitles = ['Number of Tracks',
              'Track Duration',
              'Distance Traveled',
              'Average Speed',
              'Straightness',
              'Geotactic Index'
             ]

plotYLabels = ['number',
               'seconds',
               'BLU',
               'BLU/S',
               r'R$^2$ Value',
               'Geotactic Index',
                ]

showMeans = False
showMedians = False
showExtrema = False
medianColor = 'Green'
vPlotLineShowMedian = 'cmedians'
meanColor = 'Orange'
vPlotLineShowMean = 'cmeans'


bwMethod = 'silverman'
boxLineWidth = 0.5
boxprops = dict(linestyle='--', linewidth=boxLineWidth)
whiskerprops = dict(linestyle='--', linewidth=boxLineWidth)
capprops = dict(linestyle='--', linewidth=boxLineWidth)
medianprops = dict(linestyle = None, linewidth=0)
boxPro = dict(boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)


#------ for fig 2------
ax00 = {'yticks': np.arange(0, nTrackPerMin) }
ax10 = {'yticks': np.arange(0,nTrackTotal,nNumTracksTotalStep),
        'ylim':(0,nTrackTotal+1)}
ax01 = {'yticks': np.arange(0, nSecsDurPerMin, 1) ,
        'yticklabels':  np.arange(0,nSecsDurPerMin,1), 
        'ylim':(0,nSecsDurPerMin)}
ax11 = {'yticks': np.arange(0, nSecsDurTotal, nSecsDurTotalStep),
        'yticklabels':  np.arange(0,nSecsDurTotal,nSecsDurTotalStep),
        'ylim':(0,nSecsDurTotal) }
ax02 = {'yticks': np.arange(0,disPerMin,distickStep),
        'ylim': (0,disPerMin-distickStep/2)  }
ax12 = {'yticks': np.arange(0,disTotal,disTotalStep),
        'ylim':(0,disTotal) }
ax03 = {'yticks': np.arange(0,avSpeedPerMin,2),
        'ylim': (0,avSpeedPerMin)}
ax13 = {'yticks': np.arange(0,avSpeedTotal),
        'ylim': (0,avSpeedTotal)}
ax04 = {'ylim': (0, 1.1),
        'yticks': [0, 0.25, 0.5, 0.75, 1],
        'yticklabels': [0, '', 0.5, '', 1]}
ax14 = {'ylim': (0, 1.1),
        'yticks': [0, 0.25, 0.5, 0.75, 1], 
        'yticklabels': [0, 0.25, 0.5, 0.75, 1]}

axP = [[ax10, ax11, ax12, ax13, ax14],
       [ax00, ax01, ax02, ax03, ax04]]



for fig in figList:
    figFName = baseDir+fig+'_'+currTime+'.pdf'
    gtiFigName = baseDir+fig+'_GTI_'+currTime+'.pdf'
    figFNameSvg = baseDir+fig+'_'+currTime+'.svg'
    gtiFigNameSvg = baseDir+fig+'_GTI_'+currTime+'.svg'
    figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))
    
    pooledTotalData  = {}
    pltTotalData = {}
    pltTmSrsData = {}
    
    for genotype in figGenotypes:
        for i_,d in enumerate(csvDirs):
            if genotype == d.split(os.sep)[-1]:
                print ('---Processing for Genotype: %s'%genotype)
                figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
                pldTotalData = bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
                                                      threshTrackLenMulti, inCsvHeader, headerRowId,
                                                      colIdPooledDict, sexColors, pltParamList)
                
                pooledTotalData[genotype] = pldTotalData[2]
                pltTotalData[genotype] = pldTotalData[4]
                pltTmSrsData[genotype] = pldTotalData[5]
    genotypes = pooledTotalData.keys()
    genotypes = sorted(genotypes, key=lambda x: pltGTOrder.index(x))
    colors, markers = getGTColMarkers(genotypes, div, alfa)
    fig, ax = plt.subplots(nPlotStacks,nParamsToPlot, figsize=(figWidth, figHeight),
                           tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
    fig.subplots_adjust(left=marginLeft, right=marginRight, 
                        bottom=marginBottom, top=marginTop, 
                        wspace = wSpace, hspace = hSpace)
    vPlots = []
    bPlots = []
    for i in xrange(len(pltTotalData[genotypes[0]])-1):
        axTot = ax[total5MinPlotIndex, i]
        bp = axTot.boxplot([pltTotalData[g][i] for g in genotypes], sym='', medianprops = medianprops,
                           boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,
                           zorder=1)
        vp = axTot.violinplot([pltTotalData[g][i] for g in genotypes], np.arange(len(genotypes))+1,
                              showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, 
                              bw_method=bwMethod)
        vPlots.append(vp)
        bPlots.append(bp)
        for g, gtype in enumerate(genotypes):
            colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
            scPlt1 = bf.plotScatterCentrd(axTot,pltTotalData[gtype][i], g+1, \
                                          scatterRadius=sSize, scatterColor=colorSex, \
                                          scatterEdgeColor=sEdgCol,scatterAlpha=sAlpha, \
                                          scatterWidth = sWidth, scatterLineWidth = sLinewidth, zOrder=2)
            axTot.hlines(np.mean(pltTotalData[gtype][i]), g+1-meanWidth, g+1+meanWidth,
                         colors=meanColor, alpha=1, zorder=4)
            axTot.hlines(np.median(pltTotalData[gtype][i]), g+1-medianWidth, g+1+medianWidth,
                         colors=medianColor, alpha=1, zorder=4)
            pltData, pltDataErr = pltTmSrsData[gtype][i]
            ax[tSeriesPlotIndex,i].errorbar(np.arange(len(pltData)), pltData, yerr=pltDataErr,\
                                            color=colors[g], fmt='-'+markers[g], label=pltGTLabels[genotypes[g]],\
                                            markersize=markerSize, linewidth = lineWidth)
    for vplot in vPlots:
        #vplot[vPlotLineShowMedian].set_color(medianColor)
        #vplot[vPlotLineShowMedian].set_zorder(4)
        for patch, color in zip(vplot['bodies'], colors):
            patch.set_color(color)
            patch.set_edgecolor(None)
            patch.set_alpha(vAlpha)
    
    for i in xrange(0, len(axP)):
        for j in xrange(0, nParamsToPlot):
            ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                         fontsize=figLabelSize, fontweight='bold', transform=plt.gcf().transFigure)
            plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
            plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
            ax[i,j].set_ylabel(plotYLabels[j], fontsize = yLabelSize)
            plt.setp(ax[i,j], **axP[i][j])
            ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
            if i==tSeriesPlotIndex:
                plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
            elif i==total5MinPlotIndex:
                ax[i,j].set_title(plotTitles[j], y=1, loc='center', fontsize=titleFontSize)
    plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
    ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((-0.25,-2.55),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
    #plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes', **{'fontsize':10})
    plt.text(-11.55,-0.53, 'minutes', fontsize=titleFontSize)
    legendHandles, legendLabels = ax[legendAxesRowGet, legendAxesColGet].get_legend_handles_labels()
    legend = fig.legend(handles=legendHandles,labels=legendLabels,  \
                      loc='lower center', edgecolor=(0,0,0), fontsize=8, ncol=len(genotypes),\
                       bbox_transform=plt.gcf().transFigure)
    
    plt.savefig(figFName)
    plt.savefig(figFNameSvg)
    #plt.show()
    
    gtinParamsToPlot = 1
    gtiFigWidth = 2.2
    gtiFigHeight = figHeight+0.5
    
    gtiMarginLeft = 0.24
    gtilegendVerPos = legendVerPos+0.1
    ax0 = {'yticks': np.arange(-1, 2), 'ylim':(1.2, -1.2) }
    axP1 = [ax0, ax0]
    vPlotPos = np.arange(len(genotypes))
    
    fig, ax = plt.subplots(nPlotStacks, gtinParamsToPlot, figsize=(1.8, gtiFigHeight), tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
    fig.subplots_adjust(left=gtiMarginLeft, right=marginRight, 
                        top=marginTop, bottom=marginBottom, 
                        wspace = wSpace, hspace = hSpace)
    vPlots = []
    bPlots = []
    tp = ax[tSeriesPlotIndex].errorbar
    axTot = ax[total5MinPlotIndex]
    bp = axTot.boxplot([pltTotalData[g][pltParamList.index('gti')] for g in genotypes], sym='', medianprops = medianprops,
                       boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,
                       zorder=1)
    vp = axTot.violinplot([pltTotalData[g][pltParamList.index('gti')] for g in genotypes], np.arange(len(genotypes))+1,
                          showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, 
                          bw_method=bwMethod)
    vPlots.append(vp)
    bPlots.append(bp)
    for g, gtype in enumerate(genotypes):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
        scPlt1 = bf.plotScatterCentrd(axTot,pltTotalData[gtype][pltParamList.index('gti')], g+1, \
                                      scatterRadius=sSize, scatterColor=colorSex, \
                                      scatterEdgeColor=sEdgCol,scatterAlpha=sAlpha, \
                                      scatterWidth = sWidth, scatterLineWidth = sLinewidth, zOrder=2)
        axTot.hlines(np.mean(pltTotalData[gtype][pltParamList.index('gti')]), g+1-meanWidth, g+1+meanWidth,
                     colors=meanColor, alpha=1, zorder=4)
        axTot.hlines(np.median(pltTotalData[gtype][pltParamList.index('gti')]), g+1-medianWidth, g+1+medianWidth,
                     colors=medianColor, alpha=1, zorder=4)
        pltData, pltDataErr = pltTmSrsData[gtype][pltParamList.index('gti')]
        ax[tSeriesPlotIndex].errorbar(np.arange(len(pltData)), pltData, yerr=pltDataErr,\
                                        color=colors[g], fmt='-'+markers[g], label=pltGTLabels[genotypes[g]],\
                                        markersize=markerSize, linewidth = lineWidth)
    for vplot in vPlots:
        for patch, color in zip(vplot['bodies'], colors):
            patch.set_color(color)
            patch.set_edgecolor(None)
            patch.set_alpha(vAlpha)
    for i in xrange(0, len(axP1)):
        ax[i].text(gtiFigLabelPositions[i][0], gtiFigLabelPositions[i][1], gtiFigLabels[i],\
                     fontsize=figLabelSize, transform=plt.gcf().transFigure)
        plt.setp([ax[i].spines[x].set_visible(False) for x in ['top','right']])
        ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        plt.setp(ax[i].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
        plt.setp(ax[i], ylabel = plotYLabels[-1])
        plt.setp(ax[i], **axP1[i])
        if i==tSeriesPlotIndex:
            plt.setp(ax[i],  xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
        #ax[tSeriesPlotIndex,].set_xlabel('minutes')
        legend = fig.legend(handles=legendHandles,labels=legendLabels, bbox_to_anchor=(0.5, -0.06),\
                          loc='lower center',edgecolor=(0,0,0), fontsize=6, ncol=len(genotypes),\
                           bbox_transform=plt.gcf().transFigure)
        plt.setp(ax[total5MinPlotIndex], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
    ax[tSeriesPlotIndex].add_patch(plt.Rectangle((0,1.80),4, 0.40,facecolor='0.9',clip_on=False,linewidth = 0))
    plt.text(1.3,2.1, 'minutes', fontsize=12)
    plt.savefig(gtiFigName, format='pdf')
    plt.savefig(gtiFigNameSvg, format='svg')
    
 













