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
from scipy import stats
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
figList = ['fig2', 'fig3', 'fig4', 'fig5']
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


#============================================================================================================

#for fig in figList:
#    statsFName = baseDir+fig+'_stats_perMin_'+currTime+'.csv'
#    figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))
#    totalData  = {}
#    totalDataTmSrs = {}
#    pooledTotalData  = {}
#    pooledTotalDataTmSrs = {}
#    pltTotalData = {}
#    pltTmSrsData = {}
#    for genotype in figGenotypes:
#        for i_,d in enumerate(csvDirs):
#            if genotype == d.split(os.sep)[-1]:
#                print ('---Processing for Genotype: %s'%genotype)
#                figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
#                pldTotalData = bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
#                                                      threshTrackLenMulti, inCsvHeader, headerRowId,
#                                                      colIdPooledDict, sexColors, pltParamList)
#                totalData[genotype] = pldTotalData[0]
#                totalDataTmSrs[genotype] = pldTotalData[1]
#                pooledTotalData[genotype] = pldTotalData[2]
#                pooledTotalDataTmSrs[genotype] = pldTotalData[3]
#                pltTotalData[genotype] = pldTotalData[4]
#                pltTmSrsData[genotype] = pldTotalData[5]
#    genotypes = pooledTotalDataTmSrs.keys()

#============================================================================================================





#
##for fig in figList:
##    statsFName = baseDir+fig+'_stats_perMin_'+currTime+'.csv'
#fig = figList[-1]
#
#figGenotypes = list(set([f.split(os.sep)[1].split('_')[-1] for f in figFoldersList[fig]]))
#
#pooledTotalData  = {}
#pltTotalData = {}
#pltTmSrsData = {}
#
#for genotype in figGenotypes:
#    for i_,d in enumerate(csvDirs):
#        if genotype == d.split(os.sep)[-1]:
#            print ('---Processing for Genotype: %s'%genotype)
#            figFoldList = [os.path.join(d,folder.split(os.sep)[-1].rstrip('\n')) for folder in figFoldersList[fig] if genotype in folder]
#            pldTotalData = bfTrkStats.pooledData(d, figFoldList, csvExt, unitTimeDur, threshTrackTime,
#                                                  threshTrackLenMulti, inCsvHeader, headerRowId,
#                                                  colIdPooledDict, sexColors, pltParamList)
#            
#            pooledTotalData[genotype] = pldTotalData[2]
#            pltTotalData[genotype] = pldTotalData[4]
#            pltTmSrsData[genotype] = pldTotalData[5]
#genotypes = pooledTotalData.keys()
#
##============================================================================================================
#
##------test plots
##---- Plot the total data from behaviour from total time measured ----#
#gtype = genotypes[0]
##colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
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





#for fig in figList:
#    statsFName = baseDir+fig+'_stats_perMin_'+currTime+'.csv'
fig = figList[-3]

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

#order = {'CS': 0,
#         'W1118': 1,
#         'PINK1RV': 2,
#         'Park25xW1118': 3,
#         'Lrrk-ex1xW1118': 4,
#         'Park25xLrrk-ex1': 5,
#         'Trp-Gamma': 6,
#         }
#
#gtypeDict = [{'gt':x} for x in genotypes]
#gtSorted = sorted(gtypeDict, key=lambda d: order[d['gt']])
#gtSorted = [x['gt'] for x in gtSorted]


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
figWidth = 1.6*nUnitTimes
figHeight = figWidth/1.618
#fontSize = (8/7.0)*figWidth
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
marginLeft = 0.05
marginRight = 0.99
marginTop = 0.97
marginBottom = 0.082
medianWidth = 0.25
#
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
scatterDataWidth = 0.012
#sCol = (1,1,1)
#
#legendHorPos = 0.25
#legendVerPos = 1.058
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
import random
def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))
colorsRandom = [random_color() for c in xrange(1000)]


alfa = 0.71
div = 255.0

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
        markers.append('o')
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
    print i, gt, len(colors), colors

#legendHorPos = 0.18
#legendVerPos = 1.058
figLabelSize = 12
figLabelXoffset = 0.03
xFactor = 0.2
yUp = 0.98
yDown = 0.3
figLabels = [["A","B","C","D","E"],["A'","B'","C'","D'","E'"]]
figLabelPositions = [[[i*xFactor,yUp] for i in xrange(5)], [[i*xFactor,yDown] for i in xrange(5)]]

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
#ax12 = {'yticks': np.arange(0,disTotal,disTotalStep),
#        'yticklabels': np.arange(0,disTotalTicks,disTotalTicksStep),
#        'ylim':(0,disTotal) }
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

fig, ax = plt.subplots(nPlotStacks,nParamsToPlot, figsize=(figWidth, figHeight),
                       tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight,
                    top=marginTop, wspace = wSpace, hspace = hSpace)
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
        axTot.hlines(np.median(pltTotalData[gtype][i]), g+1-medianWidth, g+1+medianWidth,
                     colors=medianColor, alpha=0.8, zorder=4)
        pltData, pltDataErr = pltTmSrsData[gtype][i]
        ax[tSeriesPlotIndex,i].errorbar(np.arange(len(pltData)), pltData, yerr=pltDataErr,\
                                        color=colors[g], fmt='-'+markers[g], label=genotypes[g],\
                                        markersize=markerSize, linewidth = lineWidth)
for vplot in vPlots:
    vplot[vPlotLineShow].set_color(medianColor)
    vplot[vPlotLineShow].set_zorder(4)
    for patch, color in zip(vplot['bodies'], colors):
        patch.set_color(color)
        patch.set_edgecolor(None)
        patch.set_alpha(vAlpha)

for i in xrange(0, len(axP)):
    for j in xrange(0, nParamsToPlot):
        ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                     fontsize=figLabelSize, transform=plt.gcf().transFigure)
        plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
        plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
        plt.setp(ax[i,j], ylabel = plotYLabels[j])
        plt.setp(ax[i,j], **axP[i][j])
        ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        if i==tSeriesPlotIndex:
            plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
        elif i==total5MinPlotIndex:
            ax[i,j].set_title(plotTitles[j], y=1.08, loc='center')
plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((-0.25,-2.25),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes')
legendHandles, legendLabels = ax[legendAxesRowGet, legendAxesColGet].get_legend_handles_labels()
legend = fig.legend(handles=legendHandles,labels=legendLabels, bbox_to_anchor=(0.5, -0.08), \
                  loc='lower center', edgecolor=(0,0,0), fontsize=8, ncol=len(genotypes),\
                   bbox_transform=plt.gcf().transFigure)

plt.savefig('/home/aman/Desktop/figTest.pdf')

"""



dataToPlot = [genotypeNTracks,
              genotypeLenTrack,
              genotypeDis,
              genotypeAvSpeed,
              genotypeStraight,
              genotypeGeoTacInd]

fig, ax = plt.subplots(nPlotStacks,nParamsToPlot, figsize=(figWidth, figHeight), tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight, top=marginTop, wspace = wSpace, hspace = hSpace)
bPlots = []
vPlots = []
for i in xrange(0, nParamsToPlot):
    plotData = dataToPlot[i]
    vp = ax[total5MinPlotIndex, i].violinplot([da for da in plotData], vPlotPos+1, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
    bp = ax[total5MinPlotIndex, i].boxplot([da for da in plotData], sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
    for s,scatterPlotData in enumerate(plotData):
        plotScatter(ax[total5MinPlotIndex, i], scatterPlotData, scatterX = s+1, scatterMarker = sMarkers[s], scatterColor = genotypeMarker[s], zOrder=2)
        ax[total5MinPlotIndex, i].hlines(np.median(scatterPlotData), s+1-medianWidth, s+1+medianWidth, colors=medianColor, alpha=0.8, zorder=4)
    vPlots.append(vp)
    bPlots.append(bp)
for vplot in vPlots:
    vplot[vPlotLineShow].set_color(medianColor)
    vplot[vPlotLineShow].set_zorder(4)
    for patch, color in zip(vplot['bodies'], colors):
        patch.set_color(color)
        patch.set_edgecolor(None)
        patch.set_alpha(vAlpha)
for i in xrange(0, len(axP)):
    for j in xrange(0, nParamsToPlot):
        ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                     fontsize=figLabelSize, transform=plt.gcf().transFigure)
        plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
        plt.setp(ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
        plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
        plt.setp(ax[i,j], ylabel = plotYLabels[j])
        plt.setp(ax[i,j], **axP[i][j])
        if i==tSeriesPlotIndex:
            plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
        elif i==total5MinPlotIndex:
            ax[i,j].set_title(plotTitles[j], y=1.08, loc='center')
plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((0,-2.25),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes')
legend = fig.legend(handles=legendHandles,labels=legendLabels, \
                  loc='lower center', edgecolor=(0,0,0), fontsize=8, ncol=len(genotypes),\
                   bbox_transform=plt.gcf().transFigure)





for c, gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    tPlots = []
    for i in xrange(0, nParamsToPlot):
        tp = ax[tSeriesPlotIndex,i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], \
               color=colors[c], fmt='-'+markers[c], label=genotypes[c], markersize=markerSize, linewidth = lineWidth)
        tPlots.append(tp)
legendHandles, legendLabels = ax[legendAxesRowGet, legendAxesColGet].get_legend_handles_labels()
bPlots = []
vPlots = []
for i in xrange(0, nParamsToPlot):
    plotData = dataToPlot[i]
    vp = ax[total5MinPlotIndex, i].violinplot([da for da in plotData], vPlotPos+1, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
    bp = ax[total5MinPlotIndex, i].boxplot([da for da in plotData], sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
    for s,scatterPlotData in enumerate(plotData):
        plotScatter(ax[total5MinPlotIndex, i], scatterPlotData, scatterX = s+1, scatterMarker = sMarkers[s], scatterColor = genotypeMarker[s], zOrder=2)
        ax[total5MinPlotIndex, i].hlines(np.median(scatterPlotData), s+1-medianWidth, s+1+medianWidth, colors=medianColor, alpha=0.8, zorder=4)
    vPlots.append(vp)
    bPlots.append(bp)
for vplot in vPlots:
    vplot[vPlotLineShow].set_color(medianColor)
    vplot[vPlotLineShow].set_zorder(4)
    for patch, color in zip(vplot['bodies'], colors):
        patch.set_color(color)
        patch.set_edgecolor(None)
        patch.set_alpha(vAlpha)
for i in xrange(0, len(axP)):
    for j in xrange(0, nParamsToPlot):
        ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                     fontsize=figLabelSize, transform=plt.gcf().transFigure)
        plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
        plt.setp(ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
        plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
        plt.setp(ax[i,j], ylabel = plotYLabels[j])
        plt.setp(ax[i,j], **axP[i][j])
        if i==tSeriesPlotIndex:
            plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
        elif i==total5MinPlotIndex:
            ax[i,j].set_title(plotTitles[j], y=1.08, loc='center')
plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((0,-2.25),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes')
legend = fig.legend(handles=legendHandles,labels=legendLabels, \
                  loc='lower center', edgecolor=(0,0,0), fontsize=8, ncol=len(genotypes),\
                   bbox_transform=plt.gcf().transFigure)
#legend.set_bbox_to_anchor((0.5, -0.06))
plt.savefig(combinedFigNameSvg, format='svg')
plt.savefig(combinedFigNamePdf, format='pdf')
plt.savefig(combinedFigNamePng, dpi=dpi, format='png')
# plt.show()







maxTimeThresh   = 300 # time in seconds for calculation of data from tracks under this much seconds
markerSize = 4.0       #4.0 for 5 minutes, 18 for 30 minutes
lineWidth = 0.95           #0.95 for 5 minutes plot
sWidth = 0.15           #0.012
sSize = 5              #5 for 5 minutes, 300 for 30 minutes

import numpy as np
disPerMin       = 4000
disTotal        = 20000
csTotalTracks   = 21
#----------- Set Values for setting X-Y limits on the plots -------
nTrackPerMin    = 6
nTrackTotal     = 35
nSecsDurPerMin  = 11
nSecsDurTotal   = 13
avSpeedPerMin   = 7 #7 for others, 5 for KCNJ10
avSpeedTotal    = 8 #8 for others, 4 for KCNJ10
disTotal        = 23000

#--- for CS, fig 2 -------
#nTrackPerMin    = 6
#nTrackTotal     = 25
#nSecsDurTotal   = 13
#avSpeedPerMin   = 7
#avSpeedTotal    = 8

##----------- for Park flies' data, fig 3 -------
#nTrackPerMin    = 5
#nSecsDurPerMin  = 7
#nTrackTotal     = 29
#nSecsDurTotal   = 9
#avSpeedPerMin   = 5
#avSpeedTotal    = 7

##----------- for Trp-Gamma data, fig 4 -------
#nTrackPerMin    = 6
#nTrackTotal     = 35
#nSecsDurPerMin  = 7
#nSecsDurTotal   = 9
#avSpeedPerMin   = 5
#avSpeedTotal    = 8
#disTotal        = 23000

##----------- for ALL flies' data, fig 5 -------
#nTrackPerMin    = 6
#nTrackTotal     = 35
#nSecsDurPerMin  = 7
#nSecsDurTotal   = 13
#avSpeedPerMin   = 7
#avSpeedTotal    = 8
#disTotal        = 23000




unitTime = 60   #seconds
nUnitTimes = maxTimeThresh/unitTime # number of minutes
figWidth = 1.6*nUnitTimes
figHeight = figWidth/1.618
fontSize = (8/7.0)*figWidth

nNumTracksTotalTicks    = 6
nDisYTicks              = 6
nSecsDurTotalTicks      = 6

ntSeriesXTicks          = 5

nNumTracksTotalStep = nTrackTotal/nNumTracksTotalTicks#5
nSecsDurTotalStep = nSecsDurTotal/nSecsDurTotalTicks#2

tSeriesXtickStep = nUnitTimes/ntSeriesXTicks

distickStep = 1000
disTotalTicks = disTotal/distickStep
disticks = disPerMin/distickStep
disTickScale = 1000
disTotalStep = (disTotal/(nDisYTicks*disTickScale))*disTickScale
disTotalTicksStep = disTotalStep/disTickScale
trackFPS = 35

#------ for fig 2------
ax00 = {'yticks': np.arange(0, nTrackPerMin) }
#ax10 = {'yticks': np.arange(0,36,5), 'ylim':(0,36)}
ax10 = {'yticks': np.arange(0,nTrackTotal,nNumTracksTotalStep), 'ylim':(0,nTrackTotal+1)}
ax01 = {'yticks': np.arange(0, trackFPS*nSecsDurPerMin, 2*trackFPS) , 'yticklabels':  np.arange(0,nSecsDurPerMin,2), 'ylim':(0,trackFPS*nSecsDurPerMin)}
ax11 = {'yticks': np.arange(0, trackFPS*nSecsDurTotal, trackFPS*nSecsDurTotalStep),'yticklabels':  np.arange(0,nSecsDurTotal,nSecsDurTotalStep), 'ylim':(0,trackFPS*nSecsDurTotal) }
ax02 = {'yticks': np.arange(0,disPerMin,distickStep), 'yticklabels': np.arange(disticks),'ylim': (0,disPerMin-distickStep/2)  }
ax12 = {'yticks': np.arange(0,disTotal,disTotalStep), 'yticklabels': np.arange(0,disTotalTicks,disTotalTicksStep), 'ylim':(0,disTotal) }
ax03 = {'yticks': np.arange(0,avSpeedPerMin,2),'ylim': (0,avSpeedPerMin)}
ax13 = {'yticks': np.arange(0,avSpeedTotal),'ylim': (0,avSpeedTotal)}
ax04 = {'ylim': (0, 1.1), 'yticks': [0, 0.5, 1], 'yticklabels': [0, 0.5, 1]}
ax14 = {'ylim': (0, 1.1), 'yticks': [0, 0.25, 0.5, 0.75, 1], 'yticklabels': [0, 0.25, 0.5, 0.75, 1]}
ax05 = {'ylim': (1.2, -1.5), 'yticks': [-1, 0, 1]}
ax15 = {'ylim': (1.2, -1.5), 'yticks': [-1, 0, 1]}

axP = [
        [ax10, ax11, ax12, ax13, ax14, ax15],
        [ax00, ax01, ax02, ax03, ax04, ax05]
      ]



pixelSize =0.055
param = 2.5# get selected parameter size in mm
blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)

AngBinMin = 0
AngBinMax = 180
AngBinStep = 1

chukFrames = 20 # number of frames to be chucked from start and end of the track to initiate data calculation
minTrackLen = blu*10


disMinThres = blu/20
disMaxThres = blu
consecWin = 7
trackLenThresh = 10*blu

legendHorPos = 0.18
legendVerPos = 1.058
figLabelSize = 12
figLabelXoffset = 0.03
xFactor = 0.2
yUp = 0.98
yDown = 0.3
figLabels = [["A","B","C","D","E"],["A'","B'","C'","D'","E'"]]
figLabelPositions = [[[i*xFactor,yUp] for i in xrange(5)], [[i*xFactor,yDown] for i in xrange(5)]]

gtiYUp = 0.91
gtiFigLabelXoffset = 0.128
gtiYDown = 0.29
gtiFigLabels = ["C","C'"]
gtiFigLabelPositions = [[gtiFigLabelXoffset,gtiYUp],[gtiFigLabelXoffset,gtiYDown]]


tSeriesPlotIndex = 1
total5MinPlotIndex = 0

nPlotStacks = 2
figRatio = [3,1]
tightLayout = False
wSpace = 0.4
hSpace = 0.15
marginLeft = 0.05
marginRight = 0.99
marginTop = 0.97
marginBottom = 0.082
medianWidth = 0.25

sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
scatterDataWidth = 0.012
sCol = (1,1,1)

legendHorPos = 0.25
legendVerPos = 1.058
legendAxesRowSet = total5MinPlotIndex
legendAxesRowGet = tSeriesPlotIndex
legendAxesColSet = 4
legendAxesColGet = 4



import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : (8/7.0)*figWidth}

plt.rc('font', **font)          # controls default text sizes
plt.rc('axes', labelsize=figWidth*3)    # fontsize of the x and y labels
col=1
plt.rcParams['axes.facecolor'] = (col,col,col)

plt.rc('font', family='serif', serif='Arial', size=fontSize)
plt.rc('ytick', labelsize=fontSize)
plt.rc('axes', labelsize=fontSize)
plt.rc('xtick', labelsize=fontSize)


angleBins = np.arange(AngBinMin, AngBinMax, AngBinStep)


def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

speedBinMin = disMinThres
speedBinMax = disMaxThres
speedBinStep = 0.1
speedBins = np.arange(speedBinMin, speedBinMax, speedBinStep)


if  'W1118' in genotypes:
    if 'CS' in genotypes:
        csIndex = genotypes.index('CS')
        w1118Index = genotypes.index('W1118')
        genotypes.pop(csIndex)
        genotypes.insert(0, 'CS')
        genotypes.pop(w1118Index)
        genotypes.insert(1, 'W1118')
    else:
        w1118Index = genotypes.index('W1118')
        genotypes.pop(w1118Index)
        genotypes.insert(0, 'W1118')
        
saveDir = baseDir+'_'
saveFiles = ''
for _,d in enumerate(dirs):
    saveFiles+='_'+d
saveFiles




alfa = 0.71
div = 255.0

colors_ = [(0/div,0/div,0/div,alfa),#gray
             (200/div,129/div,0/div,alfa),#orange
             (86/div,180/div,233/div,alfa),#Light blue
             (204/div,121/div,167/div,alfa),#pink
             (0/div,158/div,115/div,alfa),#greenish
             (0/div,114/div,178/div,alfa),#blue
             (213/div,94/div,0/div,alfa),#orange
             (240/div,228/div,66/div,alfa),#yellow
             (220/div,198/div,66/div,alfa)#dark yellowish
             ]

markers = ['^','s','v','d','o', 'P']
colors = [(230/div,218/div,66/div,alfa),(0/div,114/div,178/div,alfa)]   # for 30 minutes, KCNJ10 data

#---------declare the proper genotypes, markers and colors for the genotypes!!!!------------
genotypes = []
colors = []
markers = []
for i, gt in enumerate(dirs):
    if gt in ('CS', 'cs'):
        genotypes.append(gt)
        colors.append((0/div,0/div,0/div,alfa))
        markers.append('^')
    elif gt in ('CS_males', 'cs'):
        genotypes.append(gt)
        colors.append((0/div,0/div,0/div,alfa))
        markers.append('^')
    elif gt in ('CS_females', 'cs'):
        genotypes.append(gt)
        colors.append((0/div,0/div,0/div,alfa))
        markers.append('^')
    elif gt in ('W1118', 'w1118'):
        genotypes.append(r'W$^1$$^1$$^1$$^8$')
        colors.append((230/div,218/div,66/div,alfa))
        markers.append('P')
    elif gt in ('Trp-Gamma',  'trp'):
        genotypes.append(r'Trp-$\gamma$')
        colors.append((0/div,158/div,115/div,alfa))
        markers.append('o')
    elif gt in ('Park_+',  'PARK_+'):
        genotypes.append(r'Park$^2$$^5$/+')
        colors.append((70/div,0/div,10/div,alfa))
        markers.append('o')
    elif gt in ('1_PINK1RV', 'pink1rv'):
        genotypes.append(r'PINK1$^R$$^V$')
        colors.append((204/div,121/div,167/div,alfa))
        markers.append('d')
    elif gt in ('PARK25_TM3', 'Park25_TM3'):
        genotypes.append(r'Park$^2$$^5$/TM3')
        colors.append((86/div,180/div,233/div,alfa))
        markers.append('v')
    elif gt in ('PARK25xLrrk-ex1', '4_Park25xLrrk-ex1'):
        genotypes.append(r'Park$^2$$^5$/Lrrk$^e$$^x$$^1$')
        colors.append((86/div,180/div,233/div,alfa))
        markers.append('s')
    elif gt in ('W1118xLrrk-ex1', 'w1118xLrrk-ex1','3_Lrrk-ex1xW1118' ):
        genotypes.append(r'Lrrk$^e$$^x$$^1$/W$^1$$^1$$^1$$^8$')
        colors.append((180/div,109/div,0/div,alfa))
        markers.append('v')
    elif gt in ('2_PARK25xW1118', 'Park25xw1118', '2_Park25xW1118'):
        genotypes.append(r'Park$^2$$^5$/W$^1$$^1$$^1$$^8$')
        colors.append((70/div,0/div,10/div,alfa))
        markers.append('v')
    elif gt in ('Dop2R', 'dop2r'):
        genotypes.append(gt)
        colors.append((180/div,109/div,0/div,alfa))
        markers.append('s')
    else:
        genotypes.append(gt)
        colors.append(random.choice(colorsRandom))
        markers.append('8')
    print i, gt, len(colors), colors



sMarkers = markers




plotTitles = ['Number of Tracks',
              'Track Duration',
              'Distance Traveled',
              'Average Speed',
              'Straightness',
              'Geotactic Index'
             ]

plotYLabels = ['number',
               'seconds',
               r'BLU (x10$^3$)',
               'BLU/S',
               r'R$^2$ Value',
               'Geotactic Index',
                ]

vPlotPos = np.arange(len(genotypes))

def plotScatter(axis, data, scatterX, scatterWidth = sWidth, \
                scatterRadius = sSize , scatterColor = sCol,\
                scatterMarker = sMarker, scatterAlpha = sAlpha, \
                scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol, zOrder=0):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot
    '''
    return axis.scatter(np.linspace(scatterWidth+scatterX, -scatterWidth+scatterX,len(data)), data,\
            s=scatterRadius, color = scatterColor, marker=scatterMarker,\
            alpha=scatterAlpha, linewidths=scatterLineWidth, edgecolors=scatterEdgeColor, zorder=zOrder )


#---get the per unit time data ----

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)+0))
    ax.set_xticklabels(labels)
    ax.set_xlim(-1, len(labels))




bAlpha = 0.5
vAlpha = 0.5
vAlphaCS = 0.5

nParamsToPlot = nParams-1

dataToPlot = [genotypeNTracks,
              genotypeLenTrack,
              genotypeDis,
              genotypeAvSpeed,
              genotypeStraight,
              genotypeGeoTacInd]



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


ptime = present_time()
figDir = baseDir+'../'
csFigNamePng = ('%s/png/%s_CS.png'%(figDir, ptime))
csFigNamePdf = ('%s/%s_CS.pdf'%(figDir, ptime))
combinedFigNamePng = ('%s/png/%s_%s.png'%(figDir, ptime, '_'.join(dirs)))
combinedFigNamePdf = ('%s/%s_%s.pdf'%(figDir, ptime, '_'.join(dirs)))
csFigNameSvg = ('%s/%s_CS.svg'%(figDir, ptime))
combinedFigNameSvg = ('%s/%s_%s.svg'%(figDir, ptime, '_'.join(dirs)))
gtiFigNamePng = ('%s/png/%s_%s_GTI.png'%(figDir, ptime, '_'.join(dirs)))
gtiFigNamePdf = ('%s/%s_%s_GTI.pdf'%(figDir, ptime, '_'.join(dirs)))
gtiFigNameSvg = ('%s/%s_%s_GTI.svg'%(figDir, ptime, '_'.join(dirs)))
dpi = 300

sMarkers  = ['o' for x in sMarkers]

if 'CS' in dirs:
    csIndex = dirs.index('CS')
    csGT = allGenotypePerUT_Data[csIndex]
    data = np.nanmean(csGT[:], axis=0)
    sem = stats.sem(csGT[:], axis=0)
    vPlotPosCS = [csIndex+1]
    ax10 = {'yticks': np.arange(0,csTotalTracks,4), 'ylim':(0,csTotalTracks)}
    ax14 = {'ylim': (0, 1.2), 'yticks': [0, 0.25, 0.5, 0.75, 1], 'yticklabels': [0, 0.25, 0.5, 0.75, 1]}
    axP[0][0]=ax10
    axP[0][4]=ax14
    fig, ax = plt.subplots(nPlotStacks,nParamsToPlot, figsize=(figWidth, figHeight), tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
    fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight, top=marginTop, wspace = wSpace, hspace = hSpace)
    for i in xrange(nParamsToPlot):
            ax[tSeriesPlotIndex, i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], \
              color=colors[0], fmt='-'+markers[0], markersize=markerSize, linewidth = lineWidth)
    bPlots = []
    vPlots = []
    for i in xrange(nParamsToPlot):
        plotData = dataToPlot[i][csIndex]
        vp = ax[total5MinPlotIndex, i].violinplot(plotData, vPlotPosCS, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
        bp = ax[total5MinPlotIndex, i].boxplot(plotData, sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
        plotScatter(ax[total5MinPlotIndex, i], plotData, scatterX = vPlotPosCS[0], scatterMarker = sMarkers[csIndex], scatterColor = genotypeMarker[csIndex], zOrder=2)
        ax[total5MinPlotIndex, i].hlines(np.median(plotData), vPlotPosCS[0]-medianWidth, vPlotPosCS[0]+medianWidth, colors=medianColor, alpha=0.8, zorder=4)
        vPlots.append(vp)
        bPlots.append(bp)
    for vplot in vPlots:
        vplot[vPlotLineShow].set_color(medianColor)
        for patch, color in zip(vplot['bodies'], colors):
            patch.set_color(color)
            patch.set_edgecolor(None)
            patch.set_alpha(vAlphaCS)
    for i in xrange(len(axP)):
        for j in xrange(nParamsToPlot):
            ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                         fontsize=figLabelSize, transform=plt.gcf().transFigure)
            plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
            plt.setp(ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
            plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
            plt.setp(ax[i,j], ylabel = plotYLabels[j])
            plt.setp(ax[i,j], **axP[i][j])
            if i==tSeriesPlotIndex:
                plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes), xticklabels = np.arange(1,nUnitTimes+1))
            elif i==total5MinPlotIndex:
                ax[i,j].set_title(plotTitles[j], y=1.08, loc='center')
    plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,2], xticks = [0], xticklabels = [])
    ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((0,-2.25),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
    plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes')
    plt.savefig(csFigNamePng, dpi=dpi, format='png')
    plt.savefig(csFigNamePdf, format='pdf')
    plt.savefig(csFigNameSvg, format='svg')
    # plt.show()

ax10 = {'yticks': np.arange(0,nTrackTotal,nNumTracksTotalStep), 'ylim':(0,nTrackTotal+1)}
axP[0][0]=ax10

fig, ax = plt.subplots(nPlotStacks,nParamsToPlot, figsize=(figWidth, figHeight), tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
fig.subplots_adjust(left=marginLeft, bottom=marginBottom, right=marginRight, top=marginTop, wspace = wSpace, hspace = hSpace)
for c, gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    tPlots = []
    for i in xrange(0, nParamsToPlot):
        tp = ax[tSeriesPlotIndex,i].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i], \
               color=colors[c], fmt='-'+markers[c], label=genotypes[c], markersize=markerSize, linewidth = lineWidth)
        tPlots.append(tp)
legendHandles, legendLabels = ax[legendAxesRowGet, legendAxesColGet].get_legend_handles_labels()
bPlots = []
vPlots = []
for i in xrange(0, nParamsToPlot):
    plotData = dataToPlot[i]
    vp = ax[total5MinPlotIndex, i].violinplot([da for da in plotData], vPlotPos+1, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
    bp = ax[total5MinPlotIndex, i].boxplot([da for da in plotData], sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
    for s,scatterPlotData in enumerate(plotData):
        plotScatter(ax[total5MinPlotIndex, i], scatterPlotData, scatterX = s+1, scatterMarker = sMarkers[s], scatterColor = genotypeMarker[s], zOrder=2)
        ax[total5MinPlotIndex, i].hlines(np.median(scatterPlotData), s+1-medianWidth, s+1+medianWidth, colors=medianColor, alpha=0.8, zorder=4)
    vPlots.append(vp)
    bPlots.append(bp)
for vplot in vPlots:
    vplot[vPlotLineShow].set_color(medianColor)
    vplot[vPlotLineShow].set_zorder(4)
    for patch, color in zip(vplot['bodies'], colors):
        patch.set_color(color)
        patch.set_edgecolor(None)
        patch.set_alpha(vAlpha)
for i in xrange(0, len(axP)):
    for j in xrange(0, nParamsToPlot):
        ax[i,j].text(figLabelPositions[i][j][0]+figLabelXoffset, figLabelPositions[i][j][1], figLabels[i][j],\
                     fontsize=figLabelSize, transform=plt.gcf().transFigure)
        plt.setp([ax[i,j].spines[x].set_visible(False) for x in ['top','right']])
        plt.setp(ax[i,j].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
        plt.setp(ax[i, j].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
        plt.setp(ax[i,j], ylabel = plotYLabels[j])
        plt.setp(ax[i,j], **axP[i][j])
        if i==tSeriesPlotIndex:
            plt.setp(ax[i,j], xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
        elif i==total5MinPlotIndex:
            ax[i,j].set_title(plotTitles[j], y=1.08, loc='center')
plt.setp([axs for axs in ax[total5MinPlotIndex, :]], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
ax[tSeriesPlotIndex,total5MinPlotIndex].add_patch(plt.Rectangle((0,-2.25),29, 0.85,facecolor='0.9',clip_on=False,linewidth = 0))
plt.setp(ax[tSeriesPlotIndex,nParamsToPlot/2], xlabel = 'minutes')
legend = fig.legend(handles=legendHandles,labels=legendLabels, \
                  loc='lower center', edgecolor=(0,0,0), fontsize=8, ncol=len(genotypes),\
                   bbox_transform=plt.gcf().transFigure)
#legend.set_bbox_to_anchor((0.5, -0.06))
plt.savefig(combinedFigNameSvg, format='svg')
plt.savefig(combinedFigNamePdf, format='pdf')
plt.savefig(combinedFigNamePng, dpi=dpi, format='png')
# plt.show()

# produce a legend for the objects in the other figure
legend_fig = plt.figure()
legend = plt.figlegend(*fig.gca().get_legend_handles_labels(), edgecolor=(0,0,0), fontsize=8,  ncol=len(genotypes))
legend_fig.savefig(combinedFigNameSvg+'_legend.pdf', format='pdf')

gtinParamsToPlot = 1
gtiFigWidth = 2.2
gtiFigHeight = figHeight+0.5

gtiMarginLeft = 0.2
gtiMarginRight = marginRight
gtiMarginTop = marginTop-0.07
gtiMarginBottom  = marginBottom + 0.01
gtilegendVerPos = legendVerPos+0.1
ax0 = {'yticks': np.arange(-1, 2), 'ylim':(1.2, -1.2) }
axP1 = [ax0, ax0]


#-----GeoTacticIndex Plot-------

fig, ax = plt.subplots(nPlotStacks, gtinParamsToPlot, figsize=(1.8, gtiFigHeight), tight_layout = tightLayout, gridspec_kw = {'height_ratios':figRatio})
fig.subplots_adjust(left=gtiMarginLeft, bottom=gtiMarginBottom, right=gtiMarginRight, top=gtiMarginTop, wspace = wSpace, hspace = hSpace)
for c, gt in enumerate(allGenotypePerUT_Data):
    data = np.nanmean(gt[:], axis=0)
    sem = stats.sem(gt[:], axis=0)
    tPlots = []
    i=-1
    tp = ax[tSeriesPlotIndex].errorbar(np.arange(len(data[:,i])), data[:,i], yerr=sem[:,i],\
           color=colors[c], fmt='-'+markers[c], label=genotypes[c], markersize=markerSize, linewidth = lineWidth)
legendHandles, legendLabels = ax[tSeriesPlotIndex].get_legend_handles_labels()
plotData = dataToPlot[i]
vp = ax[total5MinPlotIndex].violinplot([da for da in plotData], vPlotPos+1, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
bp = ax[total5MinPlotIndex].boxplot([da for da in plotData], sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
for s,scatterPlotData in enumerate(plotData):
    plotScatter(ax[total5MinPlotIndex], scatterPlotData, scatterX = s+1, scatterMarker = sMarkers[s], scatterColor = genotypeMarker[s], zOrder=2)
    ax[total5MinPlotIndex].hlines(np.median(scatterPlotData), s+1-medianWidth, s+1+medianWidth, colors=medianColor, alpha=0.8, zorder=4)
vp[vPlotLineShow].set_color(medianColor)
for patch, color in zip(vp['bodies'], colors):
    patch.set_color(color)
    patch.set_edgecolor(None)
    patch.set_alpha(vAlpha)

for i in xrange(0, len(axP1)):
    ax[i].text(gtiFigLabelPositions[i][0], gtiFigLabelPositions[i][1], gtiFigLabels[i],\
                 fontsize=figLabelSize, transform=plt.gcf().transFigure)
    plt.setp([ax[i].spines[x].set_visible(False) for x in ['top','right']])
    plt.setp(ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5))
    plt.setp(ax[i].get_yticklabels(), rotation=90, horizontalalignment='center', verticalalignment='center')
    plt.setp(ax[i], ylabel = plotYLabels[-1])
    plt.setp(ax[i], **axP1[i])
    if i==tSeriesPlotIndex:
        plt.setp(ax[i],  xticks = np.arange(0,nUnitTimes, tSeriesXtickStep), xticklabels = np.arange(1,nUnitTimes+1, tSeriesXtickStep))
    ax[tSeriesPlotIndex,].set_xlabel('minutes')
    legend = fig.legend(handles=legendHandles,labels=legendLabels, bbox_to_anchor=(0.5, -0.06),\
                      loc='lower center',edgecolor=(0,0,0), fontsize=6, ncol=len(genotypes),\
                       bbox_transform=plt.gcf().transFigure)
    plt.setp(ax[total5MinPlotIndex], xlim=[0,len(genotypes)+1], xticks = [0], xticklabels = [])
ax[tSeriesPlotIndex].add_patch(plt.Rectangle((0,1.80),4, 0.40,facecolor='0.9',clip_on=False,linewidth = 0))

plt.savefig(gtiFigNamePng, dpi=dpi, format='png')
plt.savefig(gtiFigNamePdf, format='pdf')
plt.savefig(gtiFigNameSvg, format='svg')
# plt.show()


















sWidth = 0.5
vPlots = []





for i in xrange(len(pltTotalData[genotypes[0]])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(genotypes):
        colorSex = [x[colIdPooledDict['sexColor']] for i_,x in enumerate(pooledTotalData[gtype])]
        scPlt1 = bf.plotScatterCentrd(ax,pltTotalData[gtype][i], g, \
                                      scatterRadius=10, scatterColor=colorSex, \
                                      scatterEdgeColor=(1,1,1),scatterAlpha=0.65, \
                                      scatterWidth = sWidth)
        vp = plt.violinplot(pltTotalData[gtype][i], [g], showextrema=False)
        vPlots.append(vp)
    plt.xlim(-1,len(genotypes))
    plt.title(str(i)+'_'+pltParamList[i])
plt.show()

#------test plots
#---- Plot the timeSEries data from behaviour from total time measured ----#
for i in xrange(len(pltTmSrsData[genotypes[0]])):
    fig, ax = plt.subplots()
    for g, gtype in enumerate(genotypes):
        pltData, pltDataErr = pltTmSrsData[gtype][i]
        ax.errorbar(np.arange(len(pltData)), pltData, pltDataErr, alpha=0.7)
        plt.title(str(i)+'_'+pltParamList[i])
plt.show()

#============================================================================================================


































"""