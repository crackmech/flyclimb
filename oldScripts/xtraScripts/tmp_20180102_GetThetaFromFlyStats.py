# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:32:45 2018

@author: aman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:58:29 2017

@author: aman
"""

imgDatafolder = 'imageData'
trackImExtension = '.jpeg'
csvExt = '.csv'
headers = ['area_average(mm^2)', 'minorAxis_average(mm)', 'majorAxis_average(mm)',\
            'area_median(mm^2)', 'minorAxis_median(mm)', 'majorAxis_median(mm)' ,\
            'nFrames', 'FPS', 'folderName']

selectedParameter = 'minorAxis_median(mm)'
statsfName = 'flyStats_'
blankLinesAfterParam = 2
blankLinesBeforeParam = 1
startString = 'Parameters'
selParamIndex = headers.index(selectedParameter)
pixelSize =0.055
param = 2.5# get selected parameter size in mm
blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)

AngBinMin = 0
AngBinMax = 180
AngBinStep = 1



import cv2
import os
import numpy as np
import re
import sys
from datetime import datetime
from thread import start_new_thread as startNT
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
import time
import glob
import random
from math import atan2, degrees
import copy

angleBins = np.arange(AngBinMin, AngBinMax, AngBinStep)


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def getStringLineIndex(fname, string):
    '''
    returns the list of line numbers containing parameterString (Parameters)
    '''
    stringIndex = []
    with open(fname) as f:
        for num, line in enumerate(f, 1):
            if string in line:
                stringIndex.append(num)
    if stringIndex !=[]:
        print('Found parameters')
        return stringIndex
    else:
        print('No parameters found')

def getAllStats(fname, header, blanksAfterParams, blanksBeforeParams, startString):
    '''
    Get a list of all stats from Stats csv file
    returns:
        1) average stats of a single fly with STD
        2) a list of lists containing all stats read from the Stats csv file
            
    '''
    print fname
    parInd = getStringLineIndex(fname, startString) #get a list of indices with paramters line
    parInd.reverse()
    with open(fname) as f:
        lines = f.readlines()
    nLines = len(lines)
    statslen = len(header[:-1]) #remove the folder name to calculate stats
    allStats = []
    allStats.append(('parameter Details',header))
    for i in xrange(len(parInd)):
        startIndex = parInd[i]+blanksAfterParams
        if parInd[i]!=(nLines-1):
            if i==0:
                stopIndex = nLines
            else:
                stopIndex = parInd[i-1]-blanksBeforeParams-1
            stats = np.zeros((stopIndex-startIndex,statslen))
            for line in xrange(startIndex,stopIndex):
                lineSplit = (lines[line]).split(',')[:-1]
                nan = [True for x in lineSplit if 'nan' in x]
                none = [True for x in lineSplit if 'None' in x]
                if nan==[] and none==[]:
                    stats[line-startIndex,:] = lineSplit
                else:
                    stats[line-startIndex,:] = [np.nan for x in lineSplit]
            avStats = np.zeros((2, statslen))
            avStats[0,:] = np.nanmean(stats, axis=0)
            avStats[1,:] = np.nanstd(stats, axis=0)
            allStats.append((lines[parInd[i]-1],stats, avStats))
        else:
            break
    return allStats




def calcAngle3Pts(a, b, c):
    '''
    returns angle between a and c with b as the vertex 
    '''
    ba = a.flatten() - b.flatten()
    bc = c.flatten() - b.flatten()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def getData(basedir, foldername, imDatafolder, trackImExt, csvExt):
    '''
    returns the arrays containing various datasets from a folder containing data of a single fly
    '''
    d = os.path.join(basedir, foldername, imDatafolder)
    csvlist = natural_sort(glob.glob(d+'/*'+csvExt))
    imglist = natural_sort(glob.glob(d+'/*'+trackImExt))
    data = []
    for _, ids in enumerate(zip(csvlist, imglist)):
        csvData = np.genfromtxt(ids[0], dtype='float',delimiter = ',', skip_header=1)
        data.append([csvData, ids[0], ids[1]])
    return data

def getConsecData(csvData, consecStep, eudDisMinThresh, eudDisMaxThresh, fps, bodyLen):
    '''
    Input: list of lists of 
        a) csvdata 
        b) CSV file and Image file path details
    
    returns a list containing 
        a) X coordinate
        b) Y coordinate
        c) Angle b/w  i-consecStep, i, i+consecStep 
        d) Eucledian distance between i,i+consecStep
        
    CSV file and Image file path details
    '''
    allangles = []
    consecStep = int(consecStep)
    for _, data in enumerate(csvData):
        csvdata = data[0]
        angles = [] 
        for i in xrange(consecStep, len(csvdata)-consecStep-1, consecStep):
            p0 = csvdata[i-consecStep]
            p1 = csvdata[i]
            p2 = csvdata[i+consecStep]
            euDis = np.linalg.norm(csvdata[i+consecStep]-csvdata[i])
            speed = (euDis*fps)/(consecStep*bodyLen)
            angle = (calcAngle3Pts(p0,p1,p2))
            if eudDisMinThresh<euDis<eudDisMaxThresh:
                angles.append(np.array([csvdata[i][0], csvdata[i][1], angle, euDis, speed]))
        allangles.append((np.array(angles), data[1], data[2]))
    return allangles

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

def plotThetaDis(csvAngData, tracklenthresh, c, angs, consecStep, bodyLen):
    f, axarr = plt.subplots(4,1)
    im = np.ones((450, 1280, 3), dtype='uint8')*255
    for i, ids in enumerate(csvAngData):
        if np.sum(ids[:,3])>tracklenthresh:
            x = len(ids)
            #c[i] = (0,0,0)
            axarr[0].plot(np.arange(len(angs[i])), angs[i][:, -1], '. ', color = c[i], alpha = 0.2)
            axarr[1].plot(np.arange(len(angs[i])), angs[i][:, 0],'.',color = c[i], alpha = 0.2)
            axarr[2].plot(angs[i][:, -1], angs[i][:, 0], '.' ,color = colors[i], alpha = 0.1)
            for p in xrange(x):
                cv2.circle(im, (int(ids[p,0:2][0]), int(ids[p,0:2][1])), 2, [col*255 for col in c[i]], -1)
    axarr[3].imshow(im)
    axarr[0].set_ylabel('speed\n(Body length units per second)')
    axarr[1].set_ylabel('angle')
    axarr[0].set_xlabel('Body Length Units')
    axarr[1].set_xlabel('Body Length Units')
    axarr[1].set_ylim(200,-10)
    labels = [item.get_text() for item in axarr[1].get_xticklabels()]
    labels = np.array(np.linspace(0, (axarr[1].get_xlim()[-1]*consecStep)/blu, len(labels)), dtype='uint16')
    axarr[0].set_xticklabels(labels)
    axarr[1].set_xticklabels(labels)
    axarr[0].set_ylim(-1,30)
    plt.show()


def getHistMode(array, bins):
    '''
    Calculates the mode value based on the histogram of a given array
    returns: the upper and lower limit of the most common value in the array
    '''
    a = array.copy()
    freq, edgs  = np.histogram(a[~np.isnan(a)],bins)
    maxValMin = edgs[np.argmax(freq)]
    maxValMax = maxValMin+np.max(np.diff(bins))
    return maxValMin, maxValMax


def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]
      
def getFlyStats(genotypeData, consecFrameStep, minDisThres, maxDisThres):
    '''Calculates statistics of data from a genotype (folder with 'n' flies)
    
    input: 
        genotypeData:       a list of lists containing trackCSVData and fly TrackStats for each fly in the genotype
        consecFrameStep:    step size of consecutive frames to calculate 
                            a) Angle of fly calculated between (currentFrame-consecFrameStep), currentFrame, (currentFrame+consecFrameStep)
                            b) Distance covered by the fly between (currentFrame) and (currentFrame+consecFrameStep)
                            
        minDisThres:      minimum distance moved by a fly between (currentFrame) and (currentFrame+consecFrameStep)
        maxDisThres:      maximum distance moved by a fly between (currentFrame) and (currentFrame+consecFrameStep)
    
    returns:
        genotypeStats:    a list of all stats of a genotype
                            each list element contains data from all tracks of a fly, with:
                                meanAngle, medianAngle, stdAngle, meanInstantSpeed, medianInstanSpeed, stdInstantSpeed,\
                                meanInstantSpeed per BodyLengthUnit, medianInstanSpeed per BodyLengthUnit, stdInstantSpeed per BodyLengthUnit
        angles:            a list of arrays of all tracks of the each fly completed to the length of the longest track, shorter track data padded by np.nan
        angsData:          a list of lists of all trackData calculated by given parameters
    '''
    genotypeStats = []
    angles = []
    angsData = []
    for i,d in enumerate(genotypeData):
        csvData, allStats = d
        fps = allStats[1][-1][0,-1]
        pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])
        param = allStats[1][-1][0,selParamIndex]# get selected parameter size in mm
        blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)
        print blu, fps, pixelSize
        angData = getConsecData(csvData,consecFrameStep, minDisThres, maxDisThres, fps, blu)
        #print len(angData), len(angData[0])
        maxTrackLen = max([len(x[0]) for x in angData])
        angs = np.zeros((len(angData),maxTrackLen,3))# array of angle, distance and speeds for each track of a fly
        flyStats = np.zeros((maxTrackLen,9))# array of median angle, median distance and median speed and their STDEV for each fly
        angs[:] = np.nan
        for i,ang in enumerate(angData):
            for j in xrange(len(ang[0])):
                angs[i,j,0] = ang[0][j,2]# copy angles
                angs[i,j,1] = ang[0][j,3]# copy distance
                angs[i,j,2] = ang[0][j,4]# copy speeds
        for i in xrange(0, len(flyStats[0]),3):
            data = angs[:,:,i/3]
            flyStats[:,i] = np.nanmean(data, axis=0)
            flyStats[:,i+1] = getMedian(data, i)
            flyStats[:,i+2] = np.nanstd(data, axis=0)
        genotypeStats.append(flyStats)
        angles.append(angs)
        angsData.append(angData)
    return genotypeStats, angles, angsData


colors = [random_color() for c in xrange(1000)]


def plotHist(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        ax.hist(ids[:, 6], bins, alpha=0.2)
    return ax

def plots(allflystats):
    marker = '.'
    marker2 = '.'
    maxAngle = 200
    minAngle = 0
    f, axarr = plt.subplots(4,2)
    for i, ids in enumerate(allflystats):
        axarr[0,0].plot(np.arange(len(ids)), ids[:, 0], marker, alpha = 0.2)# plot mean angles
        axarr[1,0].plot(np.arange(len(ids)), ids[:, 3], marker, alpha = 0.2) # plot mean distance
        axarr[2,0].plot(np.arange(len(ids)), ids[:, 6], marker, alpha = 0.2) # plot mean speeds
        axarr[3,0].plot(ids[:, 6], ids[:, 0], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        
        axarr[0,1].plot(np.arange(len(ids)), ids[:, 1], marker, alpha = 0.2) # plot median angle
        axarr[1,1].plot(np.arange(len(ids)), ids[:, 4], marker, alpha = 0.2) # plot mean instaneous velocities
        axarr[2,1].plot(np.arange(len(ids)), ids[:, 7], marker, alpha = 0.2) # plot median speed
        axarr[3,1].plot(ids[:, 7], ids[:, 1], marker2, alpha = 0.1) # plot median angle vs. median speed
        
        axarr[1,0].set_ylim(axarr[1,1].get_ylim())
        axarr[2,0].set_ylim(axarr[2,1].get_ylim())
        axarr[3,0].set_xlim(axarr[3,1].get_xlim())
        axarr[3,0].set_ylim(minAngle,maxAngle)
        axarr[3,1].set_ylim(0,200)
        axarr[0,0].set_ylim(minAngle,maxAngle)
        axarr[0,1].set_ylim(0,200)
        axarr[0,1].set_xlim(axarr[1,1].get_xlim())
    plt.show()
    f, axarr = plt.subplots(4,2)
    for i, ids in enumerate(allflystats):
        axarr[0,0].plot(np.arange(len(ids)), ids[:, 0], marker, alpha = 0.2)# plot mean angles
        axarr[1,0].plot(ids[:, 3], ids[:, 4], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        axarr[2,0].plot(ids[:, 6], ids[:, 7], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        axarr[3,0].plot(ids[:, 6], ids[:, 0], marker2, alpha = 0.1) #plot mean angle vs. mean speed
        
        axarr[0,1].plot(np.arange(len(ids)), ids[:, 1], marker, alpha = 0.2) # plot median angle
        axarr[1,1].plot(np.arange(len(ids)), ids[:, 3], marker, alpha = 0.2) # plot mean instaneous velocities
        axarr[2,1].plot(np.arange(len(ids)), ids[:, 4], marker, alpha = 0.2) # plot median speed
        axarr[3,1].plot(ids[:, 7], ids[:, 1], marker2, alpha = 0.1) # plot median angle vs. median speed
        
        axarr[1,0].set_ylim(axarr[1,1].get_ylim())
        axarr[2,0].set_ylim(axarr[2,1].get_ylim())
        axarr[3,0].set_xlim(axarr[3,1].get_xlim())
        axarr[3,0].set_ylim(minAngle,maxAngle)
        axarr[3,1].set_ylim(0,200)
        axarr[0,0].set_ylim(minAngle,maxAngle)
        axarr[0,1].set_ylim(0,200)
        axarr[0,1].set_xlim(axarr[1,1].get_xlim())
    plt.show()




def getMedian(dataArray, i):
    '''
    returns the "median" value of dataArray
        median is calculated by the function needed to replace the np.median for the dataArray
    '''
    med = np.zeros((len(dataArray[0])))
    med = np.median(dataArray, axis=0)
    #return med
    if i==0:
        bins = angleBins
    else:
        bins = speedBins
    for j in xrange(len(med)):
        med[j] = getHistMode(dataArray[:,j], bins)[0]
    return med


disMinThres = blu/20
disMaxThres = blu
consecWin = 5
trackLenThresh = 10*blu

speedBinMin = disMinThres
speedBinMax = disMaxThres
speedBinStep = 0.1
speedBins = np.arange(speedBinMin, speedBinMax, speedBinStep)

baseDir = '/media/aman/data/flyWalk_data/20180104'
#baseDir = '/media/pointgrey/data/flywalk/20180104/'

baseDir = getFolder(baseDir)
print "Started processing directories at "+present_time()

#rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
#cs = []
#for rawDir in rawdirs:
#    print'=====raw'
#    csvData = getData(baseDir, rawDir, imgDatafolder, trackImExtension, csvExt)
#    fname = glob.glob(os.path.join(baseDir, rawDir,statsfName+'*'))[0]
#    trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
#    cs.append([csvData, trackStats])
#
#allFlyStats = getFlyStats(cs, consecWin, disMinThres, disMaxThres)[0]
#save_object(allFlyStats, baseDir.rstrip('/')+'.pkl')




def getAllFlyStats(genotypeDir):
    '''
    returns allFlyStats for a all fly folders in a given folder (genotypeDir)
    
    '''
    rawdirs = natural_sort([ name for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    cs = []
    for rawDir in rawdirs:
        print'=====raw'
        csvData = getData(genotypeDir, rawDir, imgDatafolder, trackImExtension, csvExt)
        fname = glob.glob(os.path.join(genotypeDir, rawDir,statsfName+'*'))[0]
        trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
        cs.append([csvData, trackStats])
    
    return getFlyStats(cs, consecWin, disMinThres, disMaxThres)

allFlyStats = []

dirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])


for d in dirs:
    path = os.path.join(baseDir, d)
    #flyStats = getAllFlyStats(os.path.join(baseDir, dirs[1]))
    allFlyStats.append([path, getAllFlyStats(path)])

'''
fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allFlyStats):
    a = da[1][0]
    for i in xrange(len(a)):
        ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(a[i])), a[i][:,6],\
                    s=1, alpha=0.01, linewidths=1, edgecolors=(0,0,1) )

plt.show()

fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allFlyStats):
    a = da[1][0]
    for i in xrange(len(a)):
        ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(a[i])), a[i][:,6],\
                    s=2, alpha=0.1, linewidths=1, edgecolors=(0,0,1) )

plt.show()
'''
allInsSpeeds = []

for i, d in enumerate(allFlyStats):
    a = None
    for x, ids in enumerate(d[1][0]):
        if x==0:
            a = ids
        else:
            a = np.vstack((a, ids))
    allInsSpeeds.append((x+1,a))

from scipy import stats
krus = stats.kruskal(allInsSpeeds[0][1][:,6],
              allInsSpeeds[1][1][:,6],
              allInsSpeeds[2][1][:,6],
              allInsSpeeds[3][1][:,6],
              allInsSpeeds[4][1][:,6]
              )
print krus


print stats.f_oneway(
              allInsSpeeds[0][1][:,6],
              allInsSpeeds[1][1][:,6],
              allInsSpeeds[2][1][:,6],
              allInsSpeeds[3][1][:,6],
              allInsSpeeds[4][1][:,6]
              )


fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allInsSpeeds):
    try:
        ax.boxplot(da[1][:,6], positions=[x+0.125])
    except:
        pass
    ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(da[1])), da[1][:,6],\
                s=2, color = (0.25,0,0), alpha=0.11, linewidths=1, edgecolors=(0,0,1) )
plt.ylim(0,35)
plt.xlim(-1,5)
plt.show()


binMax = 20
bins = np.linspace(0, 20, 10*binMax)

def plotPlot(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        freq, edges = np.histogram(ids[:,6], bins)
        ax.plot(freq,alpha=0.5)
    return ax

aa = allFlyStats[0][1][0]
f, ax = plt.subplots(nrows=len(aa))
plotPlot(aa, bins, ax)
plt.show()


a = [x[1][:,6] for x in allInsSpeeds]

pos = [1, 2, 4, 5, 7, 8]
data = [np.random.normal(0, std, size=100) for std in pos]

plt.violinplot(a, np.arange(len(a)), points=20, widths=0.6, showmeans=True, showextrema=False, showmedians=True)

#========================== plot the histograms spearately =============================
# binMax = 20
# bins = np.linspace(0, 20, 100*binMax)
# bins = np.linspace(0, 20, 100*binMax)

# x = allVals[0][1][:,6]


# freq, edges = np.histogram(x, bins)
# xlims = np.linspace(0, 20, len(x))
# y = stats.kde.gaussian_kde(freq)
# # y = stats.multivariate_normal.pdf(freq, mean=np.mean(x))
# y.covariance_factor = lambda : .15
# y._compute_covariance()



# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# # ax.plot(edges[:-1],freq, alpha=0.9)
# ax.plot(edges[:-1],y(edges[:-1]), alpha=0.3)

# plt.show()

# import seaborn as sns
# sns.set_style('whitegrid')
# sns.kdeplot(np.array(x), bw=0.09)





# label = ['CS','Dop2R','Park25','PINK1RV', 'Trp-Gamma']
# binMax = 20
# bins = np.linspace(0, 20, 100*binMax)

# f, ax = plt.subplots(nrows=2, ncols=3)
# col = 0
# for x, da in enumerate(allVals):
#     r = x%2
#     c = x%3
#     print x, r, c
#     ax[r, c].hist(da[1][:, 6], bins, alpha=0.2)
#     ax[r, c].set_title(label[x])
# plt.show()


# label = ['CS','Dop2R','Park25','PINK1RV', 'Trp-Gamma']
# binMax = 20
# bins = np.linspace(0, 20, 100*binMax)

# f, ax = plt.subplots(nrows=2, ncols=3)
# for x, ids in enumerate(allFlyStats):
#     r = x%2
#     c = x%3
#     print x, r, c
#     plotHist(ids[1][0], bins, ax[r, c])
#     ax[r, c].set_title(label[x])
# plt.show()

def plotHist(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        ax.hist(ids[:, 6], bins, alpha=0.2)
    return ax
#========================== plot the histograms spearately =============================



'''
tVal = st.ttest_ind(allVals[0][1][:,6],allVals[1][1][:,6], axis=0, equal_var=False)
print tVal

tVal = st.ttest_ind(allVals[0][1][:,6],allVals[1][1][:,6], axis=0, equal_var=True)
print tVal


from scipy import stats as st
tVal = st.ttest_ind(allVals[0][1][:,6],allVals[1][1][:,6], axis=0, equal_var=False)
print tVal

tVal = st.ttest_ind(allFlyStats[0][1][0][0][:,6],allFlyStats[0][1][0][6][:,6], axis=0, equal_var=True)
print tVal

for i, d in enumerate(allVals):
    print i, (np.sum(d[1][:,3])/d[0])

binMax = 20
bins = np.linspace(0, 20, 5*binMax)

f, ax = plt.subplots()
plotHist(allFlyStats, bins, ax)
plt.show()


def plotHist(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        ax.hist(ids[:, 6], bins, alpha=0.2)
    return ax

def plotPlot(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        freq, edges = np.histogram(ids[:,6], bins)
        ax.plot(freq,alpha=0.2)
    return ax

f, ax = plt.subplots()
plotPlot(allFlyStats, bins, ax)
plt.show()





def plotPlot(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        freq, edges = np.histogram(ids[:,6], bins)
        ax[i].plot(freq,alpha=0.5)
    return ax

f, ax = plt.subplots(nrows=len(allFlyStats))
plotPlot(allFlyStats, bins, ax)
plt.show()


#a = allFlyStats[0][1]

fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allFlyStats):
    a = da[1][0]
    for i in xrange(len(a)):
        ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(a[i])), a[i][:,6],\
                    s=2, alpha=0.1, linewidths=1, edgecolors=(0,0,1) )

plt.show()



fig, ax = plt.subplots()
scatterDataWidth = 0.0625
for x, da in enumerate(allFlyStats):
    a = da[1]
    for i in xrange(len(a)):
        ax.scatter(np.linspace(x+scatterDataWidth, x-scatterDataWidth,len(a[i])), np.average(a[i][:,6]),\
                    s=2, alpha=0.1, linewidths=1, edgecolors=(0,0,1) )

plt.show()


#ax.set_xlim(0, len(tracksLen))

'''



'''

#====================================================


allFlyStats = []
labels = []
for i in xrange(1,10):
    allFlyStats.append(getFlyStats(cs, i, disMinThres, disMaxThres)[0])
    labels.append(str(i))

binMax = 20
bins = np.linspace(0, 20, 5*binMax)

nrows = 3
ncols = 3

p = 1
f, ax = plt.subplots(nrows,ncols)
for i in xrange(nrows):
    for j in xrange(ncols):
        ax[i,j] = plotHist(allFlyStats[p], bins, ax[i,j])
        ax[i,j].set_ylabel(labels[p])
        p+=1
plt.suptitle('HistBased_win-3:19')
plt.show()

#====================================================

'''


#plots(allFlyStats)




#binStep = 0.2
#b = np.arange(0,10,binStep)
#
#for i, ids in enumerate(allFlyStats):
#    freq, edgs  = np.histogram(ids[:,4],b)
#    step = (binStep/(2*len(allFlyStats)))
#    edg = edgs[:-1]+(step*i)
#    plt.bar(edg,freq, width=(step), color = colors[i], alpha=0.3)
#    print np.argmax(freq), edgs[np.argmax(freq)], freq[np.argmax(freq)]
#plt.show()
#



#------- plot for average speed of ruler distance Vs. actual distance

'''
0) Fix the threshold distance for which the time taken needs to be calculated
1) calculate distance moved in each track by a fly
2) Find average time taken by the fly to cover threshold distance in actual path
3) Find average time taken by the fly to cover threshold distance in a straight vertical line (Distance moved in X-coordinate)




fig, ax = plt.subplots()
scatterDataWidth = 0.0625
ax.plot(trackstats[:,2],color='green', label='Median track Length')
ax.plot(trackstats[:,1], color='blue', label='Average track Length')
ax.plot(trackstats[:,0], color='red',marker='o', label = 'Total number of tracks')
for i in xrange(len(labels[2])):
    ax.scatter(np.linspace(i+scatterDataWidth, i-scatterDataWidth,len(tracs[i])), tracs[i][:,2],\
                s=2, alpha=0.4, linewidths=1, edgecolors=(0,0,1) )
#ax.set_xlim(0, len(tracksLen))
#ax.legend(fontsize='small').draggable()
#ax.set_xticks(np.arange(0,len(totalTracks)))
#ax.set_xticklabels(np.arange(0,len(totalTracks)),winSizes)
#ax.set_xlabel(title)
plt.show()
time.sleep(1)
plt.close()



'''





"""
baseDir = getFolder(baseDir)
rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
imgDatafolder = 'imageData'
trackImExtension = '.jpeg'
csvExt = '.csv'
print "Started processing directories at "+present_time()

cs = []
for rawDir in rawdirs:
    print'=====raw'
    csvData = getData(baseDir, rawDir, imgDatafolder, trackImExtension, csvExt)
    fname = glob.glob(os.path.join(baseDir, rawDir,statsfName+'*'))[0]
    trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
    cs.append([csvData, trackStats])


consecWin = 7
allFlyStats = getFlyStats(cs, consecWin, disMinThres, disMaxThres)[0]

binMax = 20
bins = np.linspace(0, 20, 5*binMax)

f, ax = plt.subplots()
plotHist(allFlyStats, bins, ax)
plt.show()


'''
allFlyStats = []
labels = []
for i in xrange(1,10):
    allFlyStats.append(getFlyStats(cs, i, disMinThres, disMaxThres)[0])
    labels.append(str(i))

binMax = 20
bins = np.linspace(0, 20, 5*binMax)

nrows = 3
ncols = 3

p = 1
f, ax = plt.subplots(nrows,ncols)
for i in xrange(nrows):
    for j in xrange(ncols):
        ax[i,j] = plotHist(allFlyStats[p], bins, ax[i,j])
        ax[i,j].set_ylabel(labels[p])
        p+=1
plt.suptitle('HistBased_win-3:19')
plt.show()

'''

def plotHist(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        ax.hist(ids[:, 6], bins, alpha=0.2)
    return ax

def plotPlot(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        freq, edges = np.histogram(ids[:,6], bins)
        ax.plot(freq,alpha=0.2)
    return ax

f, ax = plt.subplots()
plotPlot(allFlyStats, bins, ax)
plt.show()

def plotPlot(allflystats, bins, ax):
    for i, ids in enumerate(allflystats):
        freq, edges = np.histogram(ids[:,6], bins)
        ax[i].plot(freq,alpha=0.5)
    return ax

f, ax = plt.subplots(nrows=len(allFlyStats))
plotPlot(allFlyStats, bins, ax)
plt.show()




#plots(allFlyStats)




#binStep = 0.2
#b = np.arange(0,10,binStep)
#
#for i, ids in enumerate(allFlyStats):
#    freq, edgs  = np.histogram(ids[:,4],b)
#    step = (binStep/(2*len(allFlyStats)))
#    edg = edgs[:-1]+(step*i)
#    plt.bar(edg,freq, width=(step), color = colors[i], alpha=0.3)
#    print np.argmax(freq), edgs[np.argmax(freq)], freq[np.argmax(freq)]
#plt.show()
#



#------- plot for average speed of ruler distance Vs. actual distance

'''
0) Fix the threshold distance for which the time taken needs to be calculated
1) calculate distance moved in each track by a fly
2) Find average time taken by the fly to cover threshold distance in actual path
3) Find average time taken by the fly to cover threshold distance in a straight vertical line (Distance moved in X-coordinate)


'''



"""
















