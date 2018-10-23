# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:58:29 2017

@author: aman
"""


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


baseDir = '/media/aman/data/flyWalk_data/climbing'

selectedParameter = 'minorAxis_median(mm)'


headers = ['area_average(mm^2)', 'minorAxis_average(mm)', 'majorAxis_average(mm)',\
            'area_median(mm^2)', 'minorAxis_median(mm)', 'majorAxis_median(mm)' ,\
            'nFrames', 'FPS', 'folderName']
statsfName = 'flyStats_'
blankLinesAfterParam = 2
blankLinesBeforeParam = 1
startString = 'Parameters'
selParamIndex = headers.index(selectedParameter)

climbParams = {
                'flyAreaMin' : 300,
                'flyAreaMax' : 900,
                'block' : 91,
                'cutoff' : 35,
                'pixelSize' : 0.055 #pixel size in mm
                }

walkParams = {
                'flyAreaMin' : 1100,
                'flyAreaMax' : 5000,
                'block' : 221,
                'cutoff' : 35,
                'pixelSize' : 0.028 #pixel size in mm
                }
folder = getFolder(baseDir)
baseDir = os.path.dirname(os.path.dirname(folder))
fname = glob.glob(folder+statsfName+'*')[0]

allStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)

pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])

param = allStats[-1][-1][0,selParamIndex]# get selected parameter size in mm

blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)






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


def getAngles(csvData, winlen, cols, consecStep):
    '''
    Input: list of lists of 
        a) csvdata 
        b) CSV file and Image file path details
    
    returns a list containing 
        a) X coordinate
        b) Y coordinate
        c) Angle b/w  i-winLen, i, i+winlen 
        d) Angle b/w  i-winLen, i+winlen/2, i+winlen 
        e) Angle b/w  i-winLen, i-winlen/2, i+winlen 
        f) Eucledian distance between i,i+1
        g) Angle between i, i+1
        
    CSV file and Image file path details
    '''
    thetaCol = cols['thetaColId']
    plusThetaCol = cols['plusThetaColId']
    minusThetaCol = cols['minusThetaColId']
    consDisCol = cols['consecDisColId']
    consAngCol = cols['consecAngColId']
    nCols = len(cols)
    allangles = []
    for _, data in enumerate(csvData):
        csvdata = data[0]
        #Add three columns, 
        #    first column for angle between i-winLen,i, i+winlen            ([:,-1], last column)
        #    second column for angle between i-winLen,i+winlen/2, i+winlen  ([:,-2], second last column)
        #    third column for angle between i-winLen,i-winlen/2, i+winlen   ([:,-3], third last column)
        #    forth column for distance between i and i-1
        #    fifth column for angle between i and i-1 w.r.t to the horizontal
        angles = np.zeros((csvdata.shape[0],nCols)) 
        angles[:,:(csvdata.shape[1]-nCols)] = csvdata
        for i in xrange(winlen, len(csvdata)-winlen-1):
            p0 = csvdata[i-consecStep]
            p1 = csvdata[i]
            p2 = csvdata[i+consecStep]
            euDis = np.linalg.norm(csvdata[i+1]-csvdata[i])
            angles[i, thetaCol] = calcAngle3Pts(csvdata[i-winlen], csvdata[i], csvdata[i+winlen])
            angles[i, plusThetaCol] = calcAngle3Pts(csvdata[i-winlen], csvdata[i+winlen/2], csvdata[i+winlen])
            angles[i, minusThetaCol] = calcAngle3Pts(csvdata[i-winlen], csvdata[i-winlen/2], csvdata[i+winlen])
            angles[i, consDisCol] = euDis
            angles[i, consAngCol] = (calcAngle3Pts(p0,p1,p2))
        a = [angles]
        a.extend(data[1:])
        allangles.append(a)
    return allangles

def getTracks(allang, angThresh, colIds):
    '''
    returns a list of tracks with starting point of the track, length of track and end of track as list elements
    '''
    import copy
    allAng = copy.deepcopy(allang)
    
    thetaCol = colIds['thetaColId']
    plusThetaCol = colIds['plusThetaColId']
    minusThetaCol = colIds['minusThetaColId']

    angleOkay = True
    allTracks = []
    for _, ids in enumerate(allAng):
        angles = ids[0]
        angleAppend = 0
        tracks = []
        for i in xrange(len(angles)):
            if angles[i,thetaCol]<angThresh:
                angleOkay=False
            elif angles[i,thetaCol]>=angThresh:
                if angles[i,plusThetaCol]>=angThresh and angles[i,minusThetaCol]>=angThresh:
                    angleOkay=True
            if angleAppend==0:
                trackStart = i
            if angleOkay==True:
                angleAppend+=1
            if angleOkay==False and angleAppend!=0:
                tracks.append([trackStart, i, angleAppend])
                angleAppend = 0
        ids.extend([tracks])
        allTracks.append(ids)
    print angThresh, '----'
    for _, ids in enumerate(allTracks):
        csvData = ids[0]
        for track in ids[-1]:
            euDis = np.sum(csvData[track[0]:track[1],columnIds['consecDisColId']])
            avAngle = np.average(csvData[track[0]:track[1],columnIds['consecAngColId']])
            stdAngle = np.std(csvData[track[0]:track[1],columnIds['consecAngColId']])
            track.extend([euDis, avAngle, stdAngle])
    return allTracks
    
def getSortedTracks(trackdata, lenThresh):
    '''
    returns the array with tracks sorted for above threshold value
    
    '''
    trackData = copy.deepcopy(trackdata)
    for _, ids in enumerate(trackData):
        sortedTracks = []
        for track in ids[3]:
            if track[2]>lenThresh:
                sortedTracks.append(track)
        ids.extend(np.array([sortedTracks]))
    return trackData

def displayTracks(alltracks):
    totalTracks = len(alltracks)
    for t, ids in enumerate(alltracks):
        csvData = ids[0]
        trackIm = cv2.imread(ids[2])
        for _,track in enumerate(ids[4]):
            track = np.array(track, dtype='uint16')
            for i in range(track[0],track[1]):
                cv2.circle(trackIm, (int(csvData[i,0]), int(csvData[i,1])), 2, (0,200,200),-1)
            cv2.circle(trackIm, (int(csvData[track[0],0]), int(csvData[track[0],1])), 5, (0,0,255),-1)
            cv2.circle(trackIm, (int(csvData[i,0]), int(csvData[i,1])), 5, (0,255,0),-1)
        cv2.putText(trackIm,('Displaying %s/%s tracks'%(t,totalTracks)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100))
        cv2.imshow('sorted', trackIm); cv2.waitKey()
    cv2.destroyAllWindows()

def getTracksStats(allsortedTracks):
    '''
    pulls data of all tracks in the list 'allsortedTracks;
    and returns an array with data of all tracks, their:
        a) total number
        b) mean length
        c) median length
        d) mean distance covered
        e) median distance covered
        f) mean consecutive angle
        g) median consecutive angle
        h) mean of standard deviation of consecutive angle
        i) median of standard deviation of consecutive angle
        
    '''
    first = True
    tracks = None
    for t,ids in enumerate(allsortedTracks):
        sortedTracks = ids[4]
        if len(sortedTracks)!=0:
            if first==True:
                first = False
                tracks = sortedTracks.copy()
            else:
                tracks = np.vstack((tracks, sortedTracks))
    if tracks!=None:
        nTracks = len(tracks)
        meanLen = np.mean(tracks[:,2])
        meanDistance = np.mean(tracks[:,3])
        meanConsecAng = np.mean(tracks[:,4])
        meanStdConsAng = np.mean(tracks[:,5])
        medianLen = np.median(tracks[:,2])
        medianDistance = np.median(tracks[:,3])
        medianConsecAng = np.median(tracks[:,4])
        medianStdConsAng = np.median(tracks[:,5])
        return allsortedTracks, tracks, [nTracks,meanLen,medianLen,meanDistance,medianDistance,meanConsecAng,medianConsecAng,meanStdConsAng,medianStdConsAng]
    else:
        return allsortedTracks, np.zeros((10,6)), [0 for x in xrange(9)]

def getWinLenSlideData(allcsvs, consecStep, winLen, angThresh, trLenStart, trLenStop, trLenStep, colIds):
    '''
    input:
        allcsvs:  csvData list as input
        winStart: start of track length
        winStop: Stop point of track length
        trackStep: Step by which the track length processing would change
    returns:
        allsortedtracks: a list of all csv tracks with respective sortedTracks with detailed data (without trackLength sorting)
        tracks:         a list of details of all tracks after sorting the details
        trackStats:     a numpy array of stats of all data from tracks with multiple tracklength windows
        xlabels:        a list of labels for plotting for variable track length
    '''
    #allangles = getAngles(allcsvs, winLen, colIds)
    alltracks = getTracks(getAngles(allcsvs, winLen, colIds, consecStep), angThresh, colIds)
    trackStats = []
    tracks = []
    sortedTracks = []
    for i in xrange(trLenStart, trLenStop, trLenStep):
        sortedTr, tr, trStats = getTracksStats(getSortedTracks(alltracks, i*trackLen))
        sortedTracks.append(sortedTr)
        tracks.append(tr)
        trackStats.append(trStats)
    trackStats = np.array(trackStats)
    xlabels = [winLen, angThresh, [x for x in xrange(trLenStart, trLenStop, trLenStep)]]
    return sortedTracks, tracks, trackStats, xlabels



#baseDir = '/home/aman/git/cnnflywalk/CS/'
imgDatafolder = 'imageData'
rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])
trackImExtension = '.jpeg'
csvExt = '.csv'
print "Started processing directories at "+present_time()


winLen = blu/3
consecWin = blu/10
angleThresh = 160
trackLen = 1*(blu/10)

columnIds = {
             'xColId':              0,
             'yColId':              1,
             'thetaColId':          2,
             'plusThetaColId':      3,
             'minusThetaColId':     4,
             'consecDisColId':      5,
             'consecAngColId':      6
                }

cs = []
for rawDir in rawdirs:
    cs.append(getData(baseDir, rawDir, imgDatafolder, trackImExtension, csvExt))

allCsvs = []
for _, folder in enumerate(cs):
    for _, csvs in enumerate(folder):
        allCsvs.append(csvs)



def getConsecData(csvData, consecStep, eudDisThresh, fps, bodyLen):
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
            if euDis<eudDisThresh:
                angles.append(np.array([csvdata[i][0], csvdata[i][1], angle, euDis, speed]))
        allangles.append(np.array(angles))
    return allangles

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

def plotThetaDis(csvAngData, tracklenthresh, c, angs, consecStep, bodyLen):
    f, axarr = plt.subplots(3,1)
    im = np.ones((450, 1280, 3), dtype='uint8')*255
    for i, ids in enumerate(csvAngData):
        if np.sum(ids[:,3])>tracklenthresh:
            x = len(ids)
            #c[i] = (0,0,0)
            axarr[0].plot(np.arange(len(angs[i])), angs[i][:, -1], '.', color = (0,0,0), alpha = 0.1)
            axarr[1].plot(np.arange(len(angs[i])), angs[i][:, 0], '.',color = (0,0,0), alpha = 0.1)
            for p in xrange(x):
                cv2.circle(im, (int(ids[p,0:2][0]), int(ids[p,0:2][1])), 1, [col*255 for col in c[i]], 1)
    axarr[2].imshow(im)
    axarr[0].set_ylabel('speed\n(Body length units per second)')
    axarr[1].set_ylabel('angle')
    axarr[0].set_xlabel('Body Length Units')
    axarr[1].set_xlabel('Body Length Units')
    axarr[1].set_ylim(200,-10)
    print axarr[1].get_xlim()
    labels = [item.get_text() for item in axarr[1].get_xticklabels()]
    labels = np.array(np.linspace(0, (axarr[1].get_xlim()[-1]*consecStep)/blu, len(labels)), dtype='uint16')
    axarr[0].set_xticklabels(labels)
    axarr[1].set_xticklabels(labels)
    axarr[0].set_ylim(-1,30)
    plt.show()
      
colors = [random_color() for c in xrange(1000)]

disThres = 2*blu
consecWin = float(blu)/20

fps = 250
angData = getConsecData(cs[0],consecWin, disThres, fps, blu)
angles = np.zeros((len(angData),max([len(x) for x in angData]),2))
angles[:] = np.nan
for i,ang in enumerate(angData):
    for j in xrange(len(ang)):
        angles[i,j,0] = ang[j,2]
        angles[i,j,1] = ang[j,-1]


trackLenThresh = 10*blu

plotThetaDis(angData, trackLenThresh,  colors, angles, consecWin, blu)

'''
angData = getAngles(allCsvs, winLen, columnIds, consecWin)
_, tracs, trackstats, labels = getWinLenSlideData(allCsvs, consecWin, blu, 140, blu/10, 5*blu, blu/10, columnIds)
tracksData, tracs, trackstats, labels = getWinLenSlideData(allCsvs, consecWin, blu, 1, blu/20, blu, blu/20, columnIds)

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


winLenStart = blu/4
winLenStop = blu
winLenStep = blu/4


angThreshStart = 130
angThreshStop = 170
angThreshStep = 5

trackLenStart = blu/10
trackLenStop = blu
trackLenStep = blu/10
xlabels = [x for x in xrange(0,trackLenStop,trackLenStep)]
trackLenlabels = [round(float(x)/blu,1) for x in xrange(0,trackLenStop,trackLenStep)]
#trackLenlabels = [float(x)/2 for x in xrange(len(xlabels))]

winLenShift = []
for winLen in xrange(winLenStart, winLenStop, winLenStep):
    angThresShift = []
    for angThres in xrange(angThreshStart,angThreshStop,angThreshStep):
        processedData = getWinLenSlideData(allCsvs, consecWin, winLen, angThres, trackLenStart, trackLenStop, trackLenStep, columnIds)
        angThresShift.append(processedData)
    winLenShift.append(angThresShift)

print('done processing, now plotting')

color_nTrack = (0.5,0,0)

f, axarr = plt.subplots(len(winLenShift), len(angThresShift), sharey=True)
for i in xrange(len(winLenShift)):
    for j in xrange(len(angThresShift)):
        _, tracs, trackstats, labels = winLenShift[i][j]
        axarr[i,j].plot(trackstats[:,0], color=color_nTrack,marker='o', label = 'Total number of tracks')
        axarr_ = axarr[i,j].twinx()
        axarr_.plot(trackstats[:,2],color='green', label='Median track Length')
        axarr_.plot(trackstats[:,1], color='blue', label='Average track Length')
        axarr_.tick_params('y', colors=(0,0.2,1))
        axarr[i,j].tick_params('y', colors=color_nTrack)
        for k in xrange(len(labels[2])):
            axarr_.scatter(np.linspace(k+scatterDataWidth, k-scatterDataWidth,len(tracs[k])), tracs[k][:,2],\
                        s=2, alpha=0.4, linewidths=1, edgecolors=(0,0,1) )
        if i==(0):
            axarr[i, j].set_title('AngleThresh: %s'%(labels[1]))
        if j==0:
            axarr[i, j].set_ylabel('WinLen: %0.1f B.L.U'%(float(labels[0])/blu))
        axarr[i, j].set_xticklabels(trackLenlabels)
plt.show()
            





winS = 0
angT1 = 0
angT2 = -1
f, axarr = plt.subplots(len(winLenShift[winS:]), len(angThresShift[angT1:angT2]), sharey=True)
for i in xrange(len(winLenShift[winS:])):
    for j in xrange(len(angThresShift[angT1:angT2])):
        _, tracs, trackstats, labels = winLenShift[winS:][i][j+angT1]
        axarr_ = axarr[i,j].twinx()
        axarr[i,j].plot(trackstats[:,2],color='green', label='Median track Length')
        axarr[i,j].plot(trackstats[:,1], color='blue', label='Average track Length')
        axarr[i,j].tick_params('y', colors=(0,0.2,1))
        for k in xrange(len(labels[2])):
            axarr[i,j].scatter(np.linspace(k+scatterDataWidth, k-scatterDataWidth,len(tracs[k])), tracs[k][:,2],\
                        s=2, alpha=0.4, linewidths=1, edgecolors=(0,0,1) )
        if i==(0):
            axarr[i, j].set_title('AngleThresh: %s'%(labels[1]))
        if j==0:
            axarr[i, j].set_ylabel('WinLen: %0.1f B.L.U'%(float(labels[0])/blu))
        #axarr[i, j].set_xticks(xlabels)
        axarr[i, j].set_xticklabels(trackLenlabels)
        axarr_.plot(trackstats[:,0], color=color_nTrack,marker='o', label = 'Total number of tracks')
        axarr_.tick_params('y', colors=color_nTrack)
plt.show()
            





winS = -5
angT1 = -5
angT2 = -1
f, axarr = plt.subplots(len(winLenShift[winS:]), len(angThresShift[angT1:angT2]), sharey=True)
for i in xrange(len(winLenShift[winS:])):
    for j in xrange(len(angThresShift[angT1:angT2])):
        _, tracs, trackstats, labels = winLenShift[winS:][i][j+angT1]
        axarr_ = axarr[i,j].twinx()
        axarr[i,j].plot(trackstats[:,2],color='green', label='Median track Length')
        axarr[i,j].plot(trackstats[:,1], color='blue', label='Average track Length')
        axarr[i,j].tick_params('y', colors=(0,0.2,1))
        if i==(0):
            axarr[i, j].set_title('AngleThresh: %s'%(labels[1]))
        if j==0:
            axarr[i, j].set_ylabel('WinLen: %0.1f B.L.U'%(float(labels[0])/blu))
        #axarr[i, j].set_xticks(xlabels)
        #axarr[i, j].set_xticklabels(trackLenlabels)
        axarr_.plot(trackstats[:,0], color=color_nTrack,marker='o', label = 'Total number of tracks')
        axarr_.tick_params('y', colors=color_nTrack)
plt.show()









'''















