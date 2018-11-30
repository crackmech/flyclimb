#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:43:15 2018

@author: pointgrey
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:19:19 2018

@author: aman
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:25:20 2018

@author: aman
"""
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




unitTime = 60
nUnitTimes = maxTimeThresh/unitTime
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



imgDatafolder = 'imageData'
trackImExtension = '.jpeg'
csvExt = 'trackData*.csv'
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

import os
import glob
import re
import random
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import matplotlib.pyplot as plt
from scipy import stats
import xlwt
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
            allStats.append((lines[parInd[i]-1], header, stats, avStats))
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
        allangles.append((np.array(angles), data[1], data[2])) #data[1] contains csv filename, data[2] contins img filename
    return allangles

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

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



speedBinMin = disMinThres
speedBinMax = disMaxThres
speedBinStep = 0.1
speedBins = np.arange(speedBinMin, speedBinMax, speedBinStep)

baseDir = '/media/aman/data/flyWalk_data/climbingData/'
baseDir = '/media/pointgrey/data/flywalk/'
#baseDir = '/media/pointgrey/data/flywalk/climbingData/plots/csvDir_20180901/fig3/'
colorsRandom = [random_color() for c in xrange(1000)]

baseDir = getFolder(baseDir)
dirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

if  'W1118' in dirs:
    if 'CS' in dirs:
        csIndex = dirs.index('CS')
        w1118Index = dirs.index('W1118')
        dirs.pop(csIndex)
        dirs.insert(0, 'CS')
        dirs.pop(w1118Index)
        dirs.insert(1, 'W1118')
    else:
        w1118Index = dirs.index('W1118')
        dirs.pop(w1118Index)
        dirs.insert(0, 'W1118')
        
saveDir = baseDir+'_'
saveFiles = ''
for _,d in enumerate(dirs):
    saveFiles+='_'+d
saveFiles

print "Started processing directories at "+present_time()

def getAllFlyStats(genotypeDir):
    '''
    returns allFlyStats for a all fly folders in a given folder (genotypeDir)
    
    '''
    rawdirs = natural_sort([ name for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    cs = []
    for rawDir in rawdirs:
        print'=====raw'
        csvData = getData(genotypeDir, rawDir, imgDatafolder, trackImExtension, csvExt)
        fname = glob.glob(os.path.join(genotypeDir, rawDir, statsfName+'*'))[0]
        trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
        cs.append([csvData, trackStats])
    
    return getFlyStats(cs, consecWin, disMinThres, disMaxThres)

def getAllFlyCsvData(genotypeDir):
    '''
    returns data of all Csv's of all tracks for all fly folders in a given folder (genotypeDir)
    
    '''
    rawdirs = natural_sort([ name for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    cs = []
    for rawDir in rawdirs:
        print'=====raw'
        csvData = getData(genotypeDir, rawDir, imgDatafolder, trackImExtension, csvExt)
        fname = glob.glob(os.path.join(genotypeDir, rawDir,statsfName+'*'))[0]
        trackStats = getAllStats(fname, headers, blankLinesAfterParam, blankLinesBeforeParam, startString)
        cs.append([csvData, trackStats])
    return cs

def getTimeDiffFromTimes(t2, t1):
    '''
    returns the time difference between two times, t2 and t1, (input in format '%Y%m%d_%H%M%S')
    returns no. os seconds elapsed between t2 and t13
    '''
    time1 = datetime.strptime(t1, '%Y%m%d_%H%M%S')
    time2 = datetime.strptime(t2, '%Y%m%d_%H%M%S')
    return (time2-time1).total_seconds()


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
allGenotypesCsvData = []
for _,d in enumerate(dirs):
    path = os.path.join(baseDir, d)
    allGenotypesCsvData.append([path, getAllFlyCsvData(path)])


genoTypeDataProcessed = []
for g, genotype in enumerate(allGenotypesCsvData):
    for f, fly in enumerate(genotype[1]):
        flyAlltracks = []
        flyTimeThTracks = []
        print "------",f,"------"
        for t, tracks in enumerate(fly[0]):
            print t

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

def getTrackData(csvdata, skipFrames, consecStep, eudDisMinThresh, eudDisMaxThresh, bodyLen, fps):
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
    consecStep = int(consecStep)
    angles = []
    startFrame = consecStep*skipFrames
    stopFrame = len(csvdata)-(consecStep*skipFrames)-1
    for i in xrange(startFrame, stopFrame, consecStep):
        p0 = csvdata[i-consecStep]
        p1 = csvdata[i]
        p2 = csvdata[i+consecStep]
        euDis = np.linalg.norm(csvdata[i+consecStep]-csvdata[i])
        angle = (calcAngle3Pts(p0,p1,p2))
        if eudDisMinThresh > euDis or euDis > eudDisMaxThresh:
            angle = np.nan
            euDis = np.nan
        speed = (euDis*fps)/(consecStep*bodyLen)
        angles.append(np.array([csvdata[i][0], csvdata[i][1], angle, euDis, speed]))
    angles = np.array(angles)
    speedTrack = angles[:,4]
    trackAvInsSpeed = np.nanmean(speedTrack)
    trackDis = np.nansum(speedTrack)
    trackAvSpeed = trackDis*fps/(consecStep*bodyLen*len(angles))
    trackDirection = getTrackDirection(angles, bodyLen)
    trackDetails = [trackAvInsSpeed, trackAvSpeed, trackDis, trackDirection]
    trackDetailsHeader = ['Average InsSpeed for the track', 'AverageSpeed','Total distance of the track', 'Track Direction']
    return angles, trackDetails, trackDetailsHeader , len(angles)#data[1] contains csv filename, data[2] contins img filename

def getFlyDetails(allStats):
    '''
    returns average FPS, body length for a fly by getting details from its folder
    '''
    fps = allStats[1][-1][0,-1]
    pixelSize =float( [x for x in allStats[1][0].split(',') if 'pixelSize' in x ][0].split(':')[-1])
    param = allStats[1][-1][0,selParamIndex]# get selected parameter size in mm
    blu = int(param/pixelSize) #Body length unit, used for stats calculations w.r.t the body length (minorAxis length)
    return blu, fps


genoTypeDataProcessed = []
for g, genotype in enumerate(allGenotypesCsvData):
    allFlyAllTracks = []
    for f, fly in enumerate(genotype[1]):
        blu, fps = getFlyDetails(fly[1])
        flyAlltracks = []
        flyTimeThTracks = []
        print "------",f,"------"
        for t, tracks in enumerate(fly[0]):
            if t==0:
                trackTimeDiff = 0
                startTrackCsvName = fly[0][t][1]
                startTrackTime = startTrackCsvName.split('/')[-1].split('_trackData')[0]
            else:
                currTrackCsvName = fly[0][t][1]
                currTrackTime = currTrackCsvName.split('/')[-1].split('_trackData')[0]
                trackTimeDiff  = getTimeDiffFromTimes(currTrackTime, startTrackTime)
            if len(tracks[0])>(2+(2*consecWin*chukFrames)):
                trackData = getTrackData(tracks[0], chukFrames, consecWin, blu/50.0, blu, blu, fps)
                flyAlltracks.append([trackData, trackData[1][-1],  trackTimeDiff, tracks[1:]])
        allFlyAllTracks.append(flyAlltracks)
    genoTypeDataProcessed.append(allFlyAllTracks)



#--- avAvspeed for each fly-----
def getFlySpeedDisData(flyTrackData, timeThresh, trackLenThresh, unitTime, imFolder):
    '''
    returns the 
        average speed
        STDEV of average speed
        distanceTravelled in timeThresh
        number of tracks in timeThresh
        distanceTravelled in unitTime
        nTracks in unitTime
    '''
    print flyTrackData[0][-1][0].split('/')[-3]
    flyAllData = []
    flyAllInsSpeeds = []
    flyGeoIndex = 0
    for _,tr in enumerate(flyTrackData):
        if tr[2]<timeThresh:
            if tr[0][1][2]>trackLenThresh:
                avSpeed = tr[0][1][1] # average speed of the fly
                dis = tr[0][1][2] # distance covered by the fly
                insSpeeds = tr[0][0][:,-1] # list of instantaneous speed of the track
                flyGeoIndex+=tr[1][0] # geotactic index of the fly
                pathR = abs(tr[1][1]) # value of 'r' value of the path
                flyAllData.append([avSpeed, dis,flyGeoIndex, tr[0][-1], pathR, tr[2]])
                flyAllInsSpeeds.extend(insSpeeds[~np.isnan(insSpeeds)])
    flyAllData = np.array(flyAllData)
    flyDisPerUnitTime = []
    print flyAllData.shape
    for j in xrange(unitTime, timeThresh+1, unitTime):
        disPerUT = []
        for i in xrange(len(flyAllData[:,-1])):
            if (j-unitTime)<=flyAllData[i,-1]<j:
                disPerUT.append(flyAllData[i,:])
        flyDisPerUnitTime.append(np.array(disPerUT))
        '''
        flyAllData contains: avSpeed per track, distance moved per track, geotactic Index, nFrames per track, time from starting imaging of the fly
        flyAllInsSpeeds contains: a single arrray of all instaneous speeds of the fly
        flyDisPerUnitTime contains: a list of avSpeed,DisMoved,geotactic Index,nFrames,timeFromStarting per unit time, for time segment plots
        '''
    return np.array(flyAllData), np.array(flyAllInsSpeeds), flyTrackData[0][-1][0].split(imFolder)[0], flyDisPerUnitTime

chukFrames = 20 # number of frames to be chucked from start and end of the track to initiate data calculation
minTrackLen = blu*3
nParams = 6
unitDataIndex = -1
xlGapColumns = 2


allGenotypeMovementData = []
for i, genotypeTracks in enumerate(genoTypeDataProcessed):
    genotypeMovementData = []
    for _,flAllTrks in enumerate(genotypeTracks):
        flyMovementData = getFlySpeedDisData(flAllTrks, maxTimeThresh, minTrackLen, unitTime, imgDatafolder)
        genotypeMovementData.append(flyMovementData)
    allGenotypeMovementData.append(genotypeMovementData)     



genotypeAvSpeed = []
genotypeDis = []
genotypeNTracks = []
genotypeGeoTacInd = []
genotypeLenTrack = []
genotypeStraight =[]
genotypeInsSpeed = []
genotypeName = []
genotypeMarker = []

for _,genotype in enumerate(allGenotypeMovementData):
    flyName = []
    flyAvSpeed = []
    flyDis = []
    flyNTracks = []
    flyGeoTacInd = []
    # flyDisPerUT = []
    # flyNTrackPerUT = []
    flyInsSpeed = []
    flylenTrack = []
    flyStraight = []
    flyMarker = []
    for i,fly in enumerate(genotype):
        flyAvSpeed.append(np.mean(fly[0][:,0]))
        flyDis.append(np.nansum(fly[0][:,1]))
        if i==0:
            flyInsSpeed = fly[1]
            #flylenTrack = fly[0][:,3]# for track duration
        else:
            flyInsSpeed = np.hstack((flyInsSpeed, fly[1]))
            #flylenTrack = np.hstack((flylenTrack,fly[0][:,3]))# for track duration
        flylenTrack.append(np.median(fly[0][:,3]))# for track duration
        flyGeoTacInd.append(fly[0][-1,2]/len(fly[0]))
        flyNTracks.append(len(fly[0]))
        flyStraight.append(np.nanmean(fly[0][:,4]))
        flyName.append(fly[2])
        if '_male' in fly[2]:
            flyMarker.append((1,1,1))# open circles for males
        elif '_female' in fly[2]:
            flyMarker.append((0,0,0))# closed circles for females
    genotypeLenTrack.append(np.array(flylenTrack))
    genotypeAvSpeed.append(np.array(flyAvSpeed))
    genotypeDis.append(np.array(flyDis))
    genotypeNTracks.append(np.array(flyNTracks))
    genotypeGeoTacInd.append(np.array(flyGeoTacInd))
    genotypeInsSpeed.append(flyInsSpeed)
    genotypeStraight.append(flyStraight)
    genotypeName.append(flyName)
    genotypeMarker.append(flyMarker)

#plotTitles = ['Number of Tracks\nin 5 minutes',
#              'Duration of Tracks',
#              'Total Distance Travelled\nin 5 minutes',
#              'Average Speed',
#              'Path Straightness',
#              'Geotactic Index',
#              ]
#
#plotTitlesPerUT = ['Number of Tracks',
#              'Duration of Tracks',
#              'Total Distance Travelled',
#              'Average Speed',
#              'Path Straightness',
#              'Geotactic Index',
#              ]
#
#plotYLabels = ['Number of Tracks',
#                'duration of Tracks\n(s)',
#                'Distance Traveled\n'+r'(BLU x10$^3$)',
#                'Average Speed\n(BLU/S)',
#                'Path Straightness\n'+r'(R$^2$ Value)',
#                'Geotactic Index',
#                ]
#
#plotYLabels5min = ['Number of Tracks',
#                'Seconds',
#                r'Body Lengths (x10$^3$)',
#                'Body Length / S',
#                r'R$^2$ Value',
#                'Geotactic Index',
#                ]
#
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
               r'BLU (x10$^3$)',
               'BLU/S',
               r'R$^2$ Value',
               'Geotactic Index',
                ]

def getLenTrackStats(trackLenArray):
    if trackLenArray.size > 0:
        return np.median(trackLenArray)
    else:
        return 0


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


allGenotypePerUT_Data = []
for _,genotype in enumerate(allGenotypeMovementData):
    genotypePerUT_Data = []
    for i,fly in enumerate(genotype):
        flyDataPerUT = np.zeros((nUnitTimes, nParams))
        nTracks = 0
        for t in xrange(nUnitTimes):
            if fly[unitDataIndex][t].size > 0:
                nTracks+=len(fly[unitDataIndex][t])
                flyDataPerUT[t,0] = (len(fly[unitDataIndex][t]))# nTracks
                flyDataPerUT[t,1] = (getLenTrackStats(fly[unitDataIndex][t][:,3])) # duration of track
                flyDataPerUT[t,2] = (np.nansum(fly[unitDataIndex][t][:,1])) # Total Distance
                flyDataPerUT[t,3] = (np.nanmean(fly[unitDataIndex][t][:,0])) #Average Speed
                flyDataPerUT[t,4] = (np.nanmean(fly[unitDataIndex][t][:,4])) # Path Straightness
                flyDataPerUT[t,5] = (fly[unitDataIndex][t][-1, 2])/nTracks # Geotactic Index
        genotypePerUT_Data.append(flyDataPerUT)
    allGenotypePerUT_Data.append(genotypePerUT_Data)
    # allGenotypePerUT_Data.append(np.array(genotypePerUT_Data))


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
#ax[legendAxesRowSet, legendAxesColSet].legend(handles=legendHandles,labels=legendLabels, bbox_to_anchor=(legendHorPos, legendVerPos), loc=2, shadow=True, edgecolor=(0,0,0), fontsize='x-small', ncol=len(genotypes)).draggable()
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




sheetNames = ['NumTracks','TrackDuration','TotalDistance',\
              'AvSpeed','GeotacticIndex', 'Straightness', 'flyDetails']

columnHeader = 'TimePoint'
skipheaderCells = 2

#---- Save sheet for Per minute data------------
paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]
for g, gt in enumerate(allGenotypePerUT_Data):
    for f, fly in enumerate(gt):
        for timepoint in xrange(fly.shape[0]):
            for parameter in xrange(fly.shape[1]):
                col = g+(timepoint*(len(allGenotypePerUT_Data)+xlGapColumns))
                if f==0:
                    if g==0:
                        timepointHeader =  '%s: %d minute'%(columnHeader, timepoint+1)
                        sheets[parameter].write(f,col+len(allGenotypePerUT_Data)/2,timepointHeader)
                    sheets[parameter].write(f+1, col, dirs[g])
                row = f+skipheaderCells
                sheets[parameter].write(row,col, fly[timepoint, parameter])

xlName = "climbingPerMinuteParameters_genotypesTogether"
paramBook.save("%s%s_%s%s.xls"%(saveDir, present_time(), xlName, saveFiles))


paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]
for g, gt in enumerate(allGenotypePerUT_Data):
    for f, fly in enumerate(gt):
        for timepoint in xrange(fly.shape[0]):
            for parameter in xrange(fly.shape[1]):
                col = timepoint+(g*(fly.shape[0]+xlGapColumns))
                if f==0:
                    timepointHeader =  '%d minute'%(timepoint+1)
                    sheets[parameter].write(f+1,col,timepointHeader)
                    if timepoint==0:
                        sheets[parameter].write(f, col+len(allGenotypePerUT_Data)/2, dirs[g])
                row = f+skipheaderCells
                sheets[parameter].write(row,col, fly[timepoint, parameter])
    
xlName = "climbingPerMinuteParameters_timepointsTogether"
paramBook.save("%s%s_%s%s.xls"%(saveDir, present_time(), xlName, saveFiles))

#---- Save sheet for 5minutes data------------
genotypeParams = [genotypeNTracks,
                  genotypeLenTrack,
                  genotypeDis,
                  genotypeAvSpeed,
                  genotypeGeoTacInd,
                  genotypeStraight,
                  genotypeName]
paramBook = xlwt.Workbook(encoding='utf-8', style_compression = 0)
sheets = [paramBook.add_sheet(x, cell_overwrite_ok = True) for x in sheetNames]

for s in xrange(len(sheets)):
    sheet = sheets[s]
    for g in xrange(len(genotypeParams[s])):
        for row in xrange(len(genotypeParams[s][g])):
            if row==0:
                sheet.write(row,g,dirs[g])
            sheet.write(row+skipheaderCells,g,genotypeParams[s][g][row])

xlName = "climbingParameters5Minutes_genotypesTogether"
paramBook.save("%s%s_%s%s.xls"%(saveDir, present_time(), xlName, saveFiles))



#------- CHECK for NORMALITY --------
params = ['nTracks', 'trackDuration', 'Distance',\
          'Speed', 'PathStraightness', 'GeotacticIndex']
genotypeParams = [genotypeNTracks,
                  genotypeLenTrack,
                  genotypeDis,
                  genotypeAvSpeed,
                  genotypeStraight,
                  genotypeGeoTacInd]

f = open(("%s%s_climbing5MinutesStats%s.csv"%(saveDir, present_time(), saveFiles)), 'wa')
#--Check normality for 5 minutes data----
print '\n\n\n----Check normality for 5 minutes data----'
for p, par in enumerate(genotypeParams):
    print '------', params[p],'------'
    f.write('\n\n------Normality check for %s------\n'%params[p])
    for g, gt in enumerate(par):
        print stats.normaltest(gt)
        f.write('%s: %s\n'%(genotypes[g],str(stats.normaltest(gt))))
f.close()    
#--Check normality for Per minute data----

f = open(("%s%s_climbingPerMinuteStats%s.csv"%(saveDir, present_time(), saveFiles)), 'wa')
fn = open(("%s%s_climbingPerMinuteNormalityStats%s.csv"%(saveDir, present_time(), saveFiles)), 'wa')
f.write('\nKruskal-Wallis test for: ')
fn.write('\nD’Agostino-Pearson’s Normality test for: \n')
print '\n\n\n---Checking normality for Per minute data----'
for t in xrange(nUnitTimes):
    for p, par in enumerate(params):
        gtData = []
        for i in xrange(len(dirs)):
            parData = [allGenotypePerUT_Data[i][x][t,p] for x in xrange(len(allGenotypePerUT_Data[i]))]
            gtData.append(parData)
            print 'Normality value for: %s of %s (%d minute)'%(params[p], dirs[i], (t+1))
            fn.write('Normality value for: %s of %s (%d minute): %s\n'%(params[p], dirs[i], (t+1), str(stats.normaltest(parData))))
            print ('normal: %f, ShapiroWilk: %f'%(stats.normaltest(parData)[1], stats.shapiro(parData)[1]))
        print '\n---KruskalWallis:',stats.kruskal(*gtData)
        print '---OneWayANOVA:',stats.f_oneway(*gtData)
        f.write('\n:%s (%d minute): %s'%(params[p], t+1, str(stats.kruskal(*gtData))))
f.close()

