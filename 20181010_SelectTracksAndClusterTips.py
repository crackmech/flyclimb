#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 04:06:14 2018

@author: aman
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 02:20:00 2018

@author: aman
"""

import cv2
import os
import numpy as np
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import multiprocessing as mp
import time
import glob
import random
import csv
import itertools
from PIL import ImageTk, Image
import copy


nThreads = 4
imDataFolder = 'imageData'


frmBarName = 'frame Number'
startFrmTrkBarName = 'Select Start Frame'
stopFrmTrkBarName = 'Select Stop Frame'
addSelectionTrckBar = 'Add frames'
removeSelectionTrckBar = 'Remove frames'

windowName = 'trackImage'

startFrame = 0
stopFrame = -1
frameIdx = []


nLegs = 6
h=100
w=100
xCol = 2
yCol = 3
cusIdCol = 4

nlegRows = 2
nlegCols = 3
nCols = 7
legList = [['L1','L2','L3'],['R1','R2','R3']]
legListIds = [[(x+1)*10+y+1 for y in xrange(nlegCols)] for x in xrange(nlegRows)]
allClusImloc = [0,0]


colors = \
[(64, 96, 32), (96, 0, 160), (96, 128, 32), (128, 192, 224), (128, 32, 0), (0, 224, 64),\
 (224, 96, 0), (160, 0, 64), (32, 32, 64), (160, 192, 224), (160, 64, 96), (160, 96, 64),
 (224, 160, 224), (192, 96, 128), (128, 160, 64), (192, 32, 192), (160, 96, 32), (32, 96, 32),
 (32, 128, 96), (224, 32, 96), (128, 0, 160), (64, 224, 32), (32, 64, 32), (192, 96, 224),
 (0, 192, 0), (0, 32, 0), (128, 96, 224), (32, 224, 64), (64, 32, 64), (224, 128, 32), 
 (32, 192, 96),  (128, 96, 128), (32, 64, 224), (160, 160, 64), (32, 32, 160), (128, 192, 128),
 (128, 128, 96), (192, 0, 32), (64, 192, 224), (64, 32, 128), (96, 32, 160), (160, 160, 32),
 (224, 224, 96), (224, 192, 224), (96, 0, 64), (224, 224, 128), (32, 224, 128), (64, 64, 128),
 (64, 64, 192), (64, 64, 64), (64, 192, 224), (96, 128, 64), (192, 64, 160), (96, 64, 0),
 (192, 32, 0), (192, 96, 96), (192, 224, 0), (192, 224, 128), (224, 64, 0), (0, 96, 192)]





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

colors = [random_color() for i in xrange(20)]
def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    return rows
    
def getlegTipData(csvFname):
    return np.array(readCsv(csvFname)[1:])

def getCentroidData(csvFname):
    return readCsv(csvFname)[1:]

def imRead(x):
    return cv2.imread(x, cv2.IMREAD_COLOR)
    #return cv2.rotate(cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_COUNTERCLOCKWISE)

def getImStack(flist, pool):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    #startTime = time.time()
    imgStack = pool.map(imRead, flist)
    ims = np.zeros((len(imgStack), imgStack[0].shape[0], imgStack[0].shape[1], imgStack[0].shape[2]), dtype=np.uint8)
    for i,im in enumerate(imgStack):
        ims[i]=im
    imgStack = ims.copy()
    return imgStack

def setColors(trackData):
    blue = np.hstack((np.linspace(0, 255, num = len(trackData)/2),np.linspace(255, 0, num = (len(trackData)/2)+1)))
    green = np.linspace(255, 0, num = len(trackData))
    red = np.linspace(0, 255, num = len(trackData))
    return [(blue[i], green[i], red[i]) for i in xrange(len(trackData))]

def getImStackWithCentroids(flist, centroids, pool):
    imStack = getImStack(flist, pool)
    colors = setColors(centroids)
    for i in xrange(len(imStack)):
        for j in xrange(len(centroids)):
            cv2.circle(imStack[i], (int(centroids[j][0]), int(centroids[j][1])), 2, colors[j], 2)
        for j in xrange(len(centroids)):
            if j%50==0:
                cv2.putText(imStack[i], str(j), (int(centroids[j][0]), int(centroids[j][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    return imStack

def getImStackWithColor(imStack, centroids, color):
    for i in xrange(len(imStack)):
        for j in xrange(len(centroids)):
            cv2.circle(imStack[i], (int(centroids[j][0]), int(centroids[j][1])), 2, color, 2)
        for j in xrange(len(centroids)):
            if j%50==0:
                cv2.putText(imStack[i], str(j), (int(centroids[j][0]), int(centroids[j][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    return imStack


def displayImgs(imgs, fps):
    f = 1000/fps
    for i, img in enumerate(imgs):
        cv2.imshow('123',img)
        key = cv2.waitKey(f) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            f = 1000/fps
            cv2.waitKey(0)
        if key == ord("n"):
            cv2.imshow('123',imgs[i+1])
            f=0
            cv2.waitKey(f)
    cv2.destroyAllWindows()


def nothing(x):
    if cv2.getTrackbarPos(addSelectionTrckBar, windowName)==1:
        cv2.setTrackbarPos(addSelectionTrckBar, windowName, 0)
    #pass

def setStartFrame(x):
    global startFrame
    startFrame = cv2.getTrackbarPos(frmBarName, windowName)
    
    return startFrame

def setStopFrame(x):
    global stopFrame
    stopFrame = cv2.getTrackbarPos(frmBarName, windowName)
    
    return stopFrame

def getFinalSelection(x):
    global frameIdx, imgs
    if x==1:
        frameIdx.append([startFrame, stopFrame])
        
        print 'Selected Frames: ',frameIdx
        cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
        cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)

def popLastSelection(x):
    global frameIdx
    if x==1:
        frameIdx.pop(-1)
        print 'Selected Frames: ',frameIdx
        cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
        cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)

def selectTrackFrames(flist, centroids, pool):
    
    '''
    Displays the images from the folder with overlayed centroids on the whole imstack.
    By using the slider, we can determine where to start and stop the track for legTip clustering.
    '''
    global startFrame, stopFrame, imgs
    imgs = getImStackWithCentroids(flist, centroids, pool)
    cv2.namedWindow(windowName)
    cv2.moveWindow(windowName, 30,30)
    cv2.createTrackbar(frmBarName, windowName, 0, (len(imgs)-2), nothing)
    cv2.createTrackbar(startFrmTrkBarName, windowName, 0, 1, setStartFrame)
    cv2.createTrackbar(stopFrmTrkBarName, windowName, 0, 1, setStopFrame)
    cv2.createTrackbar(addSelectionTrckBar, windowName, 0, 1, getFinalSelection)
    cv2.createTrackbar(removeSelectionTrckBar, windowName, 0, 1, popLastSelection)
    cv2.setTrackbarPos(frmBarName, windowName, 0)
    cv2.setTrackbarPos(startFrmTrkBarName, windowName, 0)
    cv2.setTrackbarPos(stopFrmTrkBarName, windowName, 0)
    cv2.setTrackbarPos(removeSelectionTrckBar, windowName, 0)
    while (1):
        imgNum = cv2.getTrackbarPos(frmBarName, windowName)
        cv2.imshow(windowName,imgs[imgNum] )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyWindow(windowName)






def getClusImStack(xyValArr, nrows, ncols, clusids, clrs):
    clusims = [[np.zeros((2*h,2*w,3), dtype = np.uint8)+20 for x in xrange(ncols)]\
                for y in xrange(nrows) ]
    allClusImloc = [0,0]
    for i in xrange(nrows):
        for j in xrange(ncols):
            if [i,j]==allClusImloc:
                for x in xrange(len(xyValArr)):            
                    clusId = int(xyValArr[x, cusIdCol])
                    pxX = int(xyValArr[x,xCol])
                    pxY = int(xyValArr[x,yCol])
                    clusims[i][j][pxY, pxX, ] = clrs[clusId]
            else:
                nClus = clusids[i][j]
                for x in xrange(len(xyValArr)):
                    clusId = int(xyValArr[x, cusIdCol])
                    if clusId == nClus:
                        pxX = int(xyValArr[x,xCol])
                        pxY = int(xyValArr[x,yCol])
                        clusims[i][j][pxY, pxX, ] = clrs[clusId]
    return clusims

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points
    between p1 and p2
    https://stackoverflow.com/questions/43594646/how-to-calculate-the-coordinates-of-the-line-between-two-points-in-python
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])

def updateImDisplay(clusims):
    for x in range(nRows):
        for y in range(nCols):
            img = ImageTk.PhotoImage(Image.fromarray(clusims[x][y]))
            btn_matrix[x][y].config(image=img)
            btn_matrix[x][y].image = img


def updateClusters(selClusVal, valPlt, clusims, clusids):
    clusId1 = clusids[selClusVal[0][0]][selClusVal[0][1]]
    clusId2 = clusIds[selClusVal[1][0]][selClusVal[1][1]]
    clusids[selClusVal[0][0]][selClusVal[0][1]] = clusids[selClusVal[1][0]][selClusVal[1][1]]
    for x in xrange(len(valPlt)):
        clusId = int(valPlt[x, cusIdCol])
        if clusId == clusId1:
            valPlt[x, cusIdCol] = clusId2
    print('Changed cluster Id %d to %d'%(clusId1, clusId2))
    clusims = getClusImStack(valPlt, nRows, nCols, clusids, colors)
    return valPlt, clusims, clusids

def mergeClus(i,j):
    """
    merge the clusters by selecting the culsters in the images clicked
    
    """
    global nClicks, clickedImId, valuesPlt, clusIms, clusIds
    if [i,j]!=allClusImloc:
        if nClicks<1:
            clickedImId.append([i,j])
            nClicks += 1
            print('select next cluster')
        elif nClicks>=1:
            clickedImId.append([i,j])
            valuesPlt, clusIms, clusIds = updateClusters(clickedImId, valuesPlt, clusIms, clusIds)
            print 'updated display Ims at', clickedImId[0], clickedImId[1]
            nClicks  = 0
            clickedImId = []
            updateImDisplay(clusIms)
    else:
        print ("selected all cluster Image, can't process this!!")

def resetVals():
    global clusIds, valuesPlt, clusIms
    clusIds = getClusIdNew(nRows, nCols)
    valuesPlt = valuesOriSlice.copy()
    clusIms = getClusImStack(valuesPlt, nRows, nCols, clusIds, colors)
    updateImDisplay(clusIms)

def getClusIdNew(nrows, ncols):
    clusIds = [[] for x in xrange(nrows) ]
    cluAdd = -1
    for x in xrange(nrows):
        for y in xrange(ncols):
            clusIds[x].append(cluAdd)
            cluAdd += 1
    return clusIds

def toggle1():
    '''
    '''
    global nClicks, clickedImId
    nClicks = 0
    clickedImId = []
    if varMerge.get()==1:
        varAssign.set(0)
        print 'Merging clusters'
    elif varMerge.get()==0:
        varAssign.set(1)
        print 'Assigning Legs to clusters'

def toggle2():
    '''
    '''
    global nClicks, clickedImId
    nClicks = 0
    clickedImId = []
    if varAssign.get()==1:
        varMerge.set(0)
        print 'Assigning Legs to clusters'
    if varAssign.get()==0:
        varMerge.set(1)
        print 'Merging clusters'

def clickAction(i,j):
    legAssignVar = varAssign.get()
    clusMergeVar = varMerge.get()
    if clusMergeVar==1 and legAssignVar==0:
        mergeClus(i, j)
    elif clusMergeVar==0 and legAssignVar==1:
        assignLeg(i,j)
    else:
        print 'Select the action via check box'

def assignLeg(i,j):
    global nClicks, clickedImId, valuesPlt, clusIds, clusIms
    if nClicks<1:
        if i>nRows:
            print 'Select proper image'
        else:
            clickedImId.append([i,j])
            nClicks += 1
            print('Select leg Id')
    elif nClicks>=1:
        x, y = i-nRows, j-1
        if x<0 or y<0:
            print 'select proper assignment Value box'
            print ('Selected rowId: %d, colId: %d'%i, j)
        else:
            clickedImId.append([i,j])
            if valuesPlt.shape[1]-valuesOri.shape[1]<1:
                valuesPlt = np.hstack((valuesPlt, np.zeros((valuesPlt.shape[0], 1))))
            for fr in xrange(len(valuesPlt)):
                if valuesPlt[fr, cusIdCol]==clusIds[clickedImId[0][0]-nRows][clickedImId[0][1]]:
                    valuesPlt[fr, -1] = legListIds[x][y]
            cv2.putText(clusIms[clickedImId[0][0]][clickedImId[0][1]], legList[x][y], (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255))
            updateImDisplay(clusIms)
            nClicks = 0
            clickedImId = []
            print ('Assigned to leg %s'%legList[x][y])

def checkLegAssignment():
    aa = (np.unique(valuesPlt[:, -1]))
    for i,x in enumerate(legListIds):
        for j, y in enumerate(x):
            if y in aa:
                print 'found', legList[i][j]
            else:
                print 'Select ', legList[i][j]

def skipTrack():
    global skipTrackVar
    skipTrackVar=1

def getCroppedIms(imList):
    imsAll = []
    for i, imId in enumerate(imList):
        imsAll.append(cv2.imread(imId, cv2.IMREAD_GRAYSCALE))
    return np.array(imsAll, dtype=np.uint8)

def fixFrNames(frNameList, dirname):
    return [os.path.join(dirname,os.sep.join(x.split(os.sep)[-4:])) for x in frNameList]

initDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/'
initDir = '/media/pointgrey/data/flywalk/legTracking/'
dirName = getFolder(initDir)

dirs = getDirList(dirName)

rawDirs = [os.path.join(d, imDataFolder) for d in dirs]
#rawDirs = [getDirList(d) for d in dirs]

legTipFNames = []
centroidFNames = []

for _,rawdir in enumerate(rawDirs):
    legTipFNames.extend(getFiles(rawdir+'/', ['*legTipLocs.csv']))
    centroidFNames.extend(getFiles(rawdir, ['*centroids.csv']))

sortedCentrds = []
for _,f in enumerate(legTipFNames):
    for _,x in enumerate(centroidFNames):
        if f.split(os.sep)[-1].split('legTipsClus')[0] in x:
            sortedCentrds.append(x)


#legTipFNames = getFiles(dirName, ['*legTipLocs.csv'])
#centroidFNames = getFiles(dirName, ['*centroids.csv'])
for iCsv, (legTipsCsv, centroidsCsv) in enumerate(zip(legTipFNames,sortedCentrds)):
    rows = getlegTipData(legTipsCsv)
    valuesOri = np.array(rows[:,1:], dtype=np.float64)
    valuesPlt = valuesOri.copy()
    
    cents1 = getCentroidData(centroidsCsv)
    cents = [x for _,x in enumerate(cents1) if x[1]!='noContourDetected']
    frNamesAll = [x[0] for _,x in enumerate(cents) if x[1]!='noContourDetected']
    centroids = np.array(cents)[:,1:].astype(dtype=np.float64)
    frNamesAll = fixFrNames(frNamesAll, dirName)
    frNamesTotal = copy.copy(frNamesAll)
    pool = mp.Pool(nThreads)
    frameIdx = []
    windowName = legTipsCsv.split(dirName)[1].split('_legTipsClus')[0]
    selectTrackFrames(frNamesAll, centroids, pool)
    print frameIdx
    pool.close()
    for nFrameIdx,frNumb in enumerate(frameIdx):
        frNamesSlice  = frNamesAll[frNumb[0]:frNumb[1]]
        centSlice = centroids[frNumb[0]:frNumb[1], :]
        allIms = getCroppedIms(frNamesSlice)
        print len(allIms)
        frHeight, frWidth = allIms[0].shape
        selIms = []
        for i,f in enumerate(centSlice):
            try:
                cent = np.array(f).astype(dtype=np.float64)#centroids[idx].astype(np.int)
                ptW, ptH = int(cent[0]), int(cent[1])
                if w<ptW<frWidth-w and h<ptH<frHeight-h :
                    selIms.append(allIms[i][ptH-h:ptH+h, ptW-w:ptW+w])
            except:
                pass
        selIms = np.array(selIms, dtype=np.uint8)
        #displayImgs(selIms, 20)
        frNames = rows[:,0]
        vals = []
        for _, fId in enumerate(frNamesSlice):
            idx = np.where(frNames==fId)
            for _, i in enumerate(idx):
                vals.append(valuesOri[i])
        
        valuesOriSlice = np.vstack(vals)
        valuesPlt = valuesOriSlice.copy()
        nClusters = len(np.unique(valuesPlt[:, cusIdCol]))
        nRows = (nClusters/nCols)+1
        clusIds = getClusIdNew(nRows, nCols)
        clusIms = getClusImStack(valuesPlt, nRows, nCols, clusIds, colors)
    
        print (np.unique(valuesPlt[:, cusIdCol]))
        nClicks = 0
        clickedImId = []
        window = tk.Tk()
        btn_matrix = []
        legBtn_matrix = []
        for row in range(nRows):
            row_matrix = []
            for col in range(nCols):
                img = ImageTk.PhotoImage(Image.fromarray(clusIms[row][col]))
                btn = tk.Button(window, text = '(%d, %d)' % (row, col),image=img, 
                                command = lambda x = row, y = col: clickAction(x, y))
                btn.grid(row = row, column = col)
                btn.image = img
                row_matrix.append(btn)
            btn_matrix.append(row_matrix)
        for row in range(nRows, nRows+nlegRows):
            legRow_matrix = []
            for col in range(1,nlegCols+1):
                legBtn = tk.Button(window, text = ('Assign to %s')%legList[row-nRows][col-1],\
                                    command = lambda x=row, y=col: clickAction(x,y))
                legBtn.grid(row = row, column = col)
                legRow_matrix.append(btn)
            legBtn_matrix.append(legRow_matrix)
        rstBtn = tk.Button(window, text = 'Reset',bg='yellow', command = resetVals).grid(row=nRows+3, column=0)
        varMerge = tk.IntVar()
        varMerge.set(0)
        c1 = tk.Checkbutton(window, text="Merge Clusters", variable=varMerge, command = toggle1).grid(row=nRows+3, column=2)
        varAssign = tk.IntVar()
        varAssign.set(0)
        c2 = tk.Checkbutton(window, text="Assign Clusters to legs", variable=varAssign, command = toggle2).grid(row=nRows+3, column=3)
        saveBtn = tk.Button(window, text = 'Save',bg='green', command = checkLegAssignment).grid(row=nRows+4, column=0)
        skipTrackVar = 0
        skipBtn = tk.Button(window, text = 'Skip this track!!', bg='red',command = skipTrack).grid(row=nRows+4, column=1)
        window.mainloop()
        print (np.unique(valuesPlt[:, cusIdCol]))

        if skipTrackVar==0:
            legIdList = legListIds[0]+legListIds[1]
            legFrList = []
            for i, legId in enumerate(legIdList):
                legFrList.append(np.where(valuesPlt[:,-1]==legId)[0])
            legs = []
            for i,legId in enumerate(legFrList):
                lData = []
                for _, val in enumerate(legId):
                    lData.append(valuesPlt[val])
                legs.append(np.array(lData))
            
            nFrames= np.unique(valuesPlt[:,0])
            legTipsLocsAsgnd = np.zeros((nLegs, len(nFrames)+1, 3))
            for i, legData in enumerate(legs):
                for j, data in enumerate(legData):
                    legTipsLocsAsgnd[i, np.where(nFrames==int(data[0])), 0] = data[0]
                    legTipsLocsAsgnd[i, np.where(nFrames==int(data[0])), 1:] = data[2:4]
            
            #-- automatically update the missing points in the legs based on previous and next location of the legTips
            zeroId = 0
            for leg in xrange(legTipsLocsAsgnd.shape[0]):
                zeroId = 0
                for i in xrange(legTipsLocsAsgnd.shape[1]):
                    if i>0:
                        if legTipsLocsAsgnd[leg,i,0]==0:
                            zeroId +=1
                        else:
                            if zeroId>0:
                                startZeroPt = i-zeroId 
                                if zeroId>2:
                                    legTipsLocsAsgnd[leg,startZeroPt:i,1:] = intermediates(legTipsLocsAsgnd[leg,startZeroPt-1,1:], legTipsLocsAsgnd[leg,i,1:], zeroId).astype(dtype=np.int)
                                else:
                                    legTipsLocsAsgnd[leg,startZeroPt,1:] = legTipsLocsAsgnd[leg,i,1:]
                                    legTipsLocsAsgnd[leg,startZeroPt:i,1:] = legTipsLocsAsgnd[leg,i,1:]
                                legTipsLocsAsgnd[leg,startZeroPt:i,0] = np.arange(startZeroPt,i)
                                zeroId=0
            saveDir = '-'.join(centroidsCsv.split('-')[:-1])+'_'+str(nFrameIdx)+'/tmp/'
            try:
                os.makedirs(saveDir)
                os.makedirs(saveDir+'legs')
                os.makedirs(saveDir+'labelled')
            except:
                pass
            
            centSeq = np.arange(len(centSlice))
            centSeq = centSeq.reshape((len(centSeq),1))
            centSlice = np.hstack((centSeq, centSlice, centSeq))
            legNameList = legList[0]+ legList[1]
            csvOutFile = saveDir+'fly_free.tsv'
            with open(csvOutFile, "wb") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['', 'X', 'Y', 'Major', 'Minor', 'Angle', 'Slice'])
                writer.writerows(centSlice)
            for leg in xrange(legTipsLocsAsgnd.shape[0]):
                csvOutFile = saveDir+'legs/'+legNameList[leg]+'.tsv'
                with open(csvOutFile, "wb") as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([legNameList[leg], 'X','Y'])
                    writer.writerows(legTipsLocsAsgnd[leg])
            for i in xrange(len(selIms)):
                ims = cv2.cvtColor(selIms[i], cv2.COLOR_GRAY2BGR)
                for l in xrange(nLegs):
                    cv2.circle(ims,(int(legTipsLocsAsgnd[l,i,1]),int(legTipsLocsAsgnd[l,i,2])), 2, colors[l], 2 )
                cv2.imwrite(saveDir+'f'+str(i)+'.jpeg', ims)
                cv2.imwrite(saveDir+'labelled/f'+str(i)+'.jpeg', ims)
                cv2.imshow('123', ims)
                cv2.waitKey(100)
            cv2.destroyAllWindows()
    else:
        print 'Track skipped'












#--------


#-------- fix cluster code ------
#allIms = getCroppedIms(frNamesAll)
#frHeight, frWidth = allIms[0].shape
#frNames = np.unique(rows[:,0])
#selIms = []
#centroids = []
#trackedFrNames = []
#for i,f in enumerate(frNames):
#    try:
#        idx = frNamesAll.index(f)
#        cent = np.array(cents[idx][1:]).astype(dtype=np.float64)#centroids[idx].astype(np.int)
#        ptW, ptH = int(cent[0]), int(cent[1])
#        if w<ptW<frWidth-w and h<ptH<frHeight-h :
#            selIms.append(allIms[idx][ptH-h:ptH+h, ptW-w:ptW+w])
#            centroids.append(cent)
#            trackedFrNames.append(f)
#    except:
#        pass
#frNames = rows[:,0]
#vals = []
#for _, fId in enumerate(trackedFrNames):
#    idx = np.where(frNames==fId)
#    for _, i in enumerate(idx):
#        vals.append(valuesOri[i])
#
#valuesOri = np.vstack(vals)
#valuesPlt = valuesOri.copy()
#nClusters = len(np.unique(valuesPlt[:, cusIdCol]))
#nRows = (nClusters/nCols)+1
#clusIds = getClusIdNew(nRows, nCols)
#clusIms = getClusImStack(valuesPlt, nRows, nCols, clusIds, colors)
#
#print (np.unique(valuesPlt[:, cusIdCol]))
#nClicks = 0
#clickedImId = []
#window = tk.Tk()
#btn_matrix = []
#legBtn_matrix = []
#for row in range(nRows):
#    row_matrix = []
#    for col in range(nCols):
#        img = ImageTk.PhotoImage(Image.fromarray(clusIms[row][col]))
#        btn = tk.Button(window, text = '(%d, %d)' % (row, col),image=img, 
#                        command = lambda x = row, y = col: clickAction(x, y))
#        btn.grid(row = row, column = col)
#        btn.image = img
#        row_matrix.append(btn)
#    btn_matrix.append(row_matrix)
#for row in range(nRows, nRows+nlegRows):
#    legRow_matrix = []
#    for col in range(1,nlegCols+1):
#        legBtn = tk.Button(window, text = ('Assign to %s')%legList[row-nRows][col-1],\
#                            command = lambda x=row, y=col: clickAction(x,y))
#        legBtn.grid(row = row, column = col)
#        legRow_matrix.append(btn)
#    legBtn_matrix.append(legRow_matrix)
#rstBtn = tk.Button(window, text = 'Reset',command = resetVals).grid(row=nRows+3, column=0)
#varMerge = tk.IntVar()
#varMerge.set(0)
#c1 = tk.Checkbutton(window, text="Merge Clusters", variable=varMerge, command = toggle1).grid(row=nRows+3, column=2)
#varAssign = tk.IntVar()
#varAssign.set(0)
#c2 = tk.Checkbutton(window, text="Assign Clusters to legs", variable=varAssign, command = toggle2).grid(row=nRows+3, column=3)
#saveBtn = tk.Button(window, text = 'Save',command = checkLegAssignment).grid(row=nRows+4, column=0)
#window.mainloop()
#print (np.unique(valuesPlt[:, cusIdCol]))
#
#
#legIdList = legListIds[0]+legListIds[1]
#legFrList = []
#for i, legId in enumerate(legIdList):
#    print legId
#    legFrList.append(np.where(valuesPlt[:,-1]==legId)[0])
#
#legs = []
#for i,legId in enumerate(legFrList):
#    lData = []
#    for _, val in enumerate(legId):
#        lData.append(valuesPlt[val])
#    legs.append(np.array(lData))
#
#nFrames= np.unique(valuesPlt[:,0])
#
#legTipsLocsAsgnd = np.zeros((nLegs, len(nFrames)+1, 3))
#
#for i, legData in enumerate(legs):
#    for j, data in enumerate(legData):
#        legTipsLocsAsgnd[i, np.where(nFrames==int(data[0])), 0] = data[0]
#        legTipsLocsAsgnd[i, np.where(nFrames==int(data[0])), 1:] = data[2:4]
#
##-- automatically update the missing points in the legs based on previous and next location of the legTips
#zeroId = 0
#for leg in xrange(legTipsLocsAsgnd.shape[0]):
#    zeroId = 0
#    for i in xrange(legTipsLocsAsgnd.shape[1]):
#        if i>0:
#            if legTipsLocsAsgnd[leg,i,0]==0:
#                zeroId +=1
#            else:
#                if zeroId>0:
#                    startZeroPt = i-zeroId 
#                    if zeroId>2:
#                        legTipsLocsAsgnd[leg,startZeroPt:i,1:] = intermediates(legTipsLocsAsgnd[leg,startZeroPt-1,1:], legTipsLocsAsgnd[leg,i,1:], zeroId).astype(dtype=np.int)
#                    else:
#                        legTipsLocsAsgnd[leg,startZeroPt,1:] = legTipsLocsAsgnd[leg,i,1:]
#                        legTipsLocsAsgnd[leg,startZeroPt:i,1:] = legTipsLocsAsgnd[leg,i,1:]
#                    legTipsLocsAsgnd[leg,startZeroPt:i,0] = np.arange(startZeroPt,i)
#                    zeroId=0

#centroids = np.array(centroids)
#centSeq = np.arange(len(centroids))
#centSeq = centSeq.reshape((len(centSeq),1))
#centroids = np.hstack((centSeq, centroids, centSeq))
#legNameList = legList[0]+ legList[1]
#csvOutFile = dirName+'/tmp/fly_free.tsv'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f, delimiter='\t')
#    writer.writerow(['', 'X', 'Y', 'Major', 'Minor', 'Angle', 'Slice'])
#    writer.writerows(centroids)
#for leg in xrange(legTipsLocsAsgnd.shape[0]):
#    csvOutFile = dirName+'/tmp/legs/'+legNameList[leg]+'.tsv'
#    with open(csvOutFile, "wb") as f:
#        writer = csv.writer(f, delimiter='\t')
#        writer.writerow([legNameList[leg], 'X','Y'])
#        writer.writerows(legTipsLocsAsgnd[leg])
#for i,ims in enumerate(selIms):
#    cv2.imwrite(dirName+'/tmp/f'+str(i)+'.jpeg', ims)
#    cv2.imwrite(dirName+'/tmp/labelled/f'+str(i)+'.jpeg', ims)
##
#
#for i in xrange(len(selIms)):
#    im = cv2.cvtColor(selIms[i], cv2.COLOR_GRAY2BGR)
#    for l in xrange(nLegs):
#        cv2.circle(im,(int(legTipsLocsAsgnd[l,i,1]),int(legTipsLocsAsgnd[l,i,2])), 2, colors[l], 2 )
#    cv2.imshow('123', im)
#    cv2.waitKey(100)
#cv2.destroyAllWindows()
#









