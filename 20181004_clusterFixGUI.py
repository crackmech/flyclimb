#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:10:16 2018

@author: aman
"""

import Tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import csv
import cv2
import copy
import random


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



def getClusImStack(xyValArr, nrows, ncols, clusids, clrs):
    clusims = [[np.zeros((2*h,2*w,3), dtype = np.uint8) for x in xrange(ncols)]\
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
    valuesPlt = valuesOri.copy()
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


def getCroppedIms(imList):
    imsAll = []
    for i, imId in enumerate(imList):
        imsAll.append(cv2.imread(imId, cv2.IMREAD_GRAYSCALE))
    return imsAll

dirName = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'
o = '013222'
legTipsCsvName = dirName+'20180822_'+o+'_legTipsClus_n20-Climbing_legTipLocs.csv'
centroidsFile = dirName+'20180822_'+o+'_legTipsClus_n20-Climbing_centroids.csv'

rows = getlegTipData(legTipsCsvName)
valuesOri = np.array(rows[:,1:], dtype=np.float64)
valuesPlt = valuesOri.copy()

cents = getCentroidData(centroidsFile)
cents = [x for _,x in enumerate(cents) if x[1]!='noContourDetected']
frNamesAll = [x[0] for _,x in enumerate(cents) if x[1]!='noContourDetected']
centroids = np.array(cents)[:,1:].astype(dtype=np.float64)
allIms = getCroppedIms(frNamesAll)
frHeight, frWidth = allIms[0].shape
frNames = np.unique(rows[:,0])
selIms = []
centroids = []
trackedFrNames = []
for i,f in enumerate(frNames):
    try:
        idx = frNamesAll.index(f)
        cent = np.array(cents[idx][1:]).astype(dtype=np.float64)#centroids[idx].astype(np.int)
        ptW, ptH = int(cent[0]), int(cent[1])
        if w<ptW<frWidth-w and h<ptH<frHeight-h :
            selIms.append(allIms[idx][ptH-h:ptH+h, ptW-w:ptW+w])
            centroids.append(cent)
            trackedFrNames.append(f)
    except:
        pass
frNames = rows[:,0]
vals = []
for _, fId in enumerate(trackedFrNames):
    idx = np.where(frNames==fId)
    for _, i in enumerate(idx):
        vals.append(valuesOri[i])

valuesOri = np.vstack(vals)
valuesPlt = valuesOri.copy()
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
rstBtn = tk.Button(window, text = 'Reset',command = resetVals).grid(row=nRows+3, column=0)
varMerge = tk.IntVar()
varMerge.set(0)
c1 = tk.Checkbutton(window, text="Merge Clusters", variable=varMerge, command = toggle1).grid(row=nRows+3, column=2)
varAssign = tk.IntVar()
varAssign.set(0)
c2 = tk.Checkbutton(window, text="Assign Clusters to legs", variable=varAssign, command = toggle2).grid(row=nRows+3, column=3)
saveBtn = tk.Button(window, text = 'Save',command = checkLegAssignment).grid(row=nRows+4, column=0)
window.mainloop()
print (np.unique(valuesPlt[:, cusIdCol]))


legIdList = legListIds[0]+legListIds[1]
legFrList = []
for i, legId in enumerate(legIdList):
    print legId
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

centroids = np.array(centroids)
centSeq = np.arange(len(centroids))
centSeq = centSeq.reshape((len(centSeq),1))
centroids = np.hstack((centSeq, centroids, centSeq))
legNameList = legList[0]+ legList[1]
csvOutFile = dirName+'/tmp/fly_free.tsv'
with open(csvOutFile, "wb") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['', 'X', 'Y', 'Major', 'Minor', 'Angle', 'Slice'])
    writer.writerows(centroids)
for leg in xrange(legTipsLocsAsgnd.shape[0]):
    csvOutFile = dirName+'/tmp/legs/'+legNameList[leg]+'.tsv'
    with open(csvOutFile, "wb") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([legNameList[leg], 'X','Y'])
        writer.writerows(legTipsLocsAsgnd[leg])
for i,ims in enumerate(selIms):
    cv2.imwrite(dirName+'/tmp/f'+str(i)+'.jpeg', ims)
    cv2.imwrite(dirName+'/tmp/labelled/f'+str(i)+'.jpeg', ims)
#

for i in xrange(len(selIms)):
    im = cv2.cvtColor(selIms[i], cv2.COLOR_GRAY2BGR)
    for l in xrange(nLegs):
        cv2.circle(im,(int(legTipsLocsAsgnd[l,i,1]),int(legTipsLocsAsgnd[l,i,2])), 2, colors[l], 2 )
    cv2.imshow('123', im)
    cv2.waitKey(100)
cv2.destroyAllWindows()



#for i in xrange(nRows):
#    for j in xrange(nCols):
#        cv2.imshow('123', clusIms[i][j])
#        cv2.waitKey(0)
#cv2.destroyAllWindows()


#window = tk.Tk()
#btn_matrix = []
#for row in range(nRows):
#    row_matrix = []
#    for col in range(nCols):
#        img = ImageTk.PhotoImage(Image.fromarray(clusIms[row][col]))
#        btn = tk.Button(window, text = '(%d, %d)' % (row, col),image=img, 
#                        command = lambda x = row, y = col: mergeClus(x, y))
#        btn.grid(row = row, column = col)
#        btn.image = img
#        row_matrix.append(btn)
#    btn_matrix.append(row_matrix)
#rstBtn = tk.Button(window, text = 'Reset',command = resetVals).grid(row=nRows+1, column=0)
#window.mainloop()







