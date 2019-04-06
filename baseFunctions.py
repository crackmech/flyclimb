#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:18:03 2018

@author: aman
"""
import numpy as np
import os
import glob
import re
import random
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import csv

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

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    return rows

def random_color():
    levels = [x/255.0 for x in range(32,256,32)]
    return tuple(random.choice(levels) for _ in range(3))

def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]
      


sWidth = 0.15           #0.012
sSize = 5              #5 for 5 minutes, 300 for 30 minutes
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0#0.2
sEdgCol = None#(0,0,0)
scatterDataWidth = 0.012
sCol = (1,1,1)

def plotScatter(axis, data, scatterX, scatterWidth = sWidth, \
                scatterRadius = sSize , scatterColor = sCol,\
                scatterMarker = sMarker, scatterAlpha = sAlpha, \
                scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol, zOrder=0):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot
    '''
    return axis.scatter(np.linspace(-scatterWidth+scatterX, scatterWidth+scatterX, len(data)), data,\
            s=scatterRadius, color = scatterColor, marker=scatterMarker,\
            alpha=scatterAlpha, linewidths=scatterLineWidth, edgecolors=scatterEdgeColor, zorder=zOrder )


def plotScatterCentrd(axis, dataIn, scatterX, scatterWidth = sWidth, \
                      scatterRadius = sSize , scatterColor = sCol, \
                      scatterMarker = sMarker, scatterAlpha = sAlpha, \
                      scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol, zOrder=0):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot, where scatter is distributed by a histogram
    '''
    data1 = np.array(dataIn).copy()
    histData = np.histogram(data1, 20)  # generate histogramw ith binSize 20
    spaceMulti = 2
    space = (histData[0].max()*spaceMulti)+1
    xValues = np.linspace(-scatterWidth+scatterX, scatterWidth+scatterX, space)
    pltDta = np.zeros((space))*np.nan
    for i in xrange(1,len(histData[0])):
        dataIds = np.where(np.logical_and(histData[1][i-1]<=data1, data1<histData[1][i]))[0]
        dataSlice = [dataIn[x] for i_,x in enumerate(dataIds)]
        spaceDiff = space - len(dataSlice)
        if spaceDiff>0:
            pltDta[spaceDiff/spaceMulti:(spaceDiff/spaceMulti)+len(dataSlice)] = dataSlice
        if type(scatterColor)==list:
            sColor = []
            for i_,x in enumerate(pltDta):
                if np.isnan(x):
                    sColor.append((0,0,0,0))
                else:
                    colIdx = np.where(dataIn==x)[0]
                    sColor.append(scatterColor[colIdx[0]])
            #sColor = [scatterColor[x] for i_,x in enumerate(dataIds)]
            lenDiff =  len(sColor)-len(pltDta)
            if lenDiff!=0:
                print i,'Diff in color and data lengths:',lenDiff
        plot = axis.scatter(xValues, pltDta,
                        s=scatterRadius, color = sColor, marker=scatterMarker,alpha=scatterAlpha,
                        linewidths=scatterLineWidth, edgecolors=scatterEdgeColor, zorder=zOrder )
    return plot







