#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:05:26 2019

@author: aman
"""

import os
import re
from datetime import datetime

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFigFolderList(figDataDir, outFile, figFolderList):
    with open(outFile, 'w') as f:
        for dirs in getDirList(figDataDir):
            print dirs.split(os.sep)[-1] 
            if dirs.split(os.sep)[-1] in figFolderList:
                f.write('----------------\n')
                for dataDir in getDirList(dirs):
                    for d in getDirList(dataDir):
                        f.write((os.sep).join(d.split(os.sep)[-3:])+'\n')

def readFigFolderFile(figFolderFName, figFolderList):
    figFoldersDict = {}
    with open(figFolderFName, 'r') as f:
        lines = f.readlines()
    for figFold in figFolderList:
        figFoldersDict[figFold] = [line for line in lines if figFold in line]
    return figFoldersDict


baseDir = '/media/aman/data/flyWalk_data/climbingData/figRawCsvData/'
figFolders = ['fig2', 'fig3', 'fig4', 'fig5']

figFolderFName = baseDir+'figDataFiles'+present_time()+'.txt'
getFigFolderList(baseDir, figFolderFName, figFolders)

figFolders1 = readFigFolderFile(figFolderFName, figFolders)




