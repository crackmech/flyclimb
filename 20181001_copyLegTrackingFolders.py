#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 07:08:53 2018

@author: pointgrey
"""
import os
import glob
import numpy as np
import re
import sys
import Tkinter as tk
import tkFileDialog as tkd
from datetime import datetime
from time import time
import shutil
import stat

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

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def getDirList(parentDir):
    return natural_sort([ os.path.join(parentDir, name) for name in os.listdir(parentDir)\
                        if os.path.isdir(os.path.join(parentDir, name)) ])


def copytree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)




initDir = '/media/pointgrey/data/flywalk/legTracking/'
tracksExt = ['*.jpeg']
trackExt = 'jpeg'
imDataFolder = 'imageData'
camFile = 'camloop.txt'

tracksBaseDir = getFolder(initDir)
imDataBaseDir = getFolder(initDir)

outDir = os.path.join(tracksBaseDir,'../copiedLegTrackingTrackData/')
tracksDirs = getDirList(tracksBaseDir)

startTime = (present_time(), time())
print('Started at %s'%(startTime[0]))
for i,trackDir in enumerate(tracksDirs):
    tracksList = natural_sort(getFiles(trackDir, tracksExt))
    for j,track in enumerate(tracksList):
        trackDetails = track.split('/')[-1].split('.')[0].split('_trackData_')
        imDataDir = os.path.join(trackDetails[-1], imDataFolder, trackDetails[0])
        imDir = os.path.join(imDataBaseDir, trackDir.split(os.sep)[-1], imDataDir)
        print('%d folders left to copy'%(len(tracksList)-j))
        copytree(imDir, os.path.join(outDir, imDataDir))
        trackCopyDir = os.path.join(trackDetails[-1], imDataFolder)
        shutil.copy(track, os.path.join(outDir, trackCopyDir))
        shutil.copy(os.path.join(imDataBaseDir, trackDir.split(os.sep)[-1], trackDetails[-1], camFile),\
                    os.path.join(outDir, trackDetails[-1]))
        
print('Finished at %s, in %sSeconds'%(startTime[0], time()-startTime[1]))







































