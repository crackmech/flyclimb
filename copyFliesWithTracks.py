#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:42:50 2018

@author: pointgrey

script for copying tracking files from the climbing behaviour tracked folders into a new folder 'csvDir'
keeping the same directory structure.
"""

import os
from shutil import copyfile
import shutil
import Tkinter as tk
import tkFileDialog as tkd
from datetime import datetime
from time import time
import re

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

def files(rootdir='.', mindepth=0, maxdepth=float('inf')):
    root_depth = rootdir.rstrip(os.path.sep).count(os.path.sep) - 1
    for dirpath, dirs, files in os.walk(rootdir):
        depth = dirpath.count(os.path.sep) - root_depth
        if mindepth <= depth <= maxdepth:
            for filename in files:
                yield [dirpath, filename]
        elif depth > maxdepth:
            del dirs[:] # too deep, don't recurse

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

initDir = '/media/aman/easystore/data_legTracking/'
#initDir = '/media/aman/data/flyWalk_data/delete/'
imDir = 'imageData'
trackDir = '_legTipsClus_'
copyDir = 'tracksFolders'
trackedFlyDir = 'tracks'
notTrackedFlyDir = 'noTracks'
#dirname = initDir
dirname = getFolder(initDir)
print('sorting flies with tracks and without tracks')
fList = files(dirname, maxdepth=4)

startTime = time()

dirs = getDirList(dirname)


for _,d in enumerate(dirs):
    rawDirs = getDirList(d)
    for _, rawDir in enumerate(rawDirs):
        try:
            imDirs = getDirList(os.path.join(rawDir, imDir))
            trackedDirs = [x for x in imDirs if trackDir in x]
            if trackedDirs !=[]:
                #print trackedDirs
                #print '==>',rawDir, 'Gait tracked'
                #shutil.move(rawDir, os.path.join(d,trackedFlyDir))
                for _,x in enumerate(trackedDirs):
#                    print x
#                    print os.path.join(d,copyDir)
                    copytree(x, os.path.join(d,copyDir, os.sep.join(x.split(os.sep)[-3:])) )
            else:
                print rawDir
                #print '---', rawDir, 'Gait NOT tracked'
                #shutil.move(rawDir, os.path.join(d,notTrackedFlyDir))
        except:
            pass


#for i,f in enumerate(fList):
#    if 'roi' not in f[0]:
#        d = f[0].split(dirname)[-1]
#        try:
#            os.makedirs(os.path.join(dirname, csvDirName, d))
#        except OSError:
#            pass
#        d1 = os.path.join(dirname, csvDirName, d, f[1])
#        copyfile( os.path.join(f[0],f[1]), d1)
#
#print('Finished copying csv files at: %s'%present_time())
#print ('total time taken for %d files: %s Seconds'%((i+1,time()-startTime)))


























