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
import Tkinter as tk
import tkFileDialog as tkd
from datetime import datetime
from time import time

def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def files(rootdir='.', mindepth=0, maxdepth=float('inf')):
    root_depth = rootdir.rstrip(os.path.sep).count(os.path.sep) - 1
    for dirpath, dirs, files in os.walk(rootdir):
        depth = dirpath.count(os.path.sep) - root_depth
        if mindepth <= depth <= maxdepth:
            for filename in files:
                yield [dirpath, filename]
        elif depth > maxdepth:
            del dirs[:] # too deep, don't recurse

initDir = '/media/pointgrey/data/flywalk/'
csvDirName = 'csvDir'

#dirname = initDir
dirname = getFolder(initDir)
print('copying csv files from %s'%dirname)
fList = files(dirname, maxdepth=4)

#flist = [x for x in fList]
print('Started copying csv files at: %s'%present_time())
startTime = time()
for i,f in enumerate(fList):
    if 'roi' not in f[0]:
        d = f[0].split(dirname)[-1]
        try:
            os.makedirs(os.path.join(dirname, csvDirName, d))
        except OSError:
            pass
        d1 = os.path.join(dirname, csvDirName, d, f[1])
        copyfile( os.path.join(f[0],f[1]), d1)

print('Finished copying csv files at: %s'%present_time())
print ('total time taken for %d files: %s Seconds'%((i+1,time()-startTime)))


































