#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:22:56 2018

@author: aman
"""
import cv2
import os
import glob
import numpy as np
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
# import random
# import matplotlib.pyplot as plt
# from scipy import stats
# import xlwt
# import dip
# import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.markers import MarkerStyle
import shutil
 
def move(src, dest):
    shutil.move(src, dest)

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

baseDir = '/media/aman/data/flyWalk_data/climbingData/controls'
#baseDir = '/media/pointgrey/data/flywalk/20180104/'

baseDir = getFolder(baseDir)
dirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])


def nothing(x):
    pass

def changeName(x, newname):
    global rawDir
    newname = '_'+newname
    if x==0:
        if 'male' in rawDir:
            move(rawDir, rawDir.rstrip('_female'))
            rawDir = rawDir.rstrip('_female')
    if x==1:
        move(rawDir, rawDir+newname)
        rawDir+=newname
    print x, rawDir
    
def renameMale(x):
    changeName(x, 'male')

def renameFemale(x):
    changeName(x, 'female')

def selectSex(animalDir):
    
    '''
    Displays the images from the 'roi' folder of the animalDir.
    By visual selection, we can determine if the animal is male or female.
    Then, using slider we can append the determined sex of the animal in the animalDir
    '''
    imList = natural_sort(glob.glob(os.path.join(animalDir, 'roi', '*.jpeg')))
    imgs = []
    for im in xrange(len(imList)):
        imgs.append(cv2.imread(imList[im]))
    #print animalDir, len(imList)
    windowName = 'trackImage'+animalDir.rsplit('/')[-1]
    cv2.namedWindow(windowName)
    cv2.createTrackbar('imageNumber', windowName, 0, (len(imList)-2), nothing)
    cv2.createTrackbar('label Male', windowName, 0, 1, renameMale)
    cv2.createTrackbar('label Female', windowName, 0, 1, renameFemale)
    cv2.setTrackbarPos('imageNumber', windowName, 0)
    cv2.setTrackbarPos('label Male', windowName, 0)
    cv2.setTrackbarPos('label Female', windowName, 0)
    while (1):
        imgNum = cv2.getTrackbarPos('imageNumber', windowName)
        cv2.imshow(windowName,imgs[imgNum] )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyWindow(windowName)
    return animalDir

def displayIm(genotypeDir):
    '''
    returns data of all Csv's of all tracks for all fly folders in a given folder (genotypeDir)
    
    '''
    global rawDir, imList
    rawdirs = natural_sort([ os.path.join(genotypeDir, name) for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    for i, rawDir in enumerate(rawdirs):
        if 'male' not in rawDir:
            rawDir = selectSex(rawDir)
        print('Sexual Identification done')


for _,d in enumerate(dirs):
    path = os.path.join(baseDir, d)
    displayIm(path)


def renameMale_(x):
    global rawDir
    if x==0:
        if 'male' in rawDir:
            move(rawDir, rawDir.rstrip('_female'))
            rawDir = rawDir.rstrip('_female')
    if x==1:
        move(rawDir, rawDir+'_male')
        rawDir+='_male'
    print x, rawDir

def renameFemale_(x):
    global rawDir
    if x==0:
        if 'male' in rawDir:
            move(rawDir, rawDir.rstrip('_female'))
            rawDir = rawDir.rstrip('_female')
    if x==1:
        move(rawDir, rawDir+'_female')
        rawDir+='_female'
    print x, rawDir

def displayIm_(genotypeDir):
    '''
    returns data of all Csv's of all tracks for all fly folders in a given folder (genotypeDir)
    
    '''
    global rawDir, imList
    rawdirs = natural_sort([ os.path.join(genotypeDir, name) for name in os.listdir(genotypeDir) if os.path.isdir(os.path.join(genotypeDir, name)) ])
    for i, rawDir in enumerate(rawdirs):
        # imdir = os.path.join(rawDir, 'roi')
        imList = natural_sort(glob.glob(os.path.join(rawDir, 'roi', '*.jpeg')))
        imgs = []
        for im in xrange(len(imList)):
            imgs.append(cv2.imread(imList[im]))
        # print imdir, imList[0], len(imList)
        windowName = 'trackImage'+rawDir.rsplit('/')[-1]
        cv2.namedWindow(windowName)
        cv2.createTrackbar('imageNumber', windowName, 0, (len(imList)-2), nothing)
        cv2.createTrackbar('label Male', windowName, 0, 1, renameMale)
        cv2.createTrackbar('label Female', windowName, 0, 1, renameFemale)
        cv2.setTrackbarPos('imageNumber', windowName, 0)
        cv2.setTrackbarPos('label Male', windowName, 0)
        cv2.setTrackbarPos('label Female', windowName, 0)
        while (1):
            imgNum = cv2.getTrackbarPos('imageNumber', windowName)
            cv2.imshow(windowName,imgs[imgNum] )
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyWindow(windowName)













