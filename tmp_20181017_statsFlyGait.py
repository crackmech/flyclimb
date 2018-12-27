#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 01:22:47 2018

@author: aman
"""

import scikit_posthocs as sp
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.anova import AnovaRM
import pandas as pd

import numpy as np
import os
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import glob
import random
import csv

import matplotlib.pyplot as plt




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

#colors = [random_color() for i in xrange(20)]
def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row)
    return rows
    
def readConcCsv(ConcCsvFname):
    rows = []
    with open(ConcCsvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append([x.strip('[]') for _,x in enumerate(row)])
    return rows
    
def restrucDataForStats(dataSet, dataSetLabels):
    restrucData = [[[] for y in dataSetLabels] for x in dataSet[0][0]]
    
    maxLen = max([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(1, maxLen):
            try:
                temp_data = dataSet[i][j]
            except:
                temp_data = ''
            for k,d in enumerate(temp_data):
                if d!='':
                    restrucData[k][i].append(np.float(d))
    return restrucData

def restrucDataForRMA(dataSet, dataSetLabels):
    restrucData = [[[] for y in dataSetLabels] for x in dataSet[0][0]]
    
    maxLen = min([len(x) for x in dataSet])
    for i in xrange(len(dataSet)):
        for j in xrange(1, maxLen):
            try:
                temp_data = dataSet[i][j]
            except:
                temp_data = ''
            for k,d in enumerate(temp_data):
                if d!='':
                    restrucData[k][i].append(np.float(d))
    return restrucData

def getKWmultiComp(data, labels, verbose=False):
    pVals = sp.posthoc_dunn(data, p_adjust='bonferroni')
    if verbose:
        print np.hstack((np.transpose([0]+labels).reshape(4,1),np.vstack((labels,pVals))))
    return [pVals[1,0], pVals[2,0], pVals[2,1]]

def getOWANOVAmultiComp(data, labels, verbose=False):
    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(data)])
    res = pairwise_tukeyhsd(np.concatenate(data), tlabels)
    if verbose:
        print (res.summary())
    return psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total)
    

ctrl = 'W1118_'
exp1 = 'W1118xLrrk-ex1'
exp2 = 'Park25xLrrk-ex1'

dataSets = [ctrl,exp1,exp2]
 
baseDir = '/media/aman/data/flyWalk_data/climbingData/gait/allData/copied/analyzed/'

paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']

csvsCtrl = getFiles(baseDir, [ctrl+'*.csv'])
csvsExp1 = getFiles(baseDir, [exp1+'*.csv'])
csvsExp2 = getFiles(baseDir, [exp2+'*.csv'])

csvs1Ctrl = [x for _,x in enumerate(csvsCtrl) if 'CONCURRENCY' not in x]
csvs1Exp1 = [x for _,x in enumerate(csvsExp1) if 'CONCURRENCY' not in x]
csvs1Exp2 = [x for _,x in enumerate(csvsExp2) if 'CONCURRENCY' not in x]

concCsvCtrl = [x for _,x in enumerate(csvsCtrl) if 'CONCURRENCY' in x]
concCsvExp1 = [x for _,x in enumerate(csvsExp1) if 'CONCURRENCY' in x]
concCsvExp2 = [x for _,x in enumerate(csvsExp2) if 'CONCURRENCY' in x]


dataCtrl = [readCsv(x) for _,x in enumerate(csvs1Ctrl)]
dataExp1 = [readCsv(x) for _,x in enumerate(csvs1Exp1)]
dataExp2 = [readCsv(x) for _,x in enumerate(csvs1Exp2)]

concCtrl = [readConcCsv(x) for _,x in enumerate(concCsvCtrl)]
concExp1 = [readConcCsv(x) for _,x in enumerate(concCsvExp1)]
concExp2 = [readConcCsv(x) for _,x in enumerate(concCsvExp2)]

dC  = concCtrl+dataCtrl
dE1 = concExp1+dataExp1
dE2 = concExp2+dataExp2

pNormMin = 0.05
paramIdx = [0,8,9,11,12,13]
paramTitles = ['CONCURRENCY','STANCE_AMPLITUDE', 'STANCE_DURATION', 'SWING_AMPLITUDE', 'SWING_DURATION', 'WALK_SPEED']


allStats = []
allStats.append(['Test and Parameter', 'p-Value', 'p-Value', 'p-Value'])
allStats.append(['',ctrl+' vs. '+exp1, ctrl+' vs. '+exp2, exp1+' vs. '+exp2])

for p,n in enumerate(paramIdx):
    print '\n--------',paramTitles[p],'--------'
    testData = restrucDataForStats([dC[n],dE1[n],dE2[n]], dataSets)
    for i in xrange(len(testData)):
        label = '---'+paramTitles[p]+'_'+dC[n][0][i]+'---'
        print label
        normP = []
        for j in xrange(len(testData[i])):
            _, pValue = stats.normaltest(testData[i][j])
            normP.append(pValue)
        if min(normP)<pNormMin:
            testUsed = 'Kruskal-Wallis test'
            _, statsP = stats.kruskal(*testData[i])
            print testUsed+' pValue:', statsP,'---'
            multiCompP = getKWmultiComp(testData[i], dataSets, verbose=False)
        else:
            testUsed = 'One Way ANOVA'
            _, statsP = stats.f_oneway(*testData[i])
            print testUsed+' pValue:', statsP
            multiCompP = list(getOWANOVAmultiComp(testData[i], dataSets, verbose=False))
        allStats.append([label])
        allStats.append(['normalityTestStats']+normP)
        allStats.append([testUsed,statsP])
        allStats.append(['MultipleComparisons p-Value']+multiCompP)
        allStats.append([])


#for p,n in enumerate(paramIdx):
#    print '\n--------',paramTitles[p],'--------'
#    testData = restrucDataForStats([dC[n],dE1[n],dE2[n]], dataSets)
#    for i in xrange(len(testData)):
#        label = '---'+paramTitles[p]+'_'+dC[n][0][i]+'---'
#        print label
#        normP = []
#        for j in xrange(len(testData[i])):
#            _, pValue = stats.normaltest(testData[i][j])
#            normP.append(pValue)
#        if min(normP)<pNormMin:
#            testUsed = 'Kruskal-Wallis test'
#            _, statsP = stats.kruskal(*testData[i])
#            print testUsed+' pValue:', statsP,'---'
#            multiCompP = getKWmultiComp(testData[i], dataSets, verbose=False)
#        else:
#            testUsed = 'One Way ANOVA'
#            _, statsP = stats.f_oneway(*testData[i])
#            print testUsed+' pValue:', statsP
#            multiCompP = list(getOWANOVAmultiComp(testData[i], dataSets, verbose=False))
#        allStats.append([label])
#        allStats.append(['normalityTestStats']+normP)
#        allStats.append([testUsed,statsP])
#        allStats.append(['MultipleComparisons p-Value']+multiCompP)
#        allStats.append([])
#
#
#csvOutFile = baseDir+'stats_gait.csv'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(allStats)
#
#
#
#
#concData = [concCtrl[0], concExp1[0], concExp2[0]]
#concRestrucData = restrucDataForRMA(concData, dataSets)
#
#def getRMAnova(dataSet, labels, verbose=False):
#    tlabels = np.concatenate([[labels[j] for _,y in enumerate(x) ]for j,x in enumerate(dataSet)])
#    concatData = np.concatenate(dataSet)
#    ids = np.concatenate([np.arange(len(x)) for _,x in enumerate(dataSet)])
#    d = {'id':ids, 'rt':concatData, 'cond':tlabels}
#    df = pd.DataFrame(d)
#    anovarm = AnovaRM(df, 'rt', 'id', within=['cond'])
#    res = anovarm.fit()
#    if verbose:
#        print (res.summary())
#    return res
#
#
#rmAnovaStats = [['Repeated Measures ANOVA for Concurrency states (S0, S1, S2, S3)']]
#for i,x in enumerate(concRestrucData):
#    rma = getRMAnova(x, dataSets, True)
#    rmAnovaStats.append([rma.anova_table])
#    rmAnovaStats.append([rma.summary()])
#
#
#rmAnovaStats
#csvOutFile = baseDir+'stats_conc.txt'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(rmAnovaStats)
#







 


