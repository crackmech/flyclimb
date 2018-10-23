#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:48:49 2018

@author: aman
"""

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(8, 10))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

imDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'

frames = pims.ImageSequence(imDir+'*.png', as_grey=True)

print(frames[0])  # the first frame


plt.imshow(frames[0]);




f = tp.locate(frames[0], 11)

f.head()
plt.figure()  # make a new figure
tp.annotate(f, frames[0]);

f = tp.batch(frames, 11)

t = tp.link_df(f, 30, memory=111)
plt.figure()
tp.plot_traj(t);
plt.show()




