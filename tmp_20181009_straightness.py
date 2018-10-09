#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 04:06:19 2018

@author: aman
"""

import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter as sgFilter






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

def findEuDis(pt1, pt2):
    return np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))

def findTotEuDisArr(xyArr):
    dis = 0
    for i in xrange(len(xyArr)-1):
        dis += np.sqrt(np.square(xyArr[i+1][0]-xyArr[i][0])+np.square(xyArr[i+1][1]-xyArr[i][1]))
    return np.array(dis)

def findEuDisArr(xyArr):
    dis = []
    for i in xrange(len(xyArr)):
        dis.append(np.sqrt(np.square(xyArr[i][0])+np.square(xyArr[i][1])))
    return dis

def vadd(a,b):
    return (a[0]+b[0],a[1]+b[1])

def vsub(a,b):
    return (a[0]-b[0],a[1]-b[1])

def project(a, b):
    """ project a onto b
        formula: b(dot(a,b)/(|b|^2))
    """
    abdot = (a[0]*b[0])+(a[1]*b[1])
    blensq = (b[0]*b[0])+(b[1]*b[1])

    temp = float(abdot)/float(blensq)
    c = (b[0]*temp,b[1]*temp)

    print a,b,abdot,blensq,temp,c
    return c


def findDistFromLine(pt, m, c):
    y = (m*pt[0]) + c
    x = (pt[1]- c)/m
    pt1 = [pt[0], y]
    pt2 = [x, pt[1]]
    ptLine = vsub(pt1, project(vsub(pt2, pt1), vsub(pt, pt1)))
    return findEuDis(ptLine, pt)



def calcAngle3Pts(a, b, c):
    '''
    returns angle between a and c with b as the vertex 
    '''
    ba = a.flatten() - b.flatten()
    bc = c.flatten() - b.flatten()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)



def getAngles(xyArray, angleStep, movinWinSize, smoothWin):
    '''
    returns an array of angles between three points a,b,c with number of points between b-a or b-c is given by angleStep
    movingWinSize gives the step of the moving window to calculate the angle between consecutive frames
    '''
    angles = []
    for i in xrange(angleStep, len(xyArray)-angleStep, movinWinSize):
        pt1 = xyArray[i]
        pt0 = xyArray[i-angleStep]
        pt2 = xyArray[i+angleStep]
        angles.append(calcAngle3Pts(pt0, pt1, pt2))
    if smoothWin==None:
        return np.array(angles)
    else:
        return sgFilter(np.array(angles), smoothWin, 1)


dirName = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'
o = '003656'
fileName = '_legTipsClus_n20-Climbing_'
fileName = '_legTipsClustered-Climbing_'
legTipsCsvName = dirName+'20180822_'+o+fileName+'legTipLocs.csv'
centroidsFile = dirName+'20180822_'+o+fileName+'centroids.csv'

angThresh = 170
angArm = 11
angleSlidWind = 1
disThresh = 100 #in pixels
framesThresh = 20
colors = [(0,255,0),(255,0,0),(255,255,0),(0,0,255),(0,125,255),(125,211,122)]

rows = getlegTipData(legTipsCsvName)
valuesOri = np.array(rows[:,1:], dtype=np.float64)
valuesPlt = valuesOri.copy()

cents1 = getCentroidData(centroidsFile)
cents = [x for _,x in enumerate(cents1) if x[1]!='noContourDetected']
frNamesAll = [x[0] for _,x in enumerate(cents) if x[1]!='noContourDetected']
centroids = np.array(cents)[:,1:].astype(dtype=np.float64)



consecAngles = getAngles(centroids[:,:2], angArm, angleSlidWind, 7)
#sliceIdx = np.where(consecAngles<angThresh)[0]+angArm
#distances = []
#for i in xrange(len(sliceIdx)-1):
#    distances.append(findTotEuDisArr(centroids[sliceIdx[i]:sliceIdx[i+1],:2]))
#trackSlices = np.where(np.array(distances)>disThresh)[0]
#frNumbers = []
#for _, x in enumerate(trackSlices):
#    frNumbers.append(sliceIdx[x:x+2])
#
#
#im = np.zeros((580,1280,3),dtype=np.uint8)
#
#
#for i in xrange(len(centroids)):
#    cv2.circle(im, (int(centroids[i,0]), int(centroids[i,1])), 2, (0,255,255), 1)
#for i,x in enumerate(frNumbers):
#    for j in xrange(x[0],x[1]+1):
#        cv2.circle(im, (int(centroids[j,0]), int(centroids[j,1])), 2, colors[i], 1)
#cv2.imshow('123',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
##
##im = np.zeros((580,1280,3),dtype=np.uint8)
##cv2.imshow('123',im)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##
##
#plt.plot(consecAngles)
#plt.show()





sliceIdx = np.where(consecAngles>angThresh)[0]+angArm
disIdx = [sliceIdx[0]]
for i,x in enumerate(np.diff(sliceIdx)):
    if x>1:
        disIdx.append(i-1)
disIdx.append(sliceIdx[-1])
distances = []
for i in xrange(len(disIdx)-1):
    distances.append(findTotEuDisArr(centroids[disIdx[i]:disIdx[i+1],:2]))
trackSlices = np.where(np.array(distances)>disThresh)[0]
frNumbers = []
for _, x in enumerate(trackSlices):
    frNumbers.append(disIdx[x:x+2])


im = np.zeros((580,1280,3),dtype=np.uint8)


for i in xrange(len(centroids)):
    cv2.circle(im, (int(centroids[i,0]), int(centroids[i,1])), 2, (0,255,255), 1)
for i,x in enumerate(frNumbers):
    for j in xrange(x[0],x[1]+1):
        cv2.circle(im, (int(centroids[j,0]), int(centroids[j,1])), 2, colors[i], 1)
cv2.imshow('123',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

















#
#slope, intercept, r_value, p_value, std_err = stats.linregress(centroids[:,0], centroids[:,1])
#
#
#distances = []
#for i,x in enumerate(centroids[:,:2]):
#    distances.append(findDistFromLine(x, slope, intercept))
#
#plt.plot(distances)
#plt.show()
#
#im = np.zeros((580,1280,3),dtype=np.uint8)
#
#
#for i in xrange(len(centroids)):
#    cv2.circle(im, (int(centroids[i,0]), int(centroids[i,1])), 2, (0,255,2), 1)
#cv2.imshow('123',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#linePtx = [(int(x), int((slope*x)+intercept)) for x in centroids[:,0]]
#linePty = [(int((x-intercept)/slope), int(x)+10) for x in centroids[:,1]]
#
#
#im = np.zeros((580,1280,3),dtype=np.uint8)
#for i in xrange(len(centroids)):
#    cv2.circle(im, linePtx[i], 2, (255,1,210), 1)
#    cv2.circle(im, linePty[i], 2, (255,0,2), 1)
#    cv2.circle(im, (int(centroids[i,0]), int(centroids[i,1])), 2, (0,255,2), 1)
#cv2.imshow('123',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#


''''

A = (75.0,75.0)
B = (100.0,50.0)
C = (90,70)

AB = vsub(B,A)
AC = vsub(C,A)
AD = project(AC,AB)
D = vsub(A, AD)

D = vsub(A, project(vsub(C,A), vsub(B,A)))







import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#y= mx+c
#x= (y-c)/m
#pt0 = centroids[10,:2]
#y = slope*pt0[0] + intercept
#x = (pt0[1]- intercept)/slope
#pt1 = [pt0[0], y]
#pt2 = [x, pt0[1]]
#ptLine = vsub(pt1, project(vsub(pt2, pt1), vsub(pt0, pt1)))
#dist = findEuDis(ptLine, pt0)






import numpy as np
curve = [8.4663, 8.3457, 5.4507, 5.3275, 4.8305, 4.7895, 4.6889, 4.6833, 4.6819, 4.6542, 4.6501, 4.6287, 4.6162, 4.585, 4.5535, 4.5134, 4.474, 4.4089, 4.3797, 4.3494, 4.3268, 4.3218, 4.3206, 4.3206, 4.3203, 4.2975, 4.2864, 4.2821, 4.2544, 4.2288, 4.2281, 4.2265, 4.2226, 4.2206, 4.2146, 4.2144, 4.2114, 4.1923, 4.19, 4.1894, 4.1785, 4.178, 4.1694, 4.1694, 4.1694, 4.1556, 4.1498, 4.1498, 4.1357, 4.1222, 4.1222, 4.1217, 4.1192, 4.1178, 4.1139, 4.1135, 4.1125, 4.1035, 4.1025, 4.1023, 4.0971, 4.0969, 4.0915, 4.0915, 4.0914, 4.0836, 4.0804, 4.0803, 4.0722, 4.065, 4.065, 4.0649, 4.0644, 4.0637, 4.0616, 4.0616, 4.061, 4.0572, 4.0563, 4.056, 4.0545, 4.0545, 4.0522, 4.0519, 4.0514, 4.0484, 4.0467, 4.0463, 4.0422, 4.0392, 4.0388, 4.0385, 4.0385, 4.0383, 4.038, 4.0379, 4.0375, 4.0364, 4.0353, 4.0344]
nPoints = len(curve)
allCoord = np.vstack((range(nPoints), curve)).T
np.array([range(nPoints), curve])
firstPoint = allCoord[0]
lineVec = allCoord[-1] - allCoord[0]
lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
vecFromFirst = allCoord - firstPoint
scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
vecToLine = vecFromFirst - vecFromFirstParallel
distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
idxOfBestPoint = np.argmax(distToLine)




def findElbowPoint(self, rawDocScores):
    dataPoints = zip(range(0, len(rawDocScores)), rawDocScores)
    s = np.array(dataPoints[0])
    l = np.array(dataPoints[len(dataPoints)-1])
    b_vect = l-s
    b_hat = b_vect/np.linalg.norm(b_vect)
    distances = []
    for scoreVec in dataPoints[1:]:
        p = np.array(scoreVec) - s
        proj = p.dot(b_hat)*b_hat
        d = abs(np.linalg.norm(p - proj)) # orthgonal distance between b and the L-curve
        distances.append((scoreVec[0], scoreVec[1], proj, d))

    elbow_x = max(distances, key=itemgetter(3))[0]
    elbow_y = max(distances, key=itemgetter(3))[1]
    proj = max(distances, key=itemgetter(3))[2]
    max_distance = max(distances, key=itemgetter(3))[3]

    red_point = proj + s



from scipy.signal import savgol_filter


dataPoints = centroids[:,:2]#zip(range(0, len(rawDocScores)), rawDocScores)
s = np.array(dataPoints[0])
l = np.array(dataPoints[len(dataPoints)-1])
b_vect = l-s
b_hat = b_vect/np.linalg.norm(b_vect)
distances = []
for scoreVec in dataPoints[1:]:
    p = np.array(scoreVec) - s
    proj = p.dot(b_hat)*b_hat
    d = abs(np.linalg.norm(p - proj)) # orthgonal distance between b and the L-curve
    distances.append((scoreVec[0], scoreVec[1], proj, d))

elbow_x = max(distances, key=itemgetter(3))[0]
elbow_y = max(distances, key=itemgetter(3))[1]
proj = max(distances, key=itemgetter(3))[2]
max_distance = max(distances, key=itemgetter(3))[3]

red_point = proj + s

arr = np.array(savgol_filter([x[-1] for x in distances], 31, 3))
plt.plot(arr)
plt.show()






import numpy as np

def findAnglesBetweenTwoVectors(v1s, v2s):
    dot_v1_v2 = np.einsum('ij,ij->i', v1s, v2s)
    dot_v1_v1 = np.einsum('ij,ij->i', v1s, v1s)
    dot_v2_v2 = np.einsum('ij,ij->i', v2s, v2s)

    return np.arccos(dot_v1_v2/(np.sqrt(dot_v1_v1)*np.sqrt(dot_v2_v2)))

A = [75.0,75.0]
B = [100.0,50.0]
C = [90,70]

findAnglesBetweenTwoVectors([A,A,B], [B,C,C])









# coding: utf-8 # # Orthogonal Projection # In[1]: import numpy as np
import numpy.linalg as la # In[2]: # for in-line plots
#get_ipython().magic(u'matplotlib inline') # for plots in a window #
#%matplotlib qt import matplotlib.pyplot as pt from mpl_toolkits.mplot3d
import Axes3D # Make two random 3D vectors: # In[3]: 
np.random.seed(13)
x = np.random.randn(3)
y = np.random.randn(3) # Make them orthonormal: #
#In[4]: 
y = y - y.dot(x)/x.dot(x)*x
x = x/la.norm(x) 
y = y/la.norm(y) #
#Check: # In[5]: 
print(y.dot(x)) 
print(la.norm(x)) 
print(la.norm(y)) #
#Plot the two vectors: # In[6]: 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlim3d([-3, 3])
ax.set_ylim3d([-3, 3]) 
ax.set_zlim3d([-3, 3]) 
xy = np.array([x, y]).T
ax.quiver( 0, 0, 0, xy[0], xy[1], xy[2],) # Make an array with the cornerpoints of a cube: # In[7]: 
plt.show()

points = np.array([ [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1],]) 
# Plot them: # In[8]: 
fig = plt.figure() 
ax = fig.add_subplot(111,projection='3d') 
ax.scatter(points[:,0], points[:, 1], points[:, 2]) #Construct the projection matrix: # In[9]: 
plt.show()

Q = np.array([ x,y,np.zeros(3)]).T 
print(Q) # In[10]: 
P = Q.dot(Q.T) 
print(P) # Check that $P^2=P$: #In[11]: 
la.norm(P.dot(P)-P) # Project the points, assign to `proj_points`: # In[12]: 
proj_points = np.einsum("ij,nj->ni", P, points)
# In[13]: 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:, 1], points[:, 2])
ax.scatter(proj_points[:,0], proj_points[:, 1], proj_points[:, 2], color="red") 
xy = np.array([x, y]).T 
ax.quiver( 0, 0, 0, xy[0], xy[1],xy[2],)
plt.show()

'''


