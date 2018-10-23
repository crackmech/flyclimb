import Tkinter as tk
import tkFileDialog as tkd
from PIL import ImageTk, Image
import cv2
import os
import re
from datetime import datetime
import glob


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

def showtkIntIm(value=0):
    """use slider position to display new image"""
    global im
    i = imSlider.get()
    im = cv2.cvtColor(cv2.imread(flist[i]), cv2.COLOR_BGR2RGB)
    contours = getImContours((im, params))
    for cnt in contours:
        cv2.ellipse(im,cv2.fitEllipse(cnt),(0,150,255),2)
        print cv2.contourArea(cnt)
    img2 = ImageTk.PhotoImage(Image.fromarray(im))
    panel.configure(image=img2)
    panel.image = img2

def selFoler():
    global dirname, flist, im
    imSlider.set(1)
    dirname = getFolder(dirname)
    flist = getFiles(dirname, fileExtns)
    im = cv2.cvtColor(cv2.imread(flist[0]), cv2.COLOR_BGR2RGB)
    showtkIntIm()
    imSlider.config(to=len(flist)-1)
    folBoxVar.set(dirname)


def printcoords(event):
    #outputting x and y coords to console
    print (event.x,event.y)

def getImContours(args):
    '''
    returns the list of detected contour
    input:
        im:  image numpy array to be processed for contour detection
        params: dictionary of all parameters used for contour detection
        params :
            blurkernel
            block
            cutoff
            ellratio
            ellAxisRatioMin
            ellAxisRatioMax
            flyAreaMin
            flyAreaMax
    '''
    imgCnt = cv2.cvtColor(args[0], cv2.COLOR_RGB2GRAY)
    params = args[1]
    contour = []
    imgCnt = cv2.medianBlur(imgCnt,params['blurKernel'])
    th = cv2.adaptiveThreshold(imgCnt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,params['block'],params['cutoff'])
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea)[-10:]
    ellRatio = [(float(cv2.fitEllipse(cnt)[1][0])/cv2.fitEllipse(cnt)[1][1], cv2.contourArea(cnt), cnt) for cnt in contours ]
    for cnt in ellRatio:
        if params['ellaxisRatioMin']<cnt[0]<params['ellaxisRatioMax'] and params['flyareaMin']<cnt[1]<params['flyareaMax']:
            contour.append(cnt[2])
        else:
            pass
    return contour

def showImCnt(args):
    global params
    i = imSlider.get()
    im = cv2.cvtColor(cv2.imread(flist[i]), cv2.COLOR_BGR2RGB)
    params['block'] = blockSlider.get()
    if params['block']% 2 == 0:
        params['block'] = params['block']-1
    blockSlider.set(params['block'])
    params['cutoff'] = cutOffSlider.get()
    params['flyareaMin'] = minAreaSlider.get()
    params['flyareaMax'] = maxAreaSlider.get()
    contours = getImContours((im, params))
    for cnt in contours:
        cv2.ellipse(im,cv2.fitEllipse(cnt),(0,150,255),2)
    img2 = ImageTk.PhotoImage(Image.fromarray(im))
    panel.configure(image=img2)
    panel.image = img2
    
dirname = '/media/aman/data/flyWalk_data/data_20180522/tmpWalkingData/0_original_tracked/'

dirname = '/media/aman/data/flyWalk_data/flyWalking/media/flywalk/data/delete/glassChamber/20170517_173112_P0163xW1118_0200_20170509_90-101hr_1-Walking/imageData/20170517_173603/'
dirname = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male/imageData/20171201_195943/'
dirname = '/media/aman/data/flyWalk_data/tmp_climbing/CS1/tmp_20171201_195931_CS_20171128_0245_11-Climbing_male/imageData/20171201_200030_tracked/'


#imExt = '.png'
#imExt = '.jpeg'

#dirname = getFolder(dirname)
saveFolderCrop = dirname+'_tracked_cropped'
saveFolderOrg = dirname+'_tracked_original'


fileExtns = ['*.png', '*.jpeg']


params = {} # dict for holding parameter values for contour detection
params['blurKernel'] = 5
params['block'] = 93
params['cutoff'] = 15
params['ellaxisRatioMin'] = 0.2
params['ellaxisRatioMax'] = 0.5
params['flyareaMin'] = 1200 
params['flyareaMax'] = 5000


climbParams = {
                'flyAreaMin' : 300,
                'flyAreaMax' : 900,
                'block' : 91,
                'cutoff' : 35,
                'pixelSize' : 0.055 #pixel size in mm
                }

walkParams = {
                'flyAreaMin' : 1200,
                'flyAreaMax' : 5000,
                'block' : 93,
                'cutoff' : 15,
                'pixelSize' : 0.055 #pixel size in mm
                }

flist = getFiles(dirname, fileExtns)
#flist = natural_sort(glob.glob(dirname+'/*'+imExt))
im = cv2.cvtColor(cv2.imread(flist[0]), cv2.COLOR_BGR2RGB)


#This creates the main window of an application
window = tk.Tk()
window.title(dirname.split('/')[-3])
window.configure(background='grey')
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.fromarray(im))
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.

panel = tk.Label(window, image = img)
#mouseclick event
panel.bind("<Button 1>",printcoords)

scaleWidth = (im.shape[1]/2) - 2
imSlider = tk.Scale(window, label = "Move Slider to Change Displayed Image",
    from_= 0, to = len(flist)-1, tickinterval = 0, resolution = 1, length = im.shape[1],
    showvalue = 'yes', orient = 'horizontal', command=showtkIntIm)

blockSlider = tk.Scale(window, label = "Move Slider to Change block (~95)",
    from_= 0, to = 255-1, tickinterval = 0, resolution = 1, length = scaleWidth,
    showvalue = 'yes', orient = 'horizontal', command=showImCnt)

cutOffSlider = tk.Scale(window, label = "Move Slider to Change cutoff (~15)",
    from_= 0, to = 255-1, tickinterval = 0, resolution = 1, length = scaleWidth,
    showvalue = 'yes', orient = 'horizontal', command=showImCnt)

minAreaSlider = tk.Scale(window, label = "Move Slider to Change minArea",
    from_= 0, to = 10000, tickinterval = 0, resolution = 50, length = scaleWidth,
    showvalue = 'yes', orient = 'horizontal', command=showImCnt)

maxAreaSlider = tk.Scale(window, label = "Move Slider to Change maxArea",
    from_= 0, to = 10000, tickinterval = 0, resolution = 50, length = scaleWidth,
    showvalue = 'yes', orient = 'horizontal', command=showImCnt)

folBoxVar = tk.StringVar()
folderBox = tk.Entry(window, width=80, textvariable = folBoxVar)
folBoxVar.set(dirname)

btn = tk.Button(window, text="Select image folder", command=selFoler)


blockSlider.set(95)
cutOffSlider.set(15)
minAreaSlider.set(200)
maxAreaSlider.set(5000)

#pack(side = "bottom", fill = "both", expand = "yes")

panel.grid(row = 0, column = 0, 
           rowspan = 2, 
           columnspan = 2, sticky= 'W')

imSlider.grid(row = 2, column = 0, columnspan = 2, sticky= 'W')

blockSlider.grid(row = 3, column = 0, sticky='W')
cutOffSlider.grid(row = 4, column = 0, sticky='W')

minAreaSlider.grid(row = 3, column = 1, sticky='W')
maxAreaSlider.grid(row = 4, column = 1, sticky='W')

folderBox.grid(row = 5, column = 0, sticky='W')
btn.grid(row = 5, column = 1, sticky='W')


#Start the GUI

window.mainloop()
