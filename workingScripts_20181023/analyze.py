# Imports
# ==================================================================================================
from sys import argv
import re
from pylab import diff, arctan2, pi, sqrt, mean, array
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from os.path import join as pjoin, exists, isfile
from os import makedirs, listdir
import operator
from numpy import Infinity, arange
from shutil import copyfile
from collections import Counter
from numpy.linalg.linalg import _multi_svd_norm

# Data Hierarchy
# 	Stacks
# 	Stack
# 	Cropped Frames
# 	TSV Leg data
#


FLY_LENGTH = 2700.0  # um
FRAME = 4.0  # ms

PIXEL_SIZE = 27.34375   #um/px

img_w = 200.0
img_h = 200.0


class Analyze:
    # path : Path of data about a single walking bout
    def __init__(self, path):
        self.set_stack_path(path)
        self.set_cropped_frames_path(pjoin(path, 'tmp'))
        self.labelled_frames_path = pjoin(path, 'tmp', 'labelled')

        self.set_labelled_frames_path(self.labelled_frames_path)

        self.set_tsv_path(pjoin(path, 'tmp', 'legs'))

        self.meanFlyLength_px = mean(self.getBodyLength())

        self.concurrency = None

    # Function Space
    # ==================================================================================================

    # File Loading functions
    # --------------------------------------------------------------------------------------------------



    def prod(self, factors):
        return reduce(operator.mul, factors, 1)

    # load a TSV file and return a (2D) list of columns
    def _loadtrace(self, tracepath):
        with open(tracepath) as f:
            data = [map(float, (t, x, y)) for line in f
                    if not re.search('[a-zA-Z]', line)
                    for t, x, y in [line.strip().split('\t')]]
            T, X, Y = zip(*data)
            X = [x * FLY_LENGTH / self.meanFlyLength_px for x in X]
            Y = [y * FLY_LENGTH / self.meanFlyLength_px for y in Y]
        return (T, X, Y)

    # return zip(*data)


    def _loadflytrace(self, tracepath):
        with open(tracepath) as f:
            data = [map(float, (x, y, L)) for line in f
                    if not re.search('[a-zA-Z]', line)
                    for t, x, y, minor, L, angle, nslice in [line.strip().split('\t')]]
        X, Y, L = zip(*data)
        X = [x * FLY_LENGTH / mean(L) for x in X]
        Y = [y * FLY_LENGTH / mean(L) for y in Y]
        return (X, Y, L)

    # return zip(*data)


    def get_tsv_path(self):
        return self.tsv_path

    def get_labelled_frames_path(self):
        return self.labelled_frames_path

    def get_cropped_frames_path(self):
        return self.cropped_frames_path

    def get_stack_path(self):
        return self.stack_path

    def set_tsv_path(self, path):
        self.tsv_path = path

    def set_cropped_frames_path(self, path):
        self.cropped_frames_path = path

    def set_labelled_frames_path(self, path):
        self.labelled_frames_path = path

    def set_stack_path(self, path):
        self.stack_path = path

    # Plotting Charts / Summary Diagrams functions
    # -------------------------------------------------------------------------------------------------------

    # Paint the gait diagram w.r.t. gait
    # {Tripod : COLOR1, Tetrapod : COLOR2, Wavelike : COLOR3, Others : WHITE}
    def genGaitPaint(self):
        sw_w = 5
        sw_h = 30
        pad = 2
        header = 50
        h = (sw_h + 5 * pad) * 6 + header

        X = self._loadtrace(pjoin(self.get_tsv_path(), 'L1.tsv'))[1]  # to measure the frame count
        max_w = len(X) * (sw_w + pad)
        w = 100 + max_w + 2 * pad

        im = Image.new('RGB', (w, h), (255, 255, 255))
        dr = ImageDraw.Draw(im)

        leg_groups = [('L1', 'R2', 'L3'), ('R1', 'L2', 'R3')]
        group_scores = []
        for lg in leg_groups:
            Xs = [self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))[1] for leg in lg]
            taggedlists = map(self._getSwingTaggedList, Xs)
            group_scores.append([sum(row) for row in zip(*taggedlists)])

        gaits = [max(gait1, gait2) for gait1, gait2 in zip(*group_scores)]
        # #print(gaits)
        [self._band(gaits[e], dr, 100 + e * (sw_w + pad), header, sw_w, h)
         for e in range(len(gaits))
         ]
        im.save(pjoin(self.get_stack_path(), "gaitpaint.png"))

    def genGaitPlot(self):

        legs = ('L1', 'L2', 'L3', 'R1', 'R2', 'R3')
        Xs = [self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))[1] for leg in legs]
        swingtaggedlists = map(self._getSwingTaggedList, Xs)
        # mark the swings a 2s
        swingtaggedlists = [self._mult(stlist, 2) for stlist in swingtaggedlists]
        # leave the stances as 1s
        stancetaggedlists = map(self._getStanceTaggedList, Xs)
        mergedeventlists = [self._merge(swtlists, sttlists) for swtlists, sttlists in
                            zip(swingtaggedlists, stancetaggedlists)]

        event_h = 1
        bottoms = range((len(mergedeventlists) + 1) * event_h + 5, 1 + 5, -event_h)

        fig, ax = plt.subplots()
        [self._plotGait(plt, eventlist, event_h, bt) for eventlist, bt in zip(mergedeventlists, bottoms)]
        ax.set_ylim(0, 20)
        plt.savefig(pjoin(self.get_stack_path(), "gait.svg"))

    def _plotGait(self, plt, eventlist, h, bot):
        event_w = 20
        # Red swings + Green stances
        # colordefs = ['#ffffff', '#188487', '#A60628', ]
        # Green swings + Red stances
        colordefs = ['#ffffff', '#A60628', '#188487', ]
        colors = [colordefs[k] for k in eventlist]
        plt.bar(arange(len(eventlist)) + event_w, [h] * len(eventlist), width=10, bottom=bot, color=colors)

    def genGaitDiagram(self):
        event_w = 8
        event_h = 40
        pad = 2
        header = 50
        h = (event_h + 5 * pad) * 6 + header

        legs = ('L1', 'L2', 'L3', 'R1', 'R2', 'R3')
        Xs = [self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))[1] for leg in legs]

        swingtaggedlists = map(self._getSwingTaggedList, Xs)
        # mark the swings a 2s
        swingtaggedlists = [self._mult(stlist, 2) for stlist in swingtaggedlists]

        # leave the stances as 1s
        stancetaggedlists = map(self._getStanceTaggedList, Xs)

        mergedeventlists = [self._merge(swtlists, sttlists) for swtlists, sttlists in
                            zip(swingtaggedlists, stancetaggedlists)]

        # #print(map(len,taggedlists))
        max_w = max(map(len, list(swingtaggedlists) + list(stancetaggedlists))) * (event_w + pad)
        w = 100 + max_w + 2 * pad

        # 		bg = (255, 255, 255)
        bg = (220, 220, 220)
        im = Image.new('RGB', (w, h), bg)
        dr = ImageDraw.Draw(im)

        [self._tick(mergedeventlists[leg][e], dr, 100 + e * (event_w + pad), header + leg * (event_h + 5 * pad),
                    event_w, event_h) for leg in range(len(legs))
         for e in range(len(mergedeventlists[leg]))
         ]

        self._rect(dr, (pad, pad), (pad + 80, header * 0.8), "grey", "black")
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", 15)
        dr.text((pad + 5, pad + 10), "Frames : " + str(len(Xs[0])), (255, 255, 255), font=font)

        legend_x = 100
        legend_y = pad
        spacer = 70
        self._tick(2, dr, legend_x, legend_y, event_w, event_h)
        dr.text((legend_x + event_w, event_h / 3), " = swing,", "grey", font=font)

        self._tick(1, dr, legend_x + spacer, legend_y, event_w, event_h)
        dr.text((legend_x + event_w + spacer, event_h / 3), " = stance,", "grey", font=font)

        self._tick(0, dr, legend_x + 2 * spacer, legend_y, event_w, event_h)
        dr.text((legend_x + event_w + 2 * spacer, event_h / 3), " = static", "grey", font=font)

        im.save(pjoin(self.get_stack_path(), "gait.png"))

    def _merge(self, lst1, lst2):
        return [self._max(lst1_item, lst2_item) for lst1_item, lst2_item in zip(lst1, lst2)]

    def _max(self, val1, val2):
        # assert bool(val1 and val2) is False
        if val1 and val2:
            print('Collision!')
        return max(val1, val2)

    def _mult(self, arr, factor):
        return [item * factor for item in arr]

    def _band(self, evt, dr, x, y, w, h):
        pod = {3: 'TRIPOD', 2: 'TETRAPOD', 1: 'WAVELIKE', 0: 'OTHERS'}  # kLegs swinging : kPOD
        color = {'TRIPOD': (255, 0, 0), 'TETRAPOD': (0, 255, 0), 'WAVELIKE': (0, 0, 255), 'OTHERS': (255, 255, 255)}
        self._rect(dr, (x, y), (x + w, y + h), fillc=color[pod[evt]], outlinec=color[pod[evt]])

    def _tick(self, evt, dr, x, y, w, h):
        colordefs = ['#A60628', '#188487', '#CF4457', '#7A68A6', '#348ABD',
                     '#E24A33', '#467821', ]

        # 		colors = {2:'green', 1:'magenta', 0:'grey'}
        # Red swings + Green stances
        # colors = {2: colordefs[0], 1: colordefs[1], 0: 'white'}
        # Green swings + Red stances
        colors = {2: colordefs[1], 1: colordefs[0], 0: colordefs[0]}
        self._rect(dr, (x, y), (x + w, y + h), fillc=colors[evt], outlinec=(220, 220, 220))

    def _rect(self, dro, xy, wh, fillc, outlinec="blue"):
        dro.rectangle((xy, wh), fill=fillc, outline=outlinec)

    # Annotation functions
    # -------------------------------------------------------------------------------------------------------

    def labelLeg(self, leg, mode, leg_color=(255, 0, 0)):
        if not exists(self.labelled_frames_path):
            makedirs(self.labelled_frames_path)
            files = [f for f in listdir(self.get_cropped_frames_path()) if f.endswith(".jpeg")]
            print('Copying', len(files), 'files for annotation')
            for f in files:
                copyfile(pjoin(self.get_cropped_frames_path(), f), pjoin(self.labelled_frames_path, f))

        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))

        if mode == 'SWING_MODE':
            etl = self._getSwingTaggedList(X)
        elif mode == 'STANCE_MODE':
            etl = self._getStanceTaggedList(X)

        legs = {'L1': (2, 0), 'L2': (1, 0), 'L3': (0, 0), 'R1': (2, 1), 'R2': (1, 1), 'R3': (0, 1)}

        cellw = 14
        cellh = 14
        gridx = 157
        gridy = 5

        spot_d = 8

        for k in range(len(etl)-1):
            fpath = pjoin(self.get_labelled_frames_path(), "f" + str(k) + ".jpeg")
            im = Image.open(fpath)
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)

            lxcurr = X[0] * self.meanFlyLength_px / FLY_LENGTH  # um to pixels
            lycurr = Y[1] * self.meanFlyLength_px / FLY_LENGTH  # um to pixels

            if k > 0:
                lxcurr = X[k] * self.meanFlyLength_px / FLY_LENGTH  # um to pixels
                lycurr = Y[k] * self.meanFlyLength_px / FLY_LENGTH  # um to pixels

            draw.ellipse((img_w / 2 - 3, img_h / 2 - 3, img_w / 2 + 3, img_h / 2 + 3),
                         fill=(0, 0, 255))  # fly centroid (image center)
            # global grid
            [draw.rectangle(
                [gridx + m * cellw, gridy + n * cellh, gridx + m * cellw + cellw, gridy + n * cellh + cellh],
                outline=(0, 0, 255)) for m in range(3) for n in range(2)]

            # Event blobs
            if etl[k] == 1:  # if the leg shows an event
                # fill global grid
                u, v = legs[leg]
                draw.rectangle(
                    [gridx + u * cellw, gridy + v * cellh, gridx + u * cellw + cellw, gridy + v * cellh + cellh],
                    fill=leg_color)
                #
                draw.ellipse((lxcurr - spot_d / 2, lycurr - spot_d / 2, lxcurr + spot_d / 2, lycurr + spot_d / 2),
                             fill=leg_color)  # leg centroid
            #

            # box names
            # font = ImageFont.truetype("/usr/share/cups/fonts/FreeMono.tt:f", 8)
            legnames = {(0, 0): 'L3', (1, 0): 'L2', (2, 0): 'L1',
                        (0, 1): 'R3', (1, 1): 'R2', (2, 1): 'R1', }
            pad = 2
            # [draw.text((pad + gridx + m * cellw, pad + gridy + n * cellh), legnames[(m, n)], (50, 50, 50), font=font) for m in range(3) for n in range(2)]
            [draw.text((pad + gridx + m * cellw, pad + gridy + n * cellh), legnames[(m, n)], (50, 50, 50)) for m in
             range(3) for n in range(2)]
            #

            del draw
            im.save(pjoin(self.get_labelled_frames_path(), "f" + str(k)) + ".jpeg", "JPEG")

    # Data functions
    # --------------------------------------------------------------------------------------------------


    # "Derived" Parameters
    # --------------------------------------------------------------------------------------------------
    def getTripodIndex(self):
        i = 0
        # TODO
        return i

    # concurrency is defined on the basis of swing events considered over a leg group
    # If three legs are swinging, concurrency	= 3
    # If two legs are swinging, concurrency		= 2
    # If one leg is swinging, concurrency 		= 1
    # If no legs is swinging, concurrency 		= 0
    def getConcurrency(self, state):
        if self.concurrency == None:
            leg_groups = [('L1', 'R2', 'L3'), ('R1', 'L2', 'R3')]
            group_scores = []
            for lg in leg_groups:
                Xs = [self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))[1] for leg in lg]
                taggedlists = map(self._getSwingTaggedList, Xs)
                group_scores.append([sum(row) for row in zip(*taggedlists)])
                #print leg
            gaits = [max(gait1, gait2) for gait1, gait2 in zip(*group_scores)]
            #print(gaits)
            counts = Counter(gaits)
            concurrency = {'S3': [], 'S2': [], 'S1': [], 'S0': []}
            statemap = dict(zip(('S3', 'S2', 'S1', 'S0',), (3, 2, 1, 0)))
            for k in ('S3', 'S2', 'S1', 'S0',):  # loop over possible concurrency states in descending order
                concurrency[k].append(100 * counts[int(statemap[k])] / len(gaits))
            self.concurrency = concurrency
            #print('CONCURRENCIES', concurrency)
        return [self.concurrency[state]]


    def getStolenSwings(self, joker=None):
        ctr = 0
        # legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        legs = ['L2', 'L3', 'R2', 'R3']
        ledger_gain = dict(zip(legs, [0 for k in range(6)]))
        ledger_lose = dict(zip(legs, [0 for k in range(6)]))
        tracesX = [self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))[1] for leg in legs]
        SWTs = map(self._getSwingTaggedList, tracesX)
        dSWTs = diff(SWTs)  # 1 : SWING inits
        inits = []
        # build the leg inits cassette
        for k in range(len(dSWTs[0])):
            for j in range(len(dSWTs)):
                if dSWTs[j][k] == 1:
                    inits.append(j)
        #

        # 6 point moving block SET check, increment counter
        for j in range(0, len(inits), len(legs)):
            if j + len(legs) - 1 < len(inits):
                seg = inits[j:j + len(legs)]
                # print(seg)
                residue = set(range(len(legs))) - set(seg)
                ctr += len(residue)
                for k in residue: ledger_lose[legs[k]] += 1
                counter = Counter(seg)
                excess = [k for k in counter if counter[k] > 1]
                for k in excess: ledger_gain[legs[k]] += 1
        #
        # print('Gains', ledger_gain)
        # print('Losses', ledger_lose)
        cycles = len(inits) / float(len(legs))
        # print('Cycles', cycles)
        # print('Total steals', ctr)
        return [ctr / cycles]

    # "Core" Parameters
    # --------------------------------------------------------------------------------------------------


    def getWalkingSpeed(self, joker=None):
        flytracepath = pjoin(self.get_cropped_frames_path(), 'fly_free.tsv')
        x, y, L = self._loadflytrace(flytracepath)
        x1 = x[0]
        y1 = y[0]
        x2 = x[-1]
        y2 = y[-1]
        t = len(x) * FRAME
        return [self.euclidD(x1, y1, x2, y2) / t]

    def euclidD(self, x1, y1, x2, y2):
        s = (y2 - y1) ** 2 + (x2 - x1) ** 2
        d = sqrt(s)
        return d

    def getBodyLength(self):
        flytracepath = pjoin(self.get_cropped_frames_path(), 'fly_free.tsv')
        x, y, L = self._loadflytrace(flytracepath)
        return L

    def getPEA(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        bccXY = self._bcc(X, Y, (
        img_w / 2 * FLY_LENGTH / self.meanFlyLength_px, img_h / 2 * FLY_LENGTH / self.meanFlyLength_px))
        X, Y = zip(*bccXY)
        swtl = self._getSwingTaggedList(X)
        dswtl = [0] + list(diff(swtl))
        starts = [k - 1 for k in range(len(dswtl)) if dswtl[k] == 1]
        return [self._computeAngle(X[start], Y[start]) for start in starts]

    def getAEA(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        bccXY = self._bcc(X, Y, (
        img_w / 2 * FLY_LENGTH / self.meanFlyLength_px, img_h / 2 * FLY_LENGTH / self.meanFlyLength_px))
        X, Y = zip(*bccXY)
        swtl = self._getSwingTaggedList(X)
        dswtl = [0] + list(diff(swtl))
        ends = [k for k in range(len(dswtl)) if dswtl[k] == -1]
        return [self._computeAngle(X[end], Y[end]) for end in ends]

    def getLegBodyAngles(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        bccXY = self._bcc(X, Y, (
        img_w / 2 * FLY_LENGTH / self.meanFlyLength_px, img_h / 2 * FLY_LENGTH / self.meanFlyLength_px))
        return [self._computeAngle(x, y) for x, y in bccXY]

    def _computeAngle(self, x, y):
        return 180 * arctan2(y, x) / pi

    # transform image cordinates to fly Body Centered Coordinates
    def _bcc(self, X, Y, cXY):
        cx, cy = cXY
        return zip([x - cx for x in X], [cy - y for y in Y])

    # Adjacent, Ipsi-lateral Coordination Index (ICI)
    # is a number that quantifies the swing coordination between adjancent ipsi-lateral legs
    # range: 1.0 [full co-ordination] - 0.0 [no co-ordination]
    # the ICI number is incremented based on the following truth tables for adjacent leg swing events
    # LEG	1	2	bICI	iICI
    # --------------------------------------
    # 	0	0	 -	 0
    # 	1	0	True	+1
    # 	0	1	True	+1
    # 	1	1	False	-1

    # LEG	2	3	bICI	iICI
    # --------------------------------------
    # 	0	0	 -	 0
    # 	1	0	True	+1
    # 	0	1	True	+1
    # 	1	1	False	-1
    # LEG : left or right paired with left or right respectively
    def getICI(self, pair):
        pairs = {'1': ('L1', 'L2', 'L1&2'), '2': ('R1', 'R2', 'R1&2'), '3': ('L2', 'L3', 'L2&3'),
                 '4': ('R2', 'R3', 'R2&3')}
        return [self._computeCoordinationIndex(pairs, pair)]

    # Compute segment-wise Contra-lateral Coordination Index (CCI)
    # is a number that quantifies the swing coordination between contra-lateral legs of a given segment
    # range: 1.0 [full co-ordination] - 0.0 [no co-ordination]
    # the CCI number is incremented based on the following truth table for left/right swing events
    # 	L	R	bCCI	iCCI
    # --------------------------------------
    # 	0	0	 -	 	0
    # 	1	0	True	+1
    # 	0	1	True	+1
    # 	1	1	False	-1
    def getCCI(self, segment):
        segments = {'1': ('L1', 'R1', 'T1'), '2': ('L2', 'R2', 'T2'), '3': ('L3', 'R3', 'T3')}
        return [self._computeCoordinationIndex(segments, segment)]

    # computer swing cordination index for legs M & N
    def _computeCoordinationIndex(self, dict_pairs, id_pair):
        M, N, label = dict_pairs[id_pair]
        # #print(label,)
        t, MX, MY = self._loadtrace(pjoin(self.get_tsv_path(), M + '.tsv'))
        t, NX, NY = self._loadtrace(pjoin(self.get_tsv_path(), N + '.tsv'))
        Mswtl = self._getSwingTaggedList(MX)
        Nswtl = self._getSwingTaggedList(NX)
        truth_table = {(0, 0): 0,
                       (0, 1): 1,
                       (1, 0): 1,
                       (1, 1): -1}
        cilist = [truth_table[pair] for pair in zip(Mswtl, Nswtl) if truth_table[pair]]
        return float(sum(cilist)) / len(cilist)

    def getStanceDuration(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        stl = self._getStanceTaggedList(X)
        return self._getEventDuration(stl)

    # takes a list of X-coordinates, detects stance events and returns a list of 1s and 0s
    # such that,
    # stance : 1	else : 0
    def _getStanceTaggedList(self, X, smooth=True):
        return self._getEventTaggedList(X, 'STANCE_MODE', smooth)

    def getPEPx(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        bccXY = self._bcc(X, Y, (
        img_w / 2 * FLY_LENGTH / self.meanFlyLength_px, img_h / 2 * FLY_LENGTH / self.meanFlyLength_px))
        X, Y = zip(*bccXY)
        swtl = self._getSwingTaggedList(X)
        dswtl = [0] + list(diff(swtl))
        starts = [k - 1 for k in range(len(dswtl)) if dswtl[k] == 1]
        return [X[start] for start in starts]

    def getAEPx(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        bccXY = self._bcc(X, Y, (
        img_w / 2 * FLY_LENGTH / self.meanFlyLength_px, img_h / 2 * FLY_LENGTH / self.meanFlyLength_px))
        X, Y = zip(*bccXY)
        swtl = self._getSwingTaggedList(X)
        dswtl = [0] + list(diff(swtl))
        ends = [k for k in range(len(dswtl)) if dswtl[k] == -1]
        return [X[end] for end in ends]

    def getStanceAmplitude(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        sttl = self._getStanceTaggedList(X, True)
        dsttl = [0] + list(diff(sttl))
        # #print dswtl
        starts = [k - 1 for k in range(len(dsttl)) if dsttl[k] == 1]
        ends = [k - 1 for k in range(len(dsttl)) if dsttl[k] == -1]

        # if len(ends) < len(starts):
        # 	ends.append(len(dswtl) - 1)


        # ##
        # #A.
        # DECIMATE starting and ending truncated swings
        if starts and ends:
            if starts[0] > ends[0]:
                del ends[0]
            if starts[-1] > ends[-1]:
                del starts[-1]
            # #
            # ##

            # #print 'nSTARTS, nENDS', map(len, [starts, ends])
            assert (len(starts) == len(ends))

        pairs = zip(starts, ends)
        # #print pairs


        # ##
        # #B.
        # #DECIMATING "bad termini" to remove variability artifact introduced by in-swing track chopping at the termini
        # Potentially "Bad termini" are arbitrarily defined as those that are less that 5 frames away (25ms) from start and end of the video sequence
        # #This should fix the problematic artefacts that escaped A. decimation
        # ##

        pairs = filter(lambda ind: ind[0] >= 3 and ind[1] <= len(X) - 4, pairs)
        # #
        # ##

        # print ">>", pairs
        stamp = [X[start] - X[end] for start, end in pairs]
        # swamp = filter(lambda val: val != 0, swamp)
        return stamp

    def getSwingAmplitude(self, leg):
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        swtl = self._getSwingTaggedList(X, True)
        dswtl = [0] + list(diff(swtl))
        # #print dswtl
        starts = [k - 1 for k in range(len(dswtl)) if dswtl[k] == 1]
        ends = [k - 1 for k in range(len(dswtl)) if dswtl[k] == -1]

        # if len(ends) < len(starts):
        # 	ends.append(len(dswtl) - 1)


        # ##
        # #A.
        # DECIMATE starting and ending truncated swings
        if starts and ends:
            if starts[0] > ends[0]:
                del ends[0]
            if starts[-1] > ends[-1]:
                del starts[-1]
            # #
            # ##

            # #print 'nSTARTS, nENDS', map(len, [starts, ends])
            assert (len(starts) == len(ends))

        pairs = zip(starts, ends)
        # #print pairs


        # ##
        # #B.
        # #DECIMATING "bad termini" to remove variability artifact introduced by in-swing track chopping at the termini
        # Potentially "Bad termini" are arbitrarily defined as those that are less that 5 frames away (25ms) from start and end of the video sequence
        # #This should fix the problematic artefacts that escaped A. decimation
        # ##

        pairs = filter(lambda ind: ind[0] >= 3 and ind[1] <= len(X) - 4, pairs)
        # #
        # ##


        # print ">>", pairs
        swamp = [X[end] - X[start] for start, end in pairs]
        # swamp = filter(lambda val: val != 0, swamp)
        return swamp

    def getSwingDuration(self, leg):
        '''
            Returns the list of Swing Durations for the specified leg.
        '''
        t, X, Y = self._loadtrace(pjoin(self.get_tsv_path(), leg + '.tsv'))
        swtl = self._getSwingTaggedList(X)
        # print swtl
        return self._getEventDuration(swtl)

    def _getEventDuration(self, etl):
        detl = [0] + list(diff(etl))
        starts = [k for k in range(len(detl)) if detl[k] == 1]
        ends = [k for k in range(len(detl)) if detl[k] == -1]

        # if len(ends) < len(starts):
        # 	ends.append(len(detl) - 1)

        # ##
        # #A.
        # DECIMATE starting and ending truncated swings
        if starts and ends:
            if starts[0] > ends[0]:
                del ends[0]
            if starts[-1] > ends[-1]:
                del starts[-1]
            # #
            # ##

            # #print 'nSTARTS, nENDS', map(len, [starts, ends])
            assert (len(starts) == len(ends))

        pairs = zip(starts, ends)
        # #print pairs

        # ##
        # #B.
        # #DECIMATING "bad termini" to remove variability artifact introduced by in-swing track chopping at the termini
        # Potentially "Bad termini" are arbitrarily defined as those that are less that 5 frames away (25ms) from start and end of the video sequence
        # #This should fix the problematic artefacts that escaped A. decimation
        # ##

        pairs = filter(lambda ind: ind[0] >= 3 and ind[1] <= len(detl) - 4, pairs)
        # #
        # ##

        ed = [FRAME * (end - start) for start, end in pairs]  # ms
        return ed

    # takes a list of X-coordinates, detects swing events and returns a list of 1s and 0s
    # such that,
    # swing : 1	else : 0
    def _getSwingTaggedList(self, X, smooth=True):
        return self._getEventTaggedList(X, 'SWING_MODE', smooth)

    def _getEventTaggedList(self, X, mode, smooth=True):
        dX = diff(X)
        etl = self._binarize(dX, mode)
        etl = [0] + list(etl)  # prepend a zero to equalize the "len" of X and etl
        if smooth:
            etl = self._smoothen(etl)
        # ##
        # TODO
        # enable during production only!
        # "Kadi" removal
        # Assumption: Most "singleton" events i.e. 010 that persist post-smoothing are artefacts.
        # This treatment may undermine "true inter-leg co-ordination errors" (which are rare, even in mutants)
        # but will help bring out true differences w.r.t. amplitudes and durations.
        # ##
        setl = "".join(map(str, etl))
        while (setl.count('010')):    setl = setl.replace('010', '000')
        etl = [int(k) for k in setl]
        ####
        # ##
        # #
        #
        return etl

    # converts analog leg displacement data to digital data
    # such that,
    # In SWING_MODE:
    # k > (0 + pixel_jitter) : 1	&	k < (0 + pixel_jitter) : 0
    # In STANCE_MODE:
    # k < 0 : 1			&	k > 0 : 0
    def _binarize(self, anaarr, mode, pixel_jitter=1.2):
        um_jitter = pixel_jitter * FLY_LENGTH / self.meanFlyLength_px
        if mode is 'SWING_MODE':
            # #print 'SWING'
            # anaarr[anaarr < (0 + um_jitter)] = 0  # replace all values less than (0+um_jitter) by zeros
            # anaarr[anaarr > 0] = 1  # replace all positive values by ones
            anaarr[anaarr <= 0] = 0
            anaarr[anaarr >= 3 * um_jitter] = Infinity  # super-binary
            b = (anaarr > 0) & (anaarr < Infinity)
            # #print b
            anaarr[b] = 1
            b = (anaarr == Infinity)
            anaarr[b] = 2
        # #print anaarr
        elif mode is 'STANCE_MODE':
            # #print 'STANCE'
            anaarr[anaarr >= 0] = 0  # replace all values more than 0 by zeros
            anaarr[anaarr <= -2 * um_jitter] = -Infinity  # super-binary
            b = (anaarr < 0) & (anaarr > -Infinity)
            # #print b
            anaarr[b] = 1
            b = (anaarr == -Infinity)
            anaarr[b] = 2

        anaarr = array(anaarr, dtype=int)
        return anaarr

    # replaces false 1s with 0s and false 0s with 1s
    def _smoothen(self, binarr):
        str_dX_base = "".join(map(str, (map(int, binarr))))  # stringify the data

        str_dX = str_dX_base.replace('2', '1')

        # #print 'r', str_dX_base
        # #print str_dX
        # Replacement rules
        # 		(start1)		Case 0: [0100...] ==> [0000...]
        # 		(null)		Case 1: 00000 	==> 00000
        # 		(False+)1	Case 2: 00100 	==> 00000
        # 		(False+)2	Case 3: 01010 	==> 01010
        # 		(False-)1	Case 4: 1101 	==> 1111
        # 		(False-)2	Case 5: 1011 	==> 1111
        # 		(null)		Case 6: 111 	==> 111
        # 		(end)		Case 7: [...0010] ==> [...0000]


        # recover lost "TRUE" events, on the basis of the binarization criterion of being >= 2*pixel_jitter
        str_dX_fixed = ''
        for k in range(len(str_dX)):
            if str_dX_base[k] == '2':
                c = '1'
            else:
                c = str_dX[k]
            str_dX_fixed += c

        str_dX = str_dX_fixed
        # recovery complete

        # enforce head rule
        if str_dX.startswith('010'):
            str_dX = '000' + str_dX[3:]
        #

        # enforce "neighborhood" rules: if your neighbours are 0 you are 0 too and if your neighbours are 1 then you are 1 too
        # while(str_dX.count('00100')):str_dX = str_dX.replace('00100', '00000')  # remove false positive tagged frames
        while (str_dX.count('101')): str_dX = str_dX.replace('101', '111')  # include false negative tagged frames
        while (str_dX.count('010')): str_dX = str_dX.replace('010', '000')  # remove false positive tagged frames
        # while(str_dX.count('1011')):str_dX = str_dX.replace('1011', '1111')  # include false negative tagged frames
        #


        # enforce tail rules
        if str_dX.endswith('010'):
            str_dX = str_dX[:-3] + '000'
        if str_dX.endswith('01'):
            str_dX = str_dX[:-2] + '00'
        #

        # #print 'i', str_dX

        # recover lost "TRUE" events, on the basis of the binarization criterion of being >= 2*pixel_jitter
        str_dX_fixed = ''
        for k in range(len(str_dX)):
            if str_dX_base[k] == '2':
                c = '1'
            else:
                c = str_dX[k]
            str_dX_fixed += c

        str_dX = str_dX_fixed
        # recovery complete

        # #print 'f', str_dX
        return [int(k) for k in str_dX]


if __name__ == "__main__":
    path = '/media/pushkar/ccc1f18b-7f26-4c0d-b510-a582d28d075e/pushkar/walking/data/Pushkar/analysis/CS_Fast/37'
    ana = Analyze(path)
    print(ana)
    # print(ana.getSwingAmplitude('L2'))
    # print(ana.getWalkingSpeed())
    # states = ('S3', 'S2', 'S1', 'S0',)
    # for state in states:
    #     print(state, ana.getConcurrency(state=state))

    ana.genGaitDiagram()
