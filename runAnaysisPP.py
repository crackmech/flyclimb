from analyze import Analyze
from sys import argv
from os import listdir
from os.path import join as pjoin, isdir

from pylab import median, mean, std, sqrt, bar, boxplot, savefig, ylim, arange
#from pygal import Box
import sys
sys.path.append('HTML/')
import HTML


# root path containing all the stacks to be analyzed
# root path structure:
# STACK_NAME/tmp/legs/*.tsv
dirName = '/media/aman/data/flyWalk_data/climbingData/gait/allData/'
w1118Path = dirName+'W1118_legTracking/'
wlrrkPath = dirName+'W1118xLrrk-ex1_legTracking'
parkLrrkPath = dirName+'Park25xLrrk-ex1_legTracking/'
path = w1118Path#argv[1]
path = wlrrkPath#argv[1]
path = parkLrrkPath#argv[1]

stack_names = sorted(
			[ name for name in listdir(path)
				if not name.lower().count("unusable")
				if not name.lower().count("ignore")
				if isdir(pjoin(path, name))]
			)

print stack_names


# Legs prefix list

# legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
legs = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3']
# legs = ['L2', 'R2', 'L3', 'R3']

contralateral_pairs = map(str, [1, 2, 3])
ipsilateral_pairs = map(str, [1, 2, 3, 4])
concurrency_states = map(str, [3, 2, 1, 0])

# Leg and Body Parameter data holder lists
# value per event
swing_amplitudes = 	dict(zip(legs, [[] for k in range(6)]))
swing_durations = 	dict(zip(legs, [[] for k in range(6)]))
stance_durations = 	dict(zip(legs, [[] for k in range(6)]))
aepxs = 			dict(zip(legs, [[] for k in range(6)]))
pepxs = 			dict(zip(legs, [[] for k in range(6)]))
ccis = 			dict(zip(contralateral_pairs, [[] for k in range(3)]))
icis = 			dict(zip(ipsilateral_pairs, [[] for k in range(4)]))
lbas = 			dict(zip(legs, [[] for k in range(6)]))
speeds = 		{'Walk Speeds':[]}
aeas = 			dict(zip(legs, [[] for k in range(6)]))
peas = 			dict(zip(legs, [[] for k in range(6)]))
concurrencies = 		{'concurrencies':[]}

for stack_name in stack_names:
	# Init the Analyzer
	#------------------
	stack_path = pjoin(path, stack_name)
	print 'Analysing', stack_path
	print '----------------------------------------------------------------'
	ana = Analyze(stack_path)
	
	# Generate Gait Diagram
	#----------------------
	ana.genGaitDiagram()	
	
	# Param 1.
	print 'Swing Amplitude'
	#-----------------------
	# Unit of Observation : One Swing Event
	for leg in legs:
		swing_amplitudes[leg].extend(ana.getSwingAmplitude(leg))
	print
	
	# '''
	# 2
	print 'Swing Duration'
	#----------------------
	# Unit of Observation : One Swing Event
	for leg in legs:
		swing_durations[leg].extend(ana.getSwingDuration(leg))
	print
	
	# '''
	# 3
	print 'Stance Duration'
	#-----------------------
	# Unit of Observation : One Stance Event
	for leg in legs:
		stance_durations[leg].extend(ana.getStanceDuration(leg))
	print
	
	
	# '''
	# 4
	print 'AEPx'
	#----------------
	# Unit of Observation : One Swing Termination Event
	for leg in legs:
		aepxs[leg].extend(ana.getAEPx(leg))
	print
	
	
	# '''
	# 5
	print 'PEPx'
	#----------------
	# Unit of Observation : One Swing Onset Event
	for leg in legs:
		pepxs[leg].extend(ana.getPEPx(leg))
	print
	
	
	# '''
	# 6
	print 'CCI'
	#-----------
	# Unit of Observation : One Swing Event
	for segment in contralateral_pairs:
		ccis[segment].append(ana.getCCI(segment))
	print
	
	
	# '''
	# 7
	print 'ICI'
	#-----------
	# Unit of Observation : One Swing Event
	for pair in ipsilateral_pairs:
		icis[pair].append(ana.getICI(pair))
	print

	# '''
	# 8
	print 'Leg-Body Angles'
	#-----------------------
	# Unit of Observation : One Frame (5ms)
	for leg in legs:
		lbas[leg].extend(ana.getLegBodyAngles(leg))
	print

	
	# '''
	# 9
	print 'Walking Speeds'
	#-----------
	# Unit of Observation : One Walking Video Stack (a bout)
	speeds['Walk Speeds'].append(ana.getWalkingSpeed())
	print
	# '''
	
	
	# '''
	# 10
	print 'AEA'
	#-----------
	# Unit of Observation : One Swing Event
	for leg in legs:
		aeas[leg].extend(ana.getAEA(leg))
	print
	# '''


	# '''
	# 11
	print 'PEA'
	#-----------
	# Unit of Observation : One Swing Event
	for leg in legs:
		peas[leg].extend(ana.getPEA(leg))
	print
	# '''
	
		
	# '''
	# 12
	print 'Concurrencies'
	#-----------
	# Unit of Observation : One Walking Video Stack (a bout)
	concurrencies['concurrencies'].append([ana.getConcurrency(concurrency_states)])
	print
	# '''
	
	
	# '''
	# Annotate
	for leg in legs:
		ana.labelLeg(leg, 'SWING_MODE', (255, 255, 0))
		ana.labelLeg(leg, 'STANCE_MODE', (255, 0, 0))
	#
	# '''
	
	# '''
	# Gait Diagram
		ana.genGaitDiagram()
	#
	# '''
		
	# '''
	# Gait Diagram
		ana.genGaitPaint()
	#
	# '''


# data2d is a dict
def htmlTableDataForParam(subtitle, headings, data2d):
	htmlcode = ""
	walkstats = getstats(headings, data2d)
	# data table
	data2d = eqdatalens(headings, data2d)
	htmlcode += tablify(subtitle, headings, data2d)
	# statistics table
	htmlcode += HTML.table(walkstats)
	htmlcode += "<BR><BR>"
	return htmlcode


def eqdatalens(headings, data2d):
	# list lengths equalization
	data2d = [data2d[heading] for heading in headings]
	maxrows = len(max(data2d, key=len))
	data2d = [_padlist(col, maxrows) for col in data2d]
	return dict(zip(headings, data2d))
	
	#

def _padlist(l, n, c='-'):
	while len(l) < n + 1:
		l.append(c)
	return l


def getstats(headings, data2d):
	data2d = [data2d[heading] for heading in headings]
	statnames	 = ("Median", "Mean", "Min", "STD", "SEM")
	stats		 = (median, mean, min, std, sem)
	vals = [statnames]
	
	for col in data2d:
		colstats = []
		for stat in stats:
			colstats.append(round(stat(col), 2))
		vals.append(colstats)
	vals = zip(*vals)
	return [["Stats"] + headings] + vals


def sem(vals):
	return std(vals) / sqrt(len(vals))


def newlines(*stuff):
	return '\n<BR>'.join(stuff)


def bold(stuff):
	return '<em>' + stuff + '</em>'


def tablify(subtitle, headings, data):
	table_entries = zip(*[[heading] for heading in headings]) + zip(*[data[heading] for heading in headings])	
	return newlines(bold(subtitle), HTML.table(table_entries), "")


def plotForParam(param_name, headings, data2d, y_lim=None):
	imgcode = ''

	'''
	#PyGAL Box Plot
	imgpath = pjoin(path , param_name + ".svg")
	bp = Box()
	[bp.add(heading, data2d[heading]) for heading in headings]
	bp.render_to_file(imgpath)
	imgcode += '<object data="' + imgpath + '" type="image/svg+xml" width="500" height="400"></object>'
	#
	'''

	data2d = [data2d[heading] for heading in headings]
	print len(data2d)
	boxplot(data2d, hold=False)
#	if y_lim:
#		ylim(*y_lim)
	imgpath = pjoin(path , param_name + "_box.svg")
	savefig(imgpath)
	imgcode += '<img src="' + imgpath + '" width="500" height="400">'
	
	y = map(mean, data2d)
	x = range(len(data2d))
	yE = map(sem, data2d)
	bar(x, y, hold=False, yerr=yE, ecolor='black')
	if y_lim:
		ylim(*y_lim)
	imgpath = pjoin(path , param_name + "_bar.svg")
	savefig(imgpath)
	imgcode += '<img src="' + imgpath + '" width="500" height="400">'

	return imgcode


# OUPUT to HTML
#----------------
html_page = '<html><body>'



# Swing Amplitudes
#===============================================================================

paramplot = plotForParam("swing_amplitudes", legs, swing_amplitudes)
datatable = htmlTableDataForParam("Swing Amplitudes<BR>(Unit : pixels)", legs, swing_amplitudes)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Swing Durations
#===============================================================================
paramplot = plotForParam("swing_durations", legs, swing_durations)
datatable = htmlTableDataForParam("Swing Durations<BR>(Unit : frames)", legs, swing_durations)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Stance Durations
#===============================================================================
paramplot = plotForParam("stance_durations", legs, stance_durations)
datatable = htmlTableDataForParam("Stance Durations<BR>(Unit : frames)", legs, stance_durations)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Variability in Anterior Extreme Position w.r.t. X-axis
#===============================================================================
paramplot = plotForParam("aepx", legs, aepxs)
datatable = htmlTableDataForParam("AEPx (Anterior Extreme Position w.r.t. X)<BR>(Unit : pixels)", legs, aepxs)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Variability in Posterior Extreme Position w.r.t. X-axis
#===============================================================================
paramplot = plotForParam("pepx", legs, pepxs)
datatable = htmlTableDataForParam("PEPx (Posterior Extreme Position w.r.t. X)<BR>(Unit : pixels)", legs, pepxs)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Contra-lateral Co-ordination Indices
#===============================================================================
my_y_lim = (-1.2, 1.2)
#paramplot = plotForParam("cci", contralateral_pairs, ccis, my_y_lim)
datatable = htmlTableDataForParam("Contra-lateral Co-ordination Indices<BR>(Unit : NONE)", contralateral_pairs, ccis)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Ipsi-lateral Co-ordination Indices
#===============================================================================
my_y_lim = (-1.2, 1.2)
#paramplot = plotForParam("ici", ipsilateral_pairs, icis, my_y_lim)
datatable = htmlTableDataForParam("Ipsi-lateral Co-ordination Indices<BR>(Unit : NONE)", ipsilateral_pairs, icis)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"

# Leg-Body Angles
#===============================================================================
paramplot = plotForParam("lba", legs, lbas)
datatable = htmlTableDataForParam("Leg-Body Angles<BR>(Unit : degrees)", legs, lbas)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"


# Anterior Extreme Angle
#===============================================================================
paramplot = plotForParam("aea", legs, aeas)
datatable = htmlTableDataForParam("Anterior Extreme Angle<BR>(Unit : degrees)", legs, aeas)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"


# Posterior Extreme Angle
#===============================================================================
paramplot = plotForParam("pea", legs, peas)
datatable = htmlTableDataForParam("Posterior Extreme Angle<BR>(Unit : degrees)", legs, peas)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"


# Walking Speeds
#===============================================================================
my_y_lim = (0.0, 30.0)
#paramplot = plotForParam("speed", ['Walk Speeds'], speeds, my_y_lim)
datatable = htmlTableDataForParam("Walking Speeds<BR>(Unit : pixels/frame)", ['Walk Speeds'], speeds)
html_page += HTML.table(zip(*[[datatable], [paramplot]]))
html_page += "<BR><BR>"


html_page += '</body></html>'
with open(pjoin(path, "out.html"), 'w') as f: f.write(html_page)




