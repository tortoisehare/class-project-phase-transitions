#read and pre-process input data
import os,sys
import numpy as np

''' Reads inputs X, y, alpha --> stores in numpy arrays
X = matrix of inputs (#parameters x #examples)
Y = matrix of y values (1 x #examples)
alpha = matix of alpha values (1 x #examples)
'''

try:
	alphafilename = sys.argv[1]
	xfilename = sys.argv[2]
	yfilename = sys.argv[3]
	numlines = int(sys.argv[4])
	startind = int(sys.argv[5])
except:
	print '<alpha filename> <x filename> <y filename> <number of lines to import> <starting index>'
	exit()

alphalist = list()
ylist = list()
xlist = list()

count = 0
with open(alphafilename) as alphafile:
	for i, line in enumerate(alphafile):
		if count < numlines:
			if i >= startind:
				alphalist.append(float(line))
				count += 1
		else:
			break

count = 0
with open(yfilename) as yfile:
	for i, line in enumerate(yfile):
		if count < numlines:
			if i >= startind:
				ylist.append(float(line))
				count += 1
		else:
			break

count = 0
with open(xfilename) as xfile:
	for i, line in enumerate(xfile):
		if count < numlines:
			if i >= startind:
				xlist.append(np.array(map(float,list(line.split()))))
				count += 1
		else:
			break

alphalist = np.array(alphalist)
ylist = np.array(ylist)
xlist = np.array(xlist)
