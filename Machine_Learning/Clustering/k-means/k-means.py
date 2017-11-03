from numpy import *

def loaddata(filename):
	data_mat = []
	fr = open(filename)
	for line in fr.readlines():
		curline = line.strip().split('\t')
		fltline = map(float, curline)
		data_mat.append(fltline)
	return data_mat

data = loaddata("testSet.txt")
print len(data)