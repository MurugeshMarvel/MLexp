#!/usr/bin/python

import numpy as np
#collection data

x = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,1,1]])
y = np.array([[0],[0],[0],[0],[0],[1]])

print x

print "The output "

print y

#building Model


epoch_no = 60000

weight0 = 2*np.random.random((3,4)) - 1
weight1 = 2*np.random.random((4,1)) - 1

print "############"
print weight0
print "############"
print weight1

#step 3 train model

def nonlin(x, deriv=False):
	if deriv == 'True':
		return x(1-x)

	else:
		return 1/(1+np.exp(-x))

for j in xrange(epoch_no):
	#feed forward through layers 1,2,3
	l0 = x
	l1 = nonlin(np.dot(l0,weight0))
	l2 = nonlin(np.dot(l1,weight1))

	#calculating difference between the Original output and the predicted output

	error = y - l2
	#print error
	l2_del = error * nonlin(l2, deriv=True)

	l1_error = l2_del.dot(weight1.T)

	l1_delta = error * nonlin(l1,deriv=True)

	weight0 += l0.T.dot(l1_delta)
	weight1 += l1.T.dot(l2_del)

x1 = np.array([[1,1,1]])
lay0 = x1
lay1 = nonlin(np.dot(lay0,weight0))
lay2 = nonlin(np.dot(lay1,weight1))

error = y - lay2
print "######"
print y
print "#####"
print lay2
