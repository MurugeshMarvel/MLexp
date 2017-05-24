from collections import OrderedDict
import theano as th
import numpy as np
import sys
import time
import six.moves.cPickle as pickle
from __future__ import print_function
from theano import tensor as t
from theano import function 
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data

seed = 123
dataset = {"imdb" : (data.load_data, data.prepare_data)}

np.random.seed(seed)

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)
def get_minibatches_idx(n, minibatch_size, shuffle=False):
	idx_list = np.arange(n, dtype="int32")
	if shuffle:
		np.random.shuffle(idx_list)
	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if (minibatch_start != n):
		minibatches.append(idx_list[minibatch_start:])
	return zip(range(len(minibatches)), minibatches)