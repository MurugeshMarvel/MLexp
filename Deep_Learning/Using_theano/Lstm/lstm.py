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
def get_dataset(name):
	return datasets[name][0], datasets[name][1]
def zipp(params, tparams):
	for k,v in params.items():
		tparams[k].set_value(v)
def unzip(zipped):
	new_params = OrderedDict()
	for k,v in zipped.items():
		new_params[k] = v.get_value()
	return new_params

def dropout_layer(state_before, use_noise, trng):
	proj = t.switch(use_noise, (state_before * 
					trng.binomial(state_before.shape,
								p=0.5, n=1,
								dtype=state_before.dtype)),
					state_before * 0.5)
	return proj

def _p(pp, name):
	return '%s_%s' % (pp, name)

def init_parames(options):
	params = OrderedDict()
	randn = np.random.rand(options['n_words'],
							options['dim_proj'],)
	params['Wemb'] = (0.01 * randn).astype(config.floatX)
	params = get_layer(options['encoder'])[0](options, params,
									prefix=options['encoder'])
	params['U'] = 0.01 * np.random.randn(options['dim_proj'],
										)