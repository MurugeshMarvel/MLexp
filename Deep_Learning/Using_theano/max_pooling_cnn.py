#max pooling in theano
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import numpy as np
input = T.dtensor4('input')
maxpool_shape = (2,2)
pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
func =  theano.function([input],pool_out)

invals = np.random.RandomState(1).rand(3,2,5,5)
print invals
print '*'*20
print 'With Ignore_border set to Ture:'
print 'invals[0, 0,: ,:] = \n', invals[0,0,:,:]
print 'outputs[0, 0,: ,:] = \n', func(invals)[0,0,:,:]
print '*'*50
print 'With ignore border set to false'
pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=False)
func = theano.function([input], pool_out)
print 'invals[1,0,:,:] = \n',invals[1,0,:,:]
print 'Ouputs[1,0,:,:] = \n', func(invals)[1,0,:,:]
