import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv2d



import numpy as np



rng = np.random.RandomState(23455)


input = T.tensor4(name = "input")

shape = (2,3,9,9)

bound = np.sqrt(3*9*9)
w = theano.shared(numpy.asarray(rng.uniform(low=-1.0/bound, high=1.0/bound),dtype=input.dtype),name = 'W')

b_shp = (2,)
b = theano.shared(np.asarray(rng.uniform(low=-1.0,high=1.0),dtype=input.dtype),name = 'b')

conv_out = conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))

f = theano.function([input],output)