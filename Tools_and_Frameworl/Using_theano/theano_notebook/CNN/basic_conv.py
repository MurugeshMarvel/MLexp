import theano
from theano import tensor
from theano import function
import numpy as np
import pylab
from PIL import Image
from theano.tensor.nnet import conv2d

#declaring random number generator
rng = np.random.RandomState(341)

#declaring input tensor
input = tensor.tensor4(name='input')

w_shape = (2,3,9,9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(np.asarray(rng.uniform(
			low = -1./w_bound,
			high = 1./w_bound,
			size = w_shape ),
		dtype = input.dtype), name = 'W')

b_shape = (2,)
b = theano.shared(np.asarray(rng.uniform(
			low = -0.5,
			high = 0.5,
			size = b_shape),
		dtype=input.dtype),
		name = 'b')
conv_out = conv2d(input,W)
output = tensor.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
filter = function([input],output)

##Opening image for experiment
img = Image.open(open('../images/sam.png'))
img = np.asarray(img,dtype='float64') / 256.
img_ = img.transpose(2,0,1).reshape(1,3,191,191)
filtered_img = filter(img_)

#plotting original image
pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img)
#pylab.gray();
#plotting filtered images
pylab.subplot(1,3,2);pylab.axis('off');pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1,3,3);pylab.axis('off');pylab.imshow(filtered_img[0,1,:,:])
pylab.show()
