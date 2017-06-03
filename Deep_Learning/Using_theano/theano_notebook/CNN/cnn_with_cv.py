import cv2
import theano as th
from theano import function
from theano import shared
from theano import tensor as T
import numpy as np
import pylab
import sys
from theano.tensor.nnet import conv2d
rng_value = sys.argv[1]
#declaring random number generator
rng_value = int(rng_value)
rng = np.random.RandomState(rng_value)
#declaring input tensor
input = T.tensor4(name='input')
#generating and declaring weights
w_shape = (2,3,9,9)
w_bound = np.sqrt(3*9*9)
W = shared(np.asarray(rng.uniform(
			low = -1./w_bound,
			high = 1./w_bound,
			size = w_shape ),
		dtype = input.dtype), name = 'W')
#generating and declaring weights
b_shape = (2,)
b = shared(np.asarray(rng.uniform(
			low = -0.5,
			high = 0.5,
			size = b_shape),
		dtype=input.dtype),
		name = 'b')
conv_out = conv2d(input,W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
filter = function([input],output)

#getting images using opencv
img = cv2.imread('../images/sam.png')
print img
img = np.asarray(img,dtype='float64') / 256.
img_ = img.transpose(2,0,1).reshape(1,3,191,191)
filtered_img = filter(img_)

#plotting original image
pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img)
#pylab.gray();
#plotting filtered images

pylab.subplot(1,3,2);pylab.axis('off');pylab.imshow(filtered_img)
pylab.subplot(1,3,3);pylab.axis('off');pylab.imshow(filtered_img[0,1,:,:])
pylab.show()
new_img = filtered_img

cv2.imshow('sam',new_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()'''