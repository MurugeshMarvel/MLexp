import theano
import numpy as np
import theano.tensor as T

x = T.vector('x')
target = T.scalar('target')

w = theano.shared(np.asarray([0.1,0.1]),'w')
y = (x*w).sum()



cost = T.sqr(target-y)
gradient = T.grad(cost,[w])


w_update = w - (0.2 * gradient[0])
z =gradient[0]
updates = [(w,w_update)]

miniman_func = theano.function([x,target],[y,z],updates = updates)


for i in range(20):
	predicted_output,out_gradient = miniman_func([1.0,1.0],20)
	print predicted_output,'--',out_gradient
