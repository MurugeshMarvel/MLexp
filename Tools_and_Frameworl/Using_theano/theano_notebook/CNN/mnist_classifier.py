import theano as th
from theano import tensor as T
from theano import function
from theano import shared
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import cPickle
import numpy as np
import gzip
import timeit
import os

class lenet(object):
	def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape)
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
				   np.prod(poolsize)) 
		w_bound = np.sqrt(6./(fan_out + fan_in))
		self.W = shared(np.asarray(
						rng.uniform(low=-w_bound,
									high=w_bound,
									size = filter_shape),
								dtype = th.config.floatX),
							borrow = True)
		b_values = np.zeros((filter_shape[0],),dtype=th.config.floatX)
		self.b = th.shared(value = b_values, borrow=True)

		conv_out = conv2d(input= input,
						  filters = self.W,
						  filter_shape =filter_shape,
						  input_shape = image_shape)
		pooled_out = downsample.max_pool_2d(input = conv_out,
								  ds = poolsize,
								  ignore_border=True)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
		self.params = [self.W,self.b]
		self.input = input
def load_data(name):
	with gzip.open(name,'rb') as file:
		try:
			train_set, valid_set, test_set = cPickle.load(file,encoding='latin1')
		except:
			train_set , valid_set, test_set = cPickle.load(file)
	def making_shared(data_xy, borrow = True):
		data_x, data_y = data_xy
		shared_x = shared(np.asarray(data_x, 
						dtype= th.config.floatX), borrow = borrow)
		shared_y = shared(np.asarray(data_y,
						dtype = th.config.floatX) ,borrow = borrow)
		return shared_x , T.cast(shared_y,'int32')
	train_set_x, train_set_y = making_shared(train_set)
	valid_set_x, valid_set_y = making_shared(valid_set)
	test_set_x, test_set_y = making_shared(test_set)
	rval = [(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
	return rval
class hiddenlayer(object):
	def __init__(self,rng,input,n_in,n_out, activation = T.tanh,W=None, b=None):
		self.input= input
		if W is None:
			w_values = np.asarray(
						rng.uniform(
							low=-np.sqrt(6./(n_in + n_out)),
							high = np.sqrt(6./(n_in + n_out)),
							size = (n_in,n_out)),
						dtype = th.config.floatX)
			if activation == T.nnet.sigmoid:
				w_values *= 4
			W = shared(value = w_values, name='W',
						borrow = True)
		if b is None:
			b_values = np.zeros((n_out,), 
								dtype = th.config.floatX)
			b = shared(value=b_values, name='b', borrow = True)
		self.W = W
		self.b = b
		self.lin_output = T.dot(input,W) + b
		self.output = (self.lin_output if activation is None
					else activation(self.lin_output))
		self.params = [self.W,self.b]

class regression(object):
	def __init__(self,rng,input,n_in, n_out):
		self.W = shared(value = np.zeros(
					(n_in,n_out),
					dtype =th.config.floatX),
					name = 'W',
					borrow = True)

		self.b = shared(value=np.zeros((n_out),
								dtype=th.config.floatX),
							name = 'b',
							borrow = True)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W,self.b]
		self.input = input

	def negative_log_likelihood(self,y):
		ng_error = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
		return ng_error
	def errors(self,y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


def main_network(batch_size=500,nkerns=[20,50], learning_rate=0.1,
				rng=np.random.RandomState(23455),dataset = 'mnist.pkl.gz',
				n_epochs=200):
	rng = np.random.RandomState(23455)
	datasets = load_data(dataset)
	print ("Dataset loaded successfully")
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	n_valid_batches //= batch_size
	n_test_batches //= batch_size
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')
	print 'Building the model'
	layer0_input = x.reshape((batch_size,1,28,28))
	layer0 = lenet(rng,
				   input = layer0_input,
				   image_shape = (batch_size,1,28,28),
				   filter_shape = (nkerns[0],1,5,5))
	layer1 = lenet(rng,
				   input = layer0.output,
				   image_shape = (batch_size, nkerns[0],12,12),
				   filter_shape = (nkerns[1],nkerns[0],5,5)
				   )
	layer2_input = layer1.output.flatten(2)
	print layer2_input
	layer2 = hiddenlayer(rng,
						input=layer2_input,
						n_in = nkerns[1] *4 *4,
						n_out = 500,
						activation = T.tanh)
	layer3 = regression(rng, 
						input = layer2.output,
						n_in = 500,
						n_out = 10)
	cost = layer3.negative_log_likelihood(y)
	#testing model
	test_model = function([index],
						  layer3.errors(y),
						  givens={
						  x:test_set_x[index*batch_size : (index+1)*batch_size ],
						  y:test_set_y[index * batch_size :(index +1)* batch_size]
						  })
	valid_model = function([index],
							layer3.errors(y),
							givens={
							x:valid_set_x[index*batch_size : (index+1)*batch_size],
							y:valid_set_y[index*batch_size : (index+1)*batch_size]})
	params = layer3.params + layer2.params + layer1.params +layer0.params
	grads = T.grad(cost, params)
	updates = [(param_i ,param_i - learning_rate * grad_i)
				for param_i,grad_i in zip(params,grads)]
	train_model = function([index],
							cost,
							updates = updates,
							givens = {
							x : train_set_x[index*batch_size : (index+1)*batch_size],
							y : train_set_y[index*batch_size : (index+1)*batch_size]
							})
	print ('././ Training')
	patience = 10000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience // 2)
	best_validation_loss = np.inf
	best_iter = 0
	test_score = 0.0
	start_time = timeit.default_timer()
	epoch=0
	done_looping = False
	while (epoch< n_epochs) and (not done_looping):
		epoch = epoch+1
		for minibatch_index in range(n_train_batches):
			iter = (epoch - 1)  * n_train_batches + minibatch_index
			if iter % 100 == 0:
				print('training @ iter = ', iter)
			cost_ij = train_model(minibatch_index)
			if (iter + 1) % validation_frequency == 0:
				validation_losses = [valid_model(i) for i
								in range(n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					(epoch, minibatch_index + 1, n_train_batches,this_validation_loss * 100.))
				if this_validation_loss < best_validation_loss:
					if this_validation_loss < best_validation_loss *  \
					improvement_threshold:
						patience = max(patience, iter * patience_increase)
						best_validation_loss = this_validation_loss
						best_iter = iter
					test_losses = [
						test_model(i)
						for i in range(n_test_batches)]
					test_score = np.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test error of ''best model %f %%') %(epoch, minibatch_index + 1, n_train_batches,
							test_score * 100.))
			if patience <= iter:
				done_looping = True
				break
	end_time = timeit.default_timer()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, ''with test performance %f %%' %
		(best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print(('The code for file ' +os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.)))

if __name__ == '__main__':
	main_network()
