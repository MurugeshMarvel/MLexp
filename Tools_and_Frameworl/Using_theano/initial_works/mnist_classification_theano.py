#!/usr/bin/python

import gzip
import pickle
import numpy as np
import theano
import theano.tensor as T
from theano import function as function
import matplotlib.pyplot as plt

floatX = theano.config.floatX
train, val, test = pickle.load(gzip.open('/home/murugesan/DEV/Artintell/Deep_Learning/Using_theano/mnist.pkl.gz'))
x_train,y_train = train
'''
print y_train[2]
plt.imshow(x_train[2].reshape((28,28)),cmap='gray')
plt.show()
'''


np.random.seed(12)


class mnist_classifier(object):
	def __init__(self,n_features):
		hidden_layer_size = 5
		regularisation_val = 0.001
		rng = np.random.RandomState(23) 


		X = T.fmatrix('X')
		Y = T.fvector('Y')
		learning_rate = theano.tensor.fscalar('learning_rate')

		w_hidden_vals = np.asarray(rng.normal(loc=0.0,scale=0.1, size=(n_features,hidden_layer_size)),dtype=floatX)
		w_hidden = theano.shared(w_hidden_vals,'w_hidden')

		#calculating the hidden layer
		hidden = T.dot(X,w_hidden)
		hidden = T.nnet.sigmoid(hidden)

		#hidden to output weights
		w_output_vals = np.asarray(rng.normal(loc=0,scale=0.1,size = (hidden_layer_size,1)),dtype=floatX)
		w_output = theano.shared(w_output_vals,'w_output')

		#calculating the predicted value
		predicted_value = T.dot(hidden,w_output)
		predicted_value = T.nnet.sigmoid(predicted_value)

		cost = T.sqr(predicted_value - Y).sum()
		cost += regularisation_val*(T.sqr(w_hidden).sum() + T.sqr(w_output).sum())

		params = [w_hidden,w_output]
		gradient = T.grad(cost,params)
		updates = [(a, a-(learning_rate*b)) for a,b in zip(params,gradient)]

		self.train = function([X,Y,learning_rate],[cost,predicted_value],updates=updates)
        #self.test = function([X,Y], [cost, predicted_value], allow_input_downcast=True)
        #self.predict = function([X],[Y])

'''def read_dataset(file):
	dataset = []
	for dat in file:
		o,l = dat
		print o
read_dataset(train)
'''
if __name__ == '__main__':
	learningrate = 0.1
	epoch =20
	n_features = len(x_train[1])
	classifier = mnist_classifier(n_features)

	for epo in range(epoch):
		cost_sum = 0
		correct = 0

		for x_train,y_train in zip(x_train,y_train):
			cost, predicted_value = classifier.train(x_train, y_train, learningrate)
			cost_sum += cost
			if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
				correct += 1
				print "Epoch: " + str(epoch) + ", Training_cost: " + str(cost_sum) + ", Training_accuracy: " + str(float(correct) / len(data_train))
