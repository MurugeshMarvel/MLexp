import theano
import numpy as np
import theano.tensor as T
from theano import function
import lasagne
import matplotlib.pyplot as plt

import gzip
import pickle
np.random.seed(42)

train, val , test = pickle.load(gzip.open('/home/murugesan/DEV/Artintell/Deep_Learning/Using_theano/mnist.pkl.gz'))
x_train,y_train = train
x_val , y_val = val
x_test, y_test = test

'''plt.figure(figsize=(12,3))
for i in range(10):
	plt.subplot(1,10,i+1)
	plt.imshow(x_train[i].reshape((28,28)),cmap='gray',interpolation='nearest')
	plt.axis('off')
plt.hold()
plt.figure(figsize=(12,3))
for i in range(10):
	plt.subplot(1,10,i+1)
	plt.imshow(x_test[i].reshape((28,28)),cmap='gray',interpolation='nearest')
	plt.axis('off')

plt.show()'''

'''


batch_size = 32

c =  batch_gen(x_train,y_train,batch_size)

for i in c:
	print i'''


def batch_gen(x,y,n):
	while True:
		idx = np.random.choice(len(y),n)
		yield x[idx].astype('float32'),y[idx].astype('int32')


l_in  =  lasagne.layers.InputLayer((None,784))
l_out = lasagne.layers.DenseLayer(
	l_in,
	num_units = 10,
	nonlinearity = lasagne.nonlinearities.softmax)

x_sym = T.matrix()
y_sym = T.ivector() #integer vector

output = lasagne.layers.get_output(l_out, x_sym)
print output
pred = output.argmax(-1)

loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym))
acc = T.mean(T.eq(pred,y_sym))

params = lasagne.layers.get_all_params(l_out)
grad = T.grad(loss,params)
updates = lasagne.updates.sgd(grad,params,learning_rate=0.05)

f_train = theano.function([x_sym,y_sym],[loss,acc],updates=updates)
f_val = theano.function([x_sym,y_sym],[loss,acc])
f_predict = theano.function([x_sym],[output])


batch_size = 64
n_batches = len(x_train) // batch_size
n_val_batches = len(x_val) // batch_size

train_batches = batch_gen(x_train,y_train,batch_size)
val_batches = batch_gen(x_val,y_val,batch_size)
test_batches = batch_gen(x_test,y_test,batch_size)

#plotting an image and corresponding label to verify they match
x, y = next(train_batches)
'''plt.imshow(x[0].reshape((28,28)),cmap = 'gray',interpolation="nearest")
print y[0]
plt.show()'''

for epoch in range(10):
	train_loss = 0
	train_acc = 0
	for _ in range(n_batches):
		x,y = next(train_batches)
		x_v,y_v = next(val_batches)
		#print len(x[1])
		loss,acc = f_train(x,y)
		act_pre = f_predict(x_v)
		train_loss += loss
		train_acc += acc

	train_loss /= n_batches
	train_acc /= n_batches
	val_loss = 0
	val_acc = 0
	for _  in range(n_batches):
		x,y = next(val_batches)
		loss,acc = f_val(x,y)
		val_loss += loss
		val_acc += acc

	val_loss /= n_batches
	val_acc /= n_batches
	print act_pre

	print ("Epoch {}, Train (val) loss {:.03f} ({:.03f}) ratio {:.03f}".format(epoch, train_loss,val_loss, val_loss/train_loss))
	print ("Train (val) accuracy {:.03f} ({:.03f})".format(train_acc,val_acc))
i=0
