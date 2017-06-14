import gzip
import pickle
import numpy as np


train , val, test = pickle.load(gzip.open('/home/murugesan/DEV/Artintell/Deep_Learning/Using_theano/mnist.pkl.gz'))

train_file = open('train_set.txt','w')
x,y = train
data = np.asarray([])
for i in range(20):
	for j in range(len(x[i])):
		train_file.write(str(x[i][j]))
		train_file.write(' ')
	train_file.write('\n')


train_file.close()
