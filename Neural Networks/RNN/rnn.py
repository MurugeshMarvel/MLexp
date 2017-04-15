import numpy as np
import data_matrix

vocablary_size,x_train = data_matrix.data()


class rnn:
	def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
		#assign instance Variable
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate =bptt_truncate
		#Randomly initialise the network
		self.u = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
		self.v = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
		self.w = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))

	def softmax(self,val):
		self.e_x = np.exp(val - np.max(val))
		return (self.e_x/self.e_x.sum())

	def forward_propagation(self,x):
		#total number of steps
		T = len(x)
		#during forward propagation we save all hiden states in a variable 's' coz we need them later

		#we add one additional element to initial hidden, we set it to 0

		s = np.zeros((T+1,self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		#the output at each time step. Again, we save them for later
		o = np.zeros((T,self.word_dim))
		#for each time steps
		for t in np.arange(T):
			#We are indexing U by x[t]. this is the same as multiplying u with a one-ho vector
			s[t] = np.tanh(self.u[:,x[t]] + self.w.dot(s[t-1]))
			o[t] = self.softmax(self.v.dot(s[t]))
		return [o,s]

	#rnn_main.forward_propagation = forward_propagation

	def predict(self,x):
		#perform forward propagation and return index of the highest score
		o,s = self.forward_propagation(x)
		return np.argmax(o,axis=1)

	#rnn_main.predict = predict

	

np.random.seed(10)
model = rnn(vocablary_size)
o,s = model.forward_propagation(x_train[10])
print o.shape
print o