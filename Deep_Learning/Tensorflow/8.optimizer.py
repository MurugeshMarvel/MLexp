import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

LR = 0.01
BATCH_SIZE = 32

x = np.linspace(-1,1,100)[:,np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x,2) + noise

plt.scatter(x,y )
plt.show()

class Net:
	def __init__(self, opt, **kwargs):
		self.x = tf.placeholder(tf.float32, [None,1])
		self.y = tf.placeholder(tf.float32, [None, 1])
		l = tf.layers.dense(self.x, 20, tf.nn.relu)