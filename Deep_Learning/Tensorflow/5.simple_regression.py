import tensorflow as tf
import matplotlib.pyplot as plt
import nunpy as np


tf.set_random_seed(1)
np.random.seed(1)


x = np.linspace(-1,1,100)[:, np.newaxis]
noise = np.random.normal(0,0.1, size=x.shape)
y = np.power(x,2) + noise