import os
import theano as th
import theano.tensor as T
import timeit
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class lenetconvpoollayer(object):
    def __init__(self,rng,input,filter_shape, image_shape, poolsize=(2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
        w_bound = np.sqrt(6./(fan_in + fan_out))
        self.W = th.shared(np.asarray(rng.uniform(low = -w_bound
                            , high = w_bound, size = filter_shape), dtype = th.config.floatX),
                            borrow = True)
        b_values = np.zeros((filter_shape[0],), dtype= th.config.floatX)
        self.b = th.shared(value=b_values, borrow=True)

        conv_out = conv2d(input=input, filters = self.W, filter_shape = filter_shape, input_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds = poolsize, ignore_border = True)
        self.output  = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W, self.b]
        self.input = input

def evaluate_lenet(learning_rate = 0.1, n_epochs=200, dataset = 'mnist.pkl.gz',
                    nkerns = [20,50], batch_size = 500):
        rng = np.random.RandomState(23455)
        datasets = load_data(dataset)
        train_set_x,train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x , test_set_y = datasets[2]

        #compute no of minibatches for training, validating and testing
        n_train_batches = train_set_x.get_value(borrow = True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow = True).shape[0] / batch_size

        index = T.lscalar()

        x = T.matrix('x')
        y = T.ivector('y')

        print 'Building the model'

        layer0_input = x.reshape((batch_size,1,28,28))

        layer0 = lenetconvpoollayer(rng, input = layer0_input, image_shape = (batch_size,1,28,28),
                                    filter_shape = (nkerns[0],1,5,5),
                                    poolsize = (2,2))
        layer1 = lenetconvpoollayer(rng, input = layer0.output, image_shape=(batch_size, nkerns[0],12,12),
                                    filter_shape = (nkerns[1], nkerns[0],5,5), poolsize = (2,2))
        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(rng, input = layer2_input, n_in = nkerns[1] * 4 *4,
                                n_out = 500, activation = T.tanh)
        layer3 = LogisticRegression(input = layer2.output, n_in = 500, n_out =10)
        cost = layer3.negative_log_likelihood(y)
        test_model = th.function([index], layer3.errors(y),
                                givens = {
                                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: test_set_y[index * batch_size: (index+1) * batch_size]
                                })
        valid_model = th.function([index], layer3.errors(y),
                                givens = {
                                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                y: valid_set_y[index * batch_size : (index + 1 ) * batch_size]
                                })
        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)

        updates = [(params_i, params_i - learning_rate * grad_i) for params_i, grad_i in zip(params, grads)]
        train_model = th.function([index], cost, updates = updates,
                                givens = {
                                x: train_set_x[index * batch_size : (index + 1) * batch_size],
                                y: train_set_y[index * batch_size : (index + 1) * batch_size]
                                })
        print '......training'
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches , patience // 2)
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch = 1
            for minibatches_index in range(n_train_batches):
                iter = (epoch - 1) * n_train_batches + minibatches_index
                if iter % 100 == 0:
                    print  ('Training @ iter = ',iter)
                cost_ij = train_model(minibatches_index)
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [valid_model(i) for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print ('epoch %i, minibatches %i%i, validation errors %f %%' %(epoch, minibatches_index +1, n_train_batches,
                            this_validation_loss * 100.))
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                            patience = max(patience , iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        test_losses = [
                            test_model(i) for i in range(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print (('...epoch %i, mininbatch %i/%i, test error of ''best model %f %%')
                                %(epoch, minibatches_index + 1, n_train_batches, test_score * 100.))
                if patience <= iter:
                    done_looping = True
                    break
        end_time = timeit.default_timer()
        print ('Optimization complete')
        print ('ran for %.2fm' %((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet()
def experiment(state, channel):
    evaluate_lenet(state.learning_rate, dataset=state.dataset)
