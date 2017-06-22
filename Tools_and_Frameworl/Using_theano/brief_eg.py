from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))

f = function([], rv_u)
g = function([],rv_n, no_default_updates=True)
nearly_zeros = function([], rv_u+rv_n - 2 * rv_u)