import theano
import theano.tensor as T
import matplotlib.pyplot as plt
x = T.dmatrix('x')
s1 = 1/(1 + T.exp(-x))
s2 = (1 + T.tanh(x/2)) / 2

logistic = theano.function([x],s1)
logtan = theano.function([x],s2)
results = logistic([[0,1],[-1,-2]])

print results 
print "Log tan is"
print logtan([[0,1],[-1,-2]])

plt.plot(results)
plt.show()