import pandas
from matplotlib import pyplot as plt
from math import sqrt

class lin_regres:
    def __init__(self, x, y):
        self.x = list()
        for i in x:
            temp_x = float(i)
            self.x.append(temp_x)

        self.y =list()
        for i in y:
            temp_y = str(i).replace(',','.')
            self.y.append(float(temp_y))


    def scatter_plot(self):
        plt.scatter(self.x, self.y)
        plt.show()

    def coefficient(self):
        def mean(alist):
            return (sum(alist) / float(len(alist)))

        def variance(mean, value):
            return sum([(x - mean)**2 for x in value])
        def covariance():
            covar = 0.0
            for i in range(len(self.x)):
                covar += (self.x[i] - mean(self.x)) * (self.y[i] - mean(self.y))
            return covar
        self.b1 = covariance()/ variance( mean(self.x) , self.x)
        self.b0 = mean(self.y) - self.b1 * mean(self.x)
        #return [b0, b1]
    def predict(self, new_x):
        predicted_val = self.b0 + (self.b1 * new_x)
        return predicted_val
dat = pandas.read_csv("autoinsure.txt",sep='\t',header=1, names = ["claims","pay"])
claims = list(dat['claims'])
#print ([type(i) for i in claims])
pay = list(dat['pay'])
set_obj = lin_regres(claims, pay)
set_obj.coefficient()
inp = int(input("Enter the number of claims"))
print ("The approximate payable amount is: ",set_obj.predict(inp))
