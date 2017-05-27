## this program uses the web traffic data over a particular period
## The parameters are hours and the traffic count
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sc.random.seed(3)
argv = False
if len(sys.argv) > 1:
    argv = True
    option = str(sys.argv[1])
data = sc.genfromtxt(os.path.join('web_traffic.tsv'), delimiter='\t')
colors = ['g','r','b','m','k']
linestyles = ['-','-.','--',':','-']
x = data[:,0]
y = data[:,1]
x = x[~sc.isnan(y)]
y = y[~sc.isnan(y)]
def visualise(x,y, models, mx = None, ymax=None,
                xmin = None,save=False, show=True,threshold=None):

    plt.figure(num=None, figsize = (8,6))
    plt.clf()
    plt.scatter(x,y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel('Time')
    plt.ylabel('Hits/Hour')
    plt.xticks([w*7*24 for w in range(10)], ['week %i ' %w for w in range(10)])
    if models:
        if mx is None:
            if mx == None:
                mx = sc.linspace(0, x[-1], 1000)
            for model, style, color in zip(models, linestyles, colors):
                plt.plot(mx, model(mx), linestyle = style, linewidth=2,c = color)
            plt.legend(['d = %i' %m.order for m in models], loc = 'upper left')
    plt.autoscale(tight = True)
    if threshold:
        plt.plot(x[threshold],y[threshold],'or')
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin = xmin)
    plt.grid(True, linestyle='-', color='0.75')
    if show:
        plt.show()
    if save:
        plt.savefig(figure.png)
def error(f,x,y):
    return sc.sum((f(x)-y) **2)

if argv:
    if option == 'v':
        visualise(x,y , None)
fp1, res1, rank1, sv1, rcond1 = sc.polyfit(x,y,1, full=True)
print "Model Parameters of fp1 is ", fp1
print "Error of the model of fp1: ", res1
f1 = sc.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = sc.polyfit(x,y,2, full=True)
print "Model Parameters of fp2:",fp2
print "Error of the model of fp2: ", res2
f2 = sc.poly1d(fp2)
inflec_point = 3.5
inflec_value = inflec_point * 7 * 24 #calculating for weeks
print inflec_value
print x
x_before  = x[:inflec_value]
y_before = y[:inflec_value]
x_after = x[inflec_value:]
y_after = y[inflec_value:]

f_before = sc.poly1d(sc.polyfit(x_before, y_before,1))
f_after = sc.poly1d(sc.polyfit(x_after, y_after,1))
visualise(x,y,[f_before, f_after],threshold=inflec_value)
print 'error for order 1 :', error(f1,x,y)
print '\t \t order 2 :',error(f2,x,y)
print 'adding separated error of the inflection'
print 'Error is ',(error(f_before,x_before,y_before) + error(f_after, x_after, y_after))

#Slicing Data for testing and training
ratio = 0.3
split_id = int(ratio*len(x_after))
shuffled = sc.random.permutation(list(range(len(x_after))))
test = sorted(shuffled[:split_id])
train = sorted(shuffled[split_id:])
fbt1 = sc.poly1d(sc.polyfit(x_after[train], y_after[train], 1))
fbt2 = sc.poly1d(sc.polyfit(x_after[train], y_after[train],1))

