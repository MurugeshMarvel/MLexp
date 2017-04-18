## this program uses the web traffic data over a particular period
## The parameters are hours and the traffic count
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
sp.random.seed(3)
data = sc.genfromtxt(os.path.join('web_traffic.tsv'), delimiter='\t')
def visualise(data):
    colors = ['g','k','b','m','r']
    linestyles = ['-','-.','--',':',-'']
    x = data[:,0]
    y = data[:,1]
    plt.figure(num=None, figsize = (8,6))
    plt.clf()
    plt.scatter(x,y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel('Time')
    plt.ylabel('Hits/Hour')
    plt.xticks([w*7*24 for w in range(10)], ['week %i ' %w for w in range(10)])
    if models:
        if mx is None:
            mx = 
