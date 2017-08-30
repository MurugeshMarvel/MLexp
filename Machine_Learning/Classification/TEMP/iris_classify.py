import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
color = ['r','g','m']
marker = ['>','o','x']

for c,m,t  in zip(color,marker,[f for f in range(3)]):
    plt.scatter(features[target == t,0], features[target == t,1], marker=m,c=c)
labels = target_names[target]
plength = features[:,2]
is_setosa = (labels == 'setosa')
