import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

names = ['sepal_length','sepal_width','petal_length','petal_width','class']

df = pd.read_csv('dataset/iris.data', header = None, names= None)
lable_dict = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
lable_list = []
for i in df[4]:
    lable_list.append(lable_dict[i])
print lable_list
X = np.array(df.ix[:,0:4])
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print X_train
'''
#plotting_data
colors = ['red','green','blue']w
fig = plt.figure(figsize = (8,8))
label_dict = {}
plt.scatter(df[2],df[3], c=lable_list, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
'''
def train(x_train, y_train):
    #do_nothing
    return
def predict(x_train, y_train, y_test, k):
    distance = []
    targets = []
