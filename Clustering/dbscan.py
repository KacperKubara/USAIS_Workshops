# K-NN classification with k-fold cross validation
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Read Data
dataset = pd.read_csv("Mall_Customers.csv") 

# Choose which features to use
x = dataset.iloc[:, 3:5].values # Features - Age, annual income (k$)
"""
 DBSCAN clustering is unsupervised ML algorithm. It means that
it will figure the classes on its own (contrary to ML classification algorithms)
The purpose of clustering is to find out if a certain group of data points have
similar charactertics, i.e. if they create a cluster
Therefore, we won't need train-test split. Feature scaling might also be unnecessary
as the 2 features have v.similar values
Reference: https://scikit-learn.org/stable/modules/clustering.html#dbscan
"""

# Cluster the data and assign labels ('classes')
from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps = 5, min_samples = 5) # Need to choose the parameters wisely
y_pred = cluster.fit_predict(x) # Noisy samples are given label '-1'

# Visualise Results
from matplotlib import colors 
my_colors = list(colors.BASE_COLORS.keys())
fig1, ax1 = plt.subplots()
ax1.set_title("Clusters")
ax1.set_xlabel("Annual Income")
ax1.set_ylabel("Spending Score")
# Plotting cluster
for i, label in enumerate(cluster.labels_):
    if label > 0:
        ax1.scatter(x[i, 0], x[i, 1], c = my_colors[label])
    else:
        ax1.scatter(x[i, 0], x[i, 1], c = 'k')