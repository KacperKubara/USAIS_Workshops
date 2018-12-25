# K-NN classification with k-fold cross validation
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Read Data
dataset = pd.read_csv("Mall_Customers.csv") 

# Choose which features to use
x = dataset.iloc[:, 3:5].values # Features - Age, annual income (k$)
"""
 K-Means clustering is unsupervised ML algorithm. It means that
it will figure the classes on its own (contrary to ML classification algorithms)
The purpose of clustering is to find out if a certain group of data points have
similar charactertics, i.e. if they create a cluster
Therefore, we won't need train-test split. Feature scaling might also be unnecessary
as the 2 features have v.similar values
"""
inertia = []
cluster_count = []
from sklearn.cluster import KMeans
for i in range(1,15):
    cluster = KMeans(n_clusters = i, random_state = 42)
    cluster.fit(x)
    inertia.append(cluster.inertia_)
    cluster_count.append(i)
fig0, ax0 = plt.subplots()
ax0.plot(cluster_count, inertia)
ax0.set_title("Elbow Method")
ax0.set_xlabel("No. clusters")
ax0.set_ylabel("WCSS")
# By looking on the elbow method plot,
# let's choose n_clusters = 5

# Cluster the data and assign labels ('classes')
cluster = KMeans(n_clusters = 5, random_state = 42)
y_pred = cluster.fit_predict(x)

# Visualise Results
from matplotlib import colors 
my_colors = list(colors.BASE_COLORS.keys())
fig1, ax1 = plt.subplots()
ax1.set_title("Clusters")
ax1.set_xlabel("Annual Income")
ax1.set_ylabel("Spending Score")
# Plotting cluster
for i, label in enumerate(cluster.labels_):
    ax1.scatter(x[i, 0], x[i, 1], c = my_colors[label])
ax1.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1],
            s = 300, c = 'yellow', label = 'Centroids')
