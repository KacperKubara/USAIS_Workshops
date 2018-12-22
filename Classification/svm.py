# SVM classification with k-fold cross validation and PCA
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Read Data
from sklearn.datasets import load_iris
dataset = load_iris() 

# Choose which features to use
x = dataset["data"]   # It has 4 features - with PCA we will reduce it to 3 for 3D visualisation 
y = dataset["target"] # Output value

# Split data into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# PCA to reduce dimensions to 3 (from 4)
from sklearn.decomposition import PCA
pca = PCA(n_components = 3, random_state = 42)
x_train = pca.fit_transform(x_train)
x_test  = pca.transform(x_test)

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
sc_x    = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.transform(x_test)

# Train Model
from sklearn.svm import SVC
classifier = SVC(random_state = 42)
classifier.fit(x_train, y_train)

# Predict Results
y_pred = classifier.predict(x_test)

# Measure accuracy with Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x, y = y, cv = 10)
mean_accuracy = accuracies.mean()
standard_deviation = accuracies.std()

# Merge output values with features into pandas dataframe
# It is just used to make a plotting part clearer to read
x_test  = sc_x.inverse_transform(x_test) # Return back to non-scaled values
pred_df = pd.DataFrame({'x0': x_test[:,0], 'x1': x_test[:,1],'x2': x_test[:,2], 'y': y_pred})

# Visualise Results
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('X2')
ax.set_title('Predicted Datapoints')
temp_df = pred_df[pred_df['y'] == 0] # Take only rows from dataset containing y = 0
ax.scatter(temp_df['x0'], temp_df['x1'], temp_df['x2'], color = 'r')
temp_df = pred_df[pred_df['y'] == 1]
ax.scatter(temp_df['x0'], temp_df['x1'], temp_df['x2'], color = 'g')
temp_df = pred_df[pred_df['y'] == 2]
ax.scatter(temp_df['x0'], temp_df['x1'], temp_df['x2'], color = 'b')
