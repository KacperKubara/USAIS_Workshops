# K-NN classification with k-fold cross validation
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Read Data
dataset = pd.read_csv("Social_Network_Ads.csv") 

# Choose which features to use
x = dataset.iloc[:, 2:4].values # Features 
y = dataset.iloc[:, 4].values   # Output value

# Split data into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
sc_x    = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.transform(x_test)

# Train Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(x_train, y_train)

# Predict Results
y_pred = classifier.predict(x_test)

# Measure accuracy with k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x, y = y, cv = 10)
mean_accuracy = accuracies.mean()
standard_deviation = accuracies.std()

# Visualise Results - code taken from www.superdatascience.com/machine-learning
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN Classification (Test set)')
plt.xlabel('Age(scaled)')
plt.ylabel('Estimated Salary(scaled)')
plt.legend()
plt.show()