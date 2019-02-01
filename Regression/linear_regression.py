import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Read Data
from sklearn.datasets import load_diabetes 
dataset = load_diabetes() 

# Choose which features to use
x = dataset["data"][:, 2] # using BMI feature
y = dataset["target"]     # output value

# Split data into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# random_state = 42 to get the reproducible results
# test_size = 0.2, because small dataset

# Data Preprocessing
"""
None in this case, dataset has been already centered and scaled for us
Usually the code for the scaling should be as follows:

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Train Model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
# np.reshape(-1, 1) to provide the correct input format
regr.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1)) # training the model - simple as that huh

# Predict Results
y_pred = regr.predict(x_test.reshape(-1,1))
y_pred = y_pred[:,0]

# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred) # Mean Squared Error to measure accuracy

# Visualise Results
plt.title("Linear Regression Prediction")
plt.xlabel("Scaled BMI")
plt.ylabel("Result")
plt.plot(x_test, y_pred, color = 'b')
plt.scatter(x_test, y_test, color = 'r') 
plt.show()
# Ideas for improvement
"""
- Use different accuracy metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- Use a different feature from the dataset, e.g. x = dataset["data"][:, 1] or several different features at the same time
- If you use 2 features you can visualise it with 3D plot: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
- Change parameters to reduce the error
"""
