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
None in this case, dataset has been already centered and scaled.
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
from sklearn.preprocessing import PolynomialFeatures
# Creates separate variables for each degree of the feature 
poly_reg = PolynomialFeatures(degree = 4)
x_poly   = poly_reg.fit_transform(x_train.reshape(-1, 1))
poly_reg.fit(x_poly, y_train)
x_poly_test   = poly_reg.fit_transform(x_test.reshape(-1, 1))
lin_reg  = LinearRegression()
lin_reg.fit(x_poly_test, y_test)
y_pred   = lin_reg.predict(x_poly_test)

# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred)

# Visualise Results
plt.title("Polynomial Regression Prediction")
plt.xlabel("Scaled BMI")
plt.ylabel("Result")
plt.scatter(x_test, y_test, color = 'r') 
# Need to plot it in steps
x_grid = np.arange(min(x), max(x), (max(x)-min(x))/1000.0)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(x_grid, lin_reg.predict(poly_reg.fit_transform(x_grid)),
         color = 'b')


# Ideas for improvement
"""
- Use different accuracy metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- Use different feature from the dataset, e.g. x = dataset["data"][:, 1] or several different features at the same time
- If you use 2 feature you can visualise it with 3D plot: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
- Change parameters to reduce the loss 
"""