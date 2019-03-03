# 2nd place - Paulina Kulyte
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import xgboost

path_train =r"C:\Users\pauli\Desktop\House AI\train.csv"
path_test  =r"C:\Users\pauli\Desktop\House AI\test.csv"

# Read Data
x_train = pd.read_csv(path_train)
y_train = x_train.price

x_test  = pd.read_csv(path_test)
x_test_copy = x_test

x_train = x_train.drop(columns = ["price", "id", "date", "long", "condition", "zipcode","yr_built", "yr_renovated", "sqft_lot", "sqft_lot15"]) # Dropping not very useful data
x_test = x_test.drop(columns = ["id", "date", "long", "condition", "zipcode","yr_built", "yr_renovated", "sqft_lot", "sqft_lot15"]) 

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1 ,1))

from sklearn.svm import SVR
regr = SVR()

# Train test split
traindf, testdf = train_test_split(x_train, test_size = 0.2, random_state=42)

# Fit & Predict
regr.fit(x_train, y_train)
predictions = regr.predict(x_test).reshape(-1,1)
predictions = sc_y.inverse_transform(predictions) 

"""
# Measure Accuracy
acc = mean_squared_error(y_test, predictions) # Mean Squared Error to measure the accuracy
"""

# Prepare the dataframe for the submission
d = {'price': predictions[:,0], 'id': x_test_copy.id.values}
submission = pd.DataFrame(data = d)
submission.to_csv("submission.csv", index = False)