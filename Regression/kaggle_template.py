# Template for the first internal Kaggle competition
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
path_train =  "/home/kacper/Desktop/train.csv"
path_test  =  "/home/kacper/Desktop/test.csv"

# Read Data
x_train = pd.read_csv(path_train)
y_train = x_train.price
x_test  = pd.read_csv(path_test)
x_test_copy = x_test

# Choose which features to use
x_train = x_train.drop(columns = ["zipcode", "lat", "long", "date", "yr_renovated", "price", "id"]) # Dropping not very useful data
x_test  = x_test.drop(columns  = ["zipcode", "lat", "long", "date", "yr_renovated", "id"])

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1 ,1))

# Train Model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train) # training the model - simple as that huh

# Predict Results
y_pred = regr.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred) #Unscaling the result


# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred) # Mean Squared Error to measure the accuracy


# Prepare the dataframe for the submission
d = {'price': y_pred[:,0], 'id': x_test_copy.id.values}
submission = pd.DataFrame(data = d)
submission.to_csv("submission.csv", index = False)


    
