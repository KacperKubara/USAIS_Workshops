# Template for the second internal Kaggle competition
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
path_train =  "/home/kacper/Desktop/train.csv"
path_test  =  "/home/kacper/Desktop/test.csv"

# Read Data
x_train = pd.read_csv(path_train)
x_test  = pd.read_csv(path_test)
x_test_copy = x_test.copy()

# Choose which features to use
x_train = x_train.drop(columns = ['id']) # Dropping not very useful data
x_test  = x_test.drop(columns  = ['id'])

# Data Preprocessing
"""
Steps here will differ from what we have done before.
We no longer have have numerical data, only categorical one.
For example in 'cap-shape' feature, we dont have integer values but
letters which describes certain category of the data.

To cope with that, we will use LabelEncoder and OneHotEncoder from Sklearn
More info about this can be found here: 
    https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
"""
