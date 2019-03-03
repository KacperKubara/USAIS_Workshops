# Template for the second internal Kaggle competition
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Path to csv files
path_train =  "/home/kacper/Desktop/train.csv"
path_test  =  "/home/kacper/Desktop/test.csv"

# Read Data
x_train = pd.read_csv(path_train)
y = x_train['class']
x_test  = pd.read_csv(path_test)
x_test_copy = x_test.copy()

# Choose which features to use
x_train = x_train.drop(columns = ['id', 'class']) # Dropping class which is our 'y' and 'id'
x_test  = x_test.drop(columns  = ['id'])

############################################################################################## 
# Data Preprocessing
"""
Steps here will differ from what we have done before.
We no longer have have numerical data, only categorical one.
For example in 'cap-shape' feature, we dont have integer values but
letters which describes certain category of the data ().

To cope with that, we will use LabelEncoder and OneHotEncoder from Sklearn
More info about this can be found here: 
    https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
"""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
"""
I will have to do a little tweak here.
I cannot use label and hot encoding in the same way as in my other example:
https://github.com/KacperKubara/USAIS_Workshops/blob/master/Other/label_and_hot_encoding.py

That it's simply because we have multiple categorical columns in this example and the 
LabelEncoder() and OneHotEncoder() doesn't support that.
This helps to solve the problem: 
https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
"""
# Label Encoder
x_train = x_train.apply(LabelEncoder().fit_transform) # Applies fit_transform() to each column separately
x_test  = x_test.apply(LabelEncoder().fit_transform)
x_test_copy = x_test_copy.apply(LabelEncoder().fit_transform)

array = LabelEncoder().fit_transform(y)
y = pd.DataFrame({'class': array})

# Hot Encoding
"""
Instead of Sklearn OneHotEncoder class I decided to use pd.get_dummies
The reason why is that it can also remove one of the dummy variables for each feature
to avoid dummy variable trap.
An overview of the dummy variable trap can be found here:
    
"""
x_train = pd.get_dummies(x_train,columns = x_train.columns, drop_first=True)
x_test  = pd.get_dummies(x_test,columns = x_test.columns, drop_first=True)
y = pd.get_dummies(y,columns = y.columns, drop_first=True)
############################################################################################## 

# Train Model
from sklearn.svm import SVC
svm = SVC(random_state = 42)
svm.fit(x_train, y)

# Predict Results
y_pred = svm.predict(x_test)
y_pred_str = []
for number in y_pred:
    if number == 0:
        y_pred_str.append('e')
    if number == 1:
        y_pred_str.append('p')
        
# Save the results in .csv file in the correct format
submission = pd.DataFrame({'id': x_test_copy['id'], 'class': y_pred_str})
submission.to_csv('submission.csv', index = False)