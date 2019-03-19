# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection  import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
import seaborn as sns


MODEL_NUM = 1
OUTPUT_TO_CSV = True
NORMALISE_DATA = True
REDUNDANT_FEATURES = ["zipcode", "lat", "yr_renovated", "price", "id"]


def pre_process_data(train_data, test_data):
    for data in (train_data, test_data):
        conv_dates = [1 if values[:4] == "2014" else 0 for values in data.date]
        data["date"] = conv_dates

        conv_renov_dates = [0 if values == 0 else 1 for values in data.yr_renovated]
        data["yr_renovated"] = conv_renov_dates

        data["lat"] = (data["lat"]+90)*180+data["long"]

# Currently unused
def normalise_data(x_train, x_test, y_train):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.values.reshape(-1 ,1))


    return x_train, x_test, y_train, sc_y


def score_data(train_data, test_data, model):
    """
    So what did we do ? Let’s go step by step.

    We import our dependencies , for linear regression we use sklearn (built in python library) and import linear regression from it.
    We then initialize Linear Regression to a variable reg.
    Now we know that prices are to be predicted , hence we set labels (output) as price columns and we also convert dates to 1’s and 0’s so that it doesn’t influence our data much . We use 0 for houses which are new that is built after 2014.
    So now , we have train data , test data and labels for both let us fit our train and test data into linear regression model.
    After fitting our data to the model we can check the score of our data ie , prediction. in this case the prediction is 73%
    """
    # Defining regression data
    labels = train_data["price"]
    train1 = train_data.drop(REDUNDANT_FEATURES, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.3, random_state=2)

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    
    return score


def predict_data(train_data, test_data, models):
    x_train = train_data.drop(REDUNDANT_FEATURES, axis=1)
    x_test = test_data.drop([foo for foo in REDUNDANT_FEATURES if foo != "price"], axis=1)
    y_train = train_data["price"]

    if NORMALISE_DATA:
        x_train, x_test, y_train, sc_y = normalise_data(x_train, x_test, y_train)
    
    models[MODEL_NUM].fit(x_train, y_train)
    prediction = models[MODEL_NUM].predict(x_test)

    if NORMALISE_DATA:
        prediction = sc_y.inverse_transform(prediction)

    if OUTPUT_TO_CSV:
        with open ("akhilesh_submission.csv", "w") as file:
            file.write("price,id\n")
            for i in range(45129, 51613):
                file.write(str(prediction[i - 45129]) + "," + str(i) + "\n")

    return prediction


def visual_output_data(train_data, test_data, prediction):
    plt.scatter(train_data, train_data.price)
    plt.scatter(test_data, prediction)
    plt.title("? vs Price")
    plt.show()


def main():
    train_data = pd.read_csv(r"C:\Users\Akhilesh\Desktop\ML\train_very_trimmed.csv")
    test_data= pd.read_csv(r"C:\Users\Akhilesh\Desktop\ML\test.csv")
    print(train_data)

    param_grid = {'n_estimators': [300, 500], 'max_features': [10, 12, 14]}
    models = [LinearRegression(),
              RandomForestRegressor(n_estimators=100, max_features='sqrt'),
              ensemble.GradientBoostingRegressor(n_estimators = 50,
                                         max_depth = 6,
                                         min_samples_split = 2,
                                         learning_rate = 0.1,
                                         loss = 'ls'),
              KNeighborsRegressor(n_neighbors=3),
              RidgeCV(alphas=np.arange(70,100,0.05), fit_intercept=True),
              XGBRegressor(n_estimators=100,
                           eta=0.05,
                           learning_rate=0.02,
                           gamma=2,
                           max_depth=6,
                           min_child_weight=1,
                           colsample_bytree=0.8,
                           subsample=0.3,
                           reg_alpha=2,
                           base_score=9.99)]
    
    
    """
    scores=[]
    for x in range(200, 300, 10):
        print(x)
        scores.append(score_data(train_data, test_data, RandomForestRegressor(n_estimators=x, max_features='sqrt')))

    plt.plot(range(200, 300, 10), scores)
    plt.show()
    """
    pre_process_data(train_data, test_data)
    print(score_data(train_data, test_data, models[MODEL_NUM]))  
    prediction = predict_data(train_data, test_data, models)
    visual_output_data(train_data, test_data, prediction)


if __name__ == "__main__":
    main()