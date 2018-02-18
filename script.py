# Author
# Tashrif Billah, MS
# Columbia University, New York
# Latest date compiled: 2/12/2018
# Declaration: The code is unique and individually written, under the NDA with Percolata, San Fransisco
# The code is structured and conforming with the ideal of object oriented programming (OOP)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from numpy import genfromtxt
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from xgboost import XGBRegressor as xgbr
from sys import argv
import warnings

warnings.simplefilter("ignore")


# ============================================================================================================

def load_data(file1,file2):

    # Load training data
    # train_data = genfromtxt('train_data.csv', delimiter=',')
    train_data = genfromtxt(file1, delimiter=',')
    r, c = np.shape(train_data)
    Xtrain__full = train_data[1:r, 0:c - 1]
    ytrain__full = train_data[1:r:, c - 1]

    # Load testing data
    # test_data = genfromtxt('test_data.csv', delimiter=',')
    test_data = genfromtxt(file2, delimiter=',')
    r, c = np.shape(test_data)
    Xtest = test_data[1:r, :]

    return Xtrain__full, ytrain__full, Xtest


# ============================================================================================================

def preprocess(Xtrain__full, ytrain__full, Xtest):

    # PCA
    pca = PCA(n_components=10)
    pca.fit(Xtrain__full)
    Xtrain__full = Xtrain__full @ pca.components_.T
    Xtest = Xtest @ pca.components_.T

    # Standardization
    scalar = StandardScaler()
    scalar.fit(Xtrain__full)
    Xtrain__full = scalar.transform(Xtrain__full)
    Xtest = scalar.transform(Xtest)

    return Xtrain__full, ytrain__full, Xtest


# ============================================================================================================

class regressor:

    def __init__(self, lam):
        self.lam= lam
        self.theta= [ ]


    def fit(self, X_train, y_train):

        # Linear regressor abiding by the following equations
        # X*theta= y
        # theta= (X'X+lambda*I)^-1*theta
        # ypred= Xtest*theta

        m, n = np.shape(X_train)
        y_train.reshape(m, 1)
        self.theta = inv(X_train.T @ X_train + self.lam * np.identity(n)) @ X_train.T @ y_train


    def predict(self, Xtest):
        return Xtest @ self.theta


# ============================================================================================================

def show_time(x):

    print("Elapsed Time is %s seconds" % x)
    print(" ")
    print(" ")


def show_performance(model, X, y, set):

    temp= model.predict(X)
    mae = mean_absolute_error(y, temp)
    print(set+" MAE: %f" % mae)



def main( ):

    # random number initialization
    np.random.seed(123456000)

    # preprocess data by PCA and standardization
    Xtrain__full, ytrain__full, Xtest= load_data(argv[1],argv[2])
    # Xtrain__full, ytrain__full, Xtest = load_data("train_data.csv","test_data.csv")
    Xtrain__full, ytrain__full, Xtest = preprocess(Xtrain__full, ytrain__full, Xtest)

    # train-set and validation-set split
    X_train, X_val, y_train, y_val = train_test_split(Xtrain__full, ytrain__full, test_size=0.20, random_state=None)


    # ============================================================================================================
    print(" ")
    print(" ")
    print("Linear regressor classifier")
    start_time = time.time()
    LR= regressor(0.01)
    LR.fit(X_train,y_train)

    show_performance(LR, X_train, y_train, "Train")
    show_performance(LR, X_val, y_val, "Validation")

    show_time(time.time() - start_time)


    # ============================================================================================================
    print("Stochastic gradient descent regressor classifier")
    start_time = time.time()

    SGDR = SGDRegressor(loss='huber', penalty='elasticnet', max_iter=100, eta0=0.01)
    SGDR.fit(X_train, y_train.flatten())

    show_performance(SGDR, X_train, y_train, "Train")
    show_performance(SGDR, X_val, y_val, "Validation")

    show_time(time.time() - start_time)


    # ============================================================================================================
    print("Neural network classifier")
    start_time = time.time()

    def baseline_model(D):

        # Defining the NN based regressor
        model = Sequential()
        model.add(Dense(D, input_dim=D, kernel_initializer = 'glorot_uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(D, input_dim=D, kernel_initializer = 'glorot_uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, kernel_initializer = 'glorot_uniform'))
        model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        return model


    _,D= np.shape(X_train)
    # KR = KerasRegressor(build_fn=baseline_model(D), epochs=30, batch_size=16, verbose=False)
    KR = baseline_model(D)
    KR.fit(X_train, y_train, epochs=100, batch_size=16, verbose=False)

    show_performance(KR, X_train, y_train, "Train")
    show_performance(KR, X_val, y_val, "Validation")

    show_time(time.time() - start_time)



    # ============================================================================================================
    print("Extratrees regressor classifier")
    start_time = time.time()

    ET = ExtraTreesRegressor(n_estimators=200, criterion='mae', min_samples_split=2, min_samples_leaf=1)
    ET.fit(X_train, y_train.flatten())

    show_performance(ET, X_train, y_train, "Train")
    show_performance(ET, X_val, y_val, "Validation")

    show_time(time.time() - start_time)


    # ============================================================================================================
    print("Extreme gradient boosted regressor classifier")
    start_time = time.time()

    n = Xtrain__full.shape[1]
    XGBR = xgbr(n_estimators=400, max_depth=int(np.sqrt(n)))
    XGBR.fit(X_train, y_train.flatten())

    show_performance(XGBR, X_train, y_train, "Train")
    show_performance(XGBR, X_val, y_val, "Validation")

    show_time(time.time() - start_time)


    # ============================================================================================================
    print("Soft voting over best performing ET and XGBR classifiers")
    temp1= ET.predict(X_val)
    temp2= XGBR.predict(X_val)
    temp= np.average([temp1,temp2], axis=0, weights=[7, 10])
    mae = mean_absolute_error(y_val, temp)
    print("Validation MAE: %f" % mae)


    # ============================================================================================================
    print(" ")
    print(" ")
    print("Writing out the results")
    temp1= ET.predict(Xtest)
    temp2= XGBR.predict(Xtest)
    temp= np.average([temp1,temp2], axis=0, weights=[7, 10])
    predictions = temp.astype(int)

    df = pd.read_csv(argv[2])
    # df = pd.read_csv("test_data.csv")
    df['predicted_ground_truth'] = predictions
    df.to_csv(argv[2], index=False)
    # df.to_csv('test_data.csv', index=False)
    print("Task completed")


if __name__== "__main__":
    main( )

