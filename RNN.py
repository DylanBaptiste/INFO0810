import pandas as pd
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import csv

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

import os

import tensorflow as tf

import numpy as np

reframed = read_csv('./data/symptome/reframed.csv', header=0, index_col=None)

# split into train and test sets
values = reframed.values
sep = 100
train = values[:sep, :]
test = values[sep:, :]

timesteps = 10
n_features = 9

# split into input and outputs
n_obs = timesteps * n_features
train_X, train_y = train[:, :n_obs], train[:, -1:]
test_X, test_y = test[:, :n_obs], test[:, -1:]
print(train_X.shape, train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

n_class = np.unique(values[:, -1:]).size
### model ###
model = Sequential()
# hidden layer activation = sigmoid car recurent neural network
model.add(LSTM(4, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.2))

# model.add(Dense(2, activation='relu'))
# model.add(LSTM(128, activation='sigmoid'))
# model.add(Dropout(0.2))

# output layer activation = sigmoid car multilabel probleme
model.add(Dense(n_class, activation='softmax'))

opt = Adam(lr=1e-3, decay=1e-5)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
# loss https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

# fit network
history = model.fit(train_X, train_y, epochs=75, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# plot history
fig, axs = pyplot.subplots(2)
fig.suptitle('Résultats')

axs[0].set_title("loss")
axs[0].plot(history.history['loss'], label='train')
axs[0].plot(history.history['val_loss'], label='test')
axs[0].legend()

axs[1].set_title("accuracy")
axs[1].plot(history.history['accuracy'], label='train')
axs[1].plot(history.history['val_accuracy'], label='test')
axs[1].legend()

pyplot.show()
