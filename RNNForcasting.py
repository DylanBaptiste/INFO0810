import numpy as np
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
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Flatten, Input, GRU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import elu
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
import random
import os

import tensorflow as tf
tf.config.run_functions_eagerly(False)

import numpy as np
import csv
import re
from datetime import datetime


def timeConverter(time):
	v = re.findall(r'\d+', time)
	# return v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]
	return int(datetime.strptime(v[0]+"/"+v[1]+"/"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6], "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e7)
 
make_reframed = True

if(make_reframed):
	reframed = read_csv('./data/import_export/import_export_2.csv', header=0, index_col=None)

	reframed["Temps"] = reframed["Temps"].map(timeConverter).values
	reframed["Temps"] = (reframed[["Temps"]] - reframed[["Temps"]].shift())
	reframed.loc[0, "Temps"] = 0
	reframed["Temps"] = reframed["Temps"].astype(int)

	ret = reframed.copy()
	sumTime = 0
	lastrow = None
	testColumn = ["fermee","ouvert","x_conv","x_ext","x_inter","y_conv","y_sta","z_haut","z_bas","conv_debut","conv_fin","gliss1","gliss2","but_int","but_ext","stop_out","stop_in","EmStop","gliss_int","gliss_ext","OUVRIR","FERMER","XCONV","XGLISS","YCONV","YSTA","ZDESC","ZMONT","ZFREIN","CONV","BLOQINT","BLOQEXT","XSTOPINT"]
	for index, row in ret.iterrows():
		if(index > 0):
			current = row[testColumn].values
			if( (lastrow == current).all() ):
				sumTime += row[["Temps"]]
				ret.drop(index, inplace=True)
			else:
				row[["Temps"]] = row[["Temps"]] + sumTime
				sumTime = 0
				lastrow =  row[testColumn].values
		else:
			lastrow =  row[testColumn].values

	
	reframed.to_csv("./data/import_export/reframed_2.csv", sep=",", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)
else:
	reframed = read_csv("./data/import_export/reframed_2.csv", header=0, index_col=None)


def split_series(series, n_past, n_future):
	X, y = list(), list()
	for window_start in range(len(series)):
		past_end = window_start + n_past
		future_end = past_end + n_future
		if future_end > len(series): break
		# slicing the past and future parts of the window
		past, future = series[window_start:past_end, :], series[past_end:future_end, :]
		X.append(past)
		y.append(future)
	return np.array(X), np.array(y)

n_past = 10
n_future = 1 
n_features = reframed.shape[1]


# 75% train 25% test
# train_df, test_df = reframed[1:int(reframed.shape[0] * 0.75)], reframed[int(reframed.shape[0] * 0.75):] 
split = 10
print(f"split in {len(reframed)} sample to {100 - (100/split)}% train {(100/split)}% test")
# reframed = reframed.sample(frac=1).reset_index(drop=True)

reframed[["Temps"]] = reframed[["Temps"]]  / max(reframed["Temps"])
train_df, test_df = reframed.iloc[[i for i in range(len(reframed)) if i % split != 0]], reframed.iloc[[i for i in range(len(reframed)) if i % split == 0]]

print(train_df.shape, test_df.shape)

test = test_df
train = train_df

component_train = train.iloc[:, 1:]
time_train = train.iloc[:, :1]
X_time_train, y_time_train = split_series(time_train.values, n_past, n_future)
X_component_train, y_component_train = split_series(component_train.values, n_past, n_future)

X_test, y_test = split_series(test.values, n_past, n_future)
# component_test = test.iloc[:, 1:]
# time_test = test.iloc[:, :1]
# X_time_test, y_time_test = split_series(time_test.values, n_past, n_future)
# X_component_test, y_component_test = split_series(component_test.values, n_past, n_future)

# reshape
X_time_train = X_time_train.reshape((X_time_train.shape[0], X_time_train.shape[1], 1))
X_component_train = X_component_train.reshape((X_component_train.shape[0], X_component_train.shape[1], n_features - 1))
y_time_train = y_time_train.reshape((y_time_train.shape[0], y_time_train.shape[1], 1))
y_component_train = y_component_train.reshape((y_component_train.shape[0], y_component_train.shape[1], n_features - 1))


# X_time_test = X_time_test.reshape((X_time_test.shape[0], X_time_test.shape[1], 1))
# X_component_test = X_component_test.reshape((X_component_test.shape[0], X_component_test.shape[1], n_features - 1))
# y_time_test = y_time_test.reshape((y_time_test.shape[0], y_time_test.shape[1], 1))
# y_component_test = y_component_test.reshape((y_component_test.shape[0], y_component_test.shape[1], n_features - 1))


# scale
# scalers = {}

# train = train_df
# for i in train_df.columns:
#     scaler = MinMaxScaler(feature_range=(-1,1))
#     s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
#     s_s=np.reshape(s_s,len(s_s))
#     scalers['scaler_'+ i] = scaler
#     train[i]=s_s

# test = test_df
# for i in train_df.columns:
#     scaler = scalers['scaler_'+i]
#     s_s = scaler.transform(test[i].values.reshape(-1,1))
#     s_s=np.reshape(s_s,len(s_s))
#     scalers['scaler_'+i] = scaler
#     test[i]=s_s

# X_train, y_train = split_series(train.values, n_past, n_future)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

# y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))


# X_test, y_test = split_series(test.values, n_past, n_future)
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))




def customLossFunc(ypred, ytrue):
	# K.square((ypred[:1] - ytrue[:1])[0]) + 
	return (K.sum(K.square(ypred[1:] - ytrue[1:]), axis=1) )

# print(customLossFunc(K.constant([0.8, 1,1,1,1]), K.constant([0.8, 1,.6,-.2,-.4])))
# a = K.constant([0.7, 1,1,1,1])
# b = K.constant([0.7, -1,-1,-1,-1])

# (K.square((a[:1] - b[:1])[0]) + K.mean(a[1:] - b[1:]) )

def learning_rate_scheduler(epoch, lr): 
	#Say you want to decay linearly by 5 after every 10 epochs the lr
	#(epoch + 1) since it starts from epoch 0
	if (epoch + 1) % 10 == 0:
		lr = float(lr / 100)
	return lr


# callbacks = LearningRateScheduler(learning_rate_scheduler, verbose=1)
def component_acc(y_true, y_pred):
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	return K.sum(K.equal(K.round(y_true), K.round(y_pred)))

def time_acc(y_true, y_pred):
    return K.square((y_pred[:1] - y_true[:1])[0])


# model = Sequential()
# # model.add(Input())
# model.add(GRU(100, return_sequences=True, input_shape=(n_past, n_features)))
# # model.add(Dropout(0.2))
# model.add(GRU(100, return_sequences=True))
# model.add(Dense(n_features, activation="sigmoid"))
# # model.add(Dense(n_features, activation="tanh"))
# model.compile(loss="mse", metrics=[], optimizer=SGD(lr=0.001)) # Adam(lr=1e-3, decay=1e-6) SGD(lr=0.01)
# # "cosine_similarity"
# model.summary()


def fit(model, epochs, batch_size, callbacks=None):

	history = model.fit([X_time_train, X_component_train], [y_time_train, y_component_train], epochs=epochs, validation_data=([X_test, y_test]), batch_size=batch_size, verbose=1, callbacks=callbacks)

	fig, axs = pyplot.subplots(2)
	fig.suptitle('Résultats')
	print(history.history)
	axs[0].set_title("loss")
	axs[0].plot(history.history['loss'], label='train')
	axs[0].plot(history.history['val_loss'], label='test')
	axs[0].legend()

	# axs[1].set_title("component_acc")
	# axs[1].plot(history.history['component_acc'], label='train')
	# axs[1].plot(history.history['val_component_acc'], label='test')
	# axs[1].legend()

	pyplot.show()

def build_model():
	# Multi input mixed data
	time_input = Input(shape=(n_past, 1))
	component_input = Input(shape=(n_past, n_features - 1))
	# time_branch
	time_branch = Dense(10, activation="sigmoid")(time_input)
	time_branch = LSTM(10, return_sequences=True)(time_branch)
	time_branch = LSTM(10, return_sequences=False)(time_branch)
	time_branch = Dense(1, activation="sigmoid")(time_branch)
	# time_branch = Model(inputs=time_input, outputs=time_branch)
	# component_branch
	component_branch = Dense(10, activation="tanh")(component_input)
	component_branch = LSTM(10, return_sequences=True)(component_branch)
	component_branch = LSTM(10, return_sequences=False)(component_branch)
	component_branch = Dense(n_features - 1, activation="tanh")(component_branch)
	# component_branch = Model(inputs=component_input, outputs=component_branch)
	# concatenation
	combined = Concatenate(axis=1)([time_branch, component_branch])
	# combined = Flatten()(combined)
	# FC
	# z = Dense(2, activation="linear")(combined)
	z = Dense(n_features, activation="sigmoid")(combined)
	
	model = Model(inputs=[time_input, component_input], outputs=z)
	return model

model = build_model()
model.compile(loss="binary_crossentropy", metrics = [], optimizer=Adam(lr=1e-3, decay=1e-6)) # Adam(lr=1e-3, decay=1e-6) SGD(lr=0.01)

model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, expand_nested=True, dpi=96*2)



fit(model, 200, 50)




def make_prediction(X):
	return model.predict(np.array( [X,]))[0]

model.evaluate([X_time_test, X_component_test], [y_time_test, y_component_test])



choix = 1
expected = concatenate([y_time_test[choix], y_component_test[choix]], axis=1)

pred = model.predict([X_time_test, X_component_test])
pred = model.predict_on_batch([X_time_test, X_component_test])
pred[choix]