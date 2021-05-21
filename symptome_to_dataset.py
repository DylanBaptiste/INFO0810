import numpy as np
import pandas as pd
import os
import csv
import re
from datetime import datetime

def timeConverter(time):
	v = re.findall(r'\d+', time)
	# return v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]
	return int(datetime.strptime(v[0]+"/"+v[1]+"/"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6], "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e7)
 



data = pd.read_csv('./data/symptome/symptome.csv', sep=',')

# output = data.select_dtypes('int64')
output = (data.loc[:, ((data.dtypes == np.int64) | (data.columns == 'Temps')) ]).copy()

output["Temps"] = output["Temps"].map(timeConverter).values

symp = ["S1_P1S", "S1_P1VP2", "S1_P1VP4", "S1_P2E", "S1_P4E"]
output["Normal"] = output.apply(lambda row: row[symp].eq(0).all().astype(int), axis=1)

label = ["S1_P1S", "S1_P1VP2", "S1_P1VP4", "S1_P2E", "S1_P4E", "Normal"]
output['label'] = output[label].apply(lambda row: ''.join(row.values.astype(str)), axis=1).values
output = output.drop(label, axis=1)

output.dtypes
output["Temps"] = (output[["Temps"]] - output[["Temps"]].shift())
output.loc[0, "Temps"] = 0
output["Temps"] = output["Temps"].astype(int)


# output.insert(output.columns.shape[0] - 1, 'Temps', output.pop('Temps'))

ret = output.copy()
sumTime = 0
lastrow = None
for index, row in ret.iterrows():
	if(index > 0):
		current = row[["A1B2","A1B4","P1S","P1VP2","P1VP4","P2E","P4E", "label"]].values
		if( (lastrow == current).all() ):
			sumTime += row[["Temps"]]
			ret.drop(index, inplace=True)
		else:
			row[["Temps"]] = row[["Temps"]] + sumTime
			sumTime = 0
			lastrow =  row[["A1B2","A1B4","P1S","P1VP2","P1VP4","P2E","P4E", "label"]].values
	else:
		lastrow =  row[["A1B2","A1B4","P1S","P1VP2","P1VP4","P2E","P4E", "label"]].values

ret.to_csv("./data/symptome/dataset.csv", sep=",", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)


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


#convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('%s(t-%d)' % (dataset.columns[j], i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('%s(t)' % (dataset.columns[j])) for j in range(n_vars)]
		else:
			names += [('%s(t+%d)' % (dataset.columns[j], i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load dataset
dataset = read_csv('./data/symptome/dataset.csv', header=0, index_col=None)
dataset = dataset.astype( {"Temps": float} )
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,8] = encoder.fit_transform(values[:,8])
# ensure all data is float
# values[0] = values[0].astype('float32')
# normalize features
values[:,0] = values[:,0] / max(values[:,0])
# frame as supervised learning
timesteps = 10
n_features = 9
reframed = series_to_supervised(values, timesteps, 1)
# drop columns we don't want to predict
print(reframed.shape)
#reframed.drop(reframed.columns[[97,96,95,94,93,92,91,90]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[140, 141, 142, 143, 144, 145, 146, 147]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13, 15]], axis=1, inplace=True)
print(reframed.shape)
pd.DataFrame(reframed).to_csv("./data/symptome/reframed.csv", sep=",", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)