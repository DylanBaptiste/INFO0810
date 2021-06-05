import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # log que les erreurs de tensorflow

# Nummpy
import numpy as np
from numpy.core.shape_base import block
from numpy import concatenate
np.set_printoptions(suppress=True)

# Pandas
import pandas as pd
from pandas import read_csv, DataFrame, concat

# sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

print("Chargement de Tensorflow...")
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Reshape, Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Flatten, Input, GRU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import elu
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.backend import reshape, epsilon, _to_tensor
tf.config.run_functions_eagerly(False)
print("")

# Utils
import random
import csv
import re
from datetime import datetime
from matplotlib import pyplot



def timeConverter(time):
	v = re.findall(r'\d+', time)
	# return v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]
	return int(datetime.strptime(v[0]+"/"+v[1]+"/"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6], "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e7)

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

class PlotLosses(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []

		self.acc1 = []
		self.val_acc1 = []

		self.acc2 = []
		self.val_acc2 = []

		self.fig, self.axs = pyplot.subplots(ncols=1, nrows=3, sharex=True, sharey=False)
		self.axs[0].legend()
		self.axs[1].legend()
		self.axs[2].legend()
		# pyplot.show(block=False)
		# pyplot.ion()
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):
		

		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.acc1.append(logs.get('composant_acc1'))
		self.val_acc1.append(logs.get('val_composant_acc1'))
		self.acc2.append(logs.get('composant_acc2'))
		self.val_acc2.append(logs.get('val_composant_acc2'))
		self.i += 1
		pyplot.cla()
		# pyplot.plot(self.x, self.losses, label="loss")
		# pyplot.plot(self.x, self.val_losses, label="val_loss")
		self.axs[0].plot(self.losses,  label="loss")
		self.axs[0].plot(self.val_losses, label="val_loss")
		self.axs[1].plot(self.acc1,  label="acc1")
		self.axs[1].plot(self.val_acc1, label="val_acc1")
		self.axs[2].plot(self.acc2,  label="acc2")
		self.axs[2].plot(self.val_acc2, label="val_acc2")
		
		if(epoch == 2):
			pyplot.ion()
			pyplot.show()
			pyplot.legend()

		
		pyplot.draw()
		pyplot.pause(0.001)

def composant_acc1(y_true, y_pred):
	correct_prediction = tf.equal(tf.round(y_pred[:, :, 1:]), y_true[:, :, 1:])
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def composant_acc2(y_true, y_pred):
	correct_prediction = tf.equal(tf.round(y_pred[:, :, 1:]), y_true[:, :, 1:])
	all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 2)
	return tf.reduce_mean(all_labels_true)

# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
def composant_loss(y_true, y_pred):
	return tf.keras.losses.binary_crossentropy(y_true[:, :, 1:], y_pred[:, :, 1:])

def time_loss(y_true, y_pred):
	return tf.reduce_mean(K.square(y_pred[:, :, :1] - y_true[:, :, :1]))



def custom_loss(y_true, y_pred, k1=1, k2=1):
	return time_loss(y_true, y_pred) * k1 + composant_loss(y_true, y_pred) * k2

class CustomModel(Model):
	def train_step(self, data):
		x, y = data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			# Compute our own loss
			loss = custom_loss(y, y_pred, k1, k2)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Compute our own metrics
		
		# % de bonne prediction par composants (on arrondit la prediction, on compare à y réel, on cast les bool en 0.0 et 1.0, puis on peut effectuer des moyennes )
		# le premier reduce_mean reduit le tableau (batch_size, n_future, n_feature-1) par l'axe batch_size
		# le deuxieme reduce_mean reduit le tableau (n_future, n_feature-1) par l'axe n_future
		# on obtient donc un tableau 1D de shape (n_feature-1) à la fin
		# On fait donc la moyennes des moyennes
		t = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.round(y_pred[:, :, 1:]) == y[:, :, 1:], dtype=tf.float16), axis=0), axis=0)
		# De cette maniere on peut tracker pour chaque composant sont accuracy au fil de l'entrainement:
		metrics = {n:v for (n, v) in zip(columnsNames[1:], tf.unstack(t)) }

		# la loss sera de la shape (batch_size, n_future)
		# on peut faire plusieurs choix ici :
		# - La moyenne par l'axe batch_size -> produit un vectueur de shape (n_future,) puis faire sa moyenne pour reduire le tout à 1 seule valeur
		# - La moyenne par l'axe n_future -> produit un vectueur de shape (batch_size,) puis faire sa moyenne pour reduire le tout à 1 seule valeur
		# - faire la moyenne sur toute la matrice directement
		# il est peut etre possible de garder la loss sous forme de vecteur (n_future) pour observer si la moyenne des etapes T+n sont similaires entre elles ou pas
		metrics["loss"] = tf.reduce_mean(loss) # moyenne sur toute la matrice
		
		
		metrics["time_loss"] = time_loss(y, y_pred) # plus ou moins le meme principe que la loss sauf que je reduce_mean directement dans le fonction, l'ecart au carré juste avant est de la shape (batch_size, n_future, 1)
		metrics["composant_loss"] = tf.reduce_mean(composant_loss(y, y_pred)) # composant_loss c'est la binarycrossentropy, la shape est (batch_size, n_future), meme remaque que pour la loss au dessus
		metrics["composant_acc1"] = composant_acc1(y, y_pred) # moyenne des prediction des coposants predits correctement, meme remarque que pour la loss : possibilité de garder la valeur à traver les n_future apres avoir moyénné sur l'axe batch_size
		metrics["composant_acc2"] = composant_acc2(y, y_pred) # moyenne des predictions correctes par batch_size et par n_future, meme remarque que pour la loss : possibilité de garder la valeur à traver les n_future apres avoir moyénné sur l'axe batch_size
		return metrics
	
	def test_step(self, data):
		# Unpack the data
		x, y = data

		# Compute predictions
		y_pred = self(x, training=False)
		
		# Compute our own metrics
		# idem à train_step
		loss = custom_loss(y, y_pred, k1, k2)
		t = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.round(y_pred[:, :, 1:]) == y[:, :, 1:], dtype=tf.float16), axis=0), axis=0)
		metrics = {n:v for (n, v) in zip(columnsNames[1:], tf.unstack(t)) }
		metrics["loss"] = tf.reduce_mean(loss)
		metrics["time_loss"] = time_loss(y, y_pred)
		metrics["composant_loss"] = tf.reduce_mean(composant_loss(y, y_pred))
		metrics["composant_acc1"] = composant_acc1(y, y_pred)
		metrics["composant_acc2"] = composant_acc2(y, y_pred)
		return metrics



# Variables
n_past = 20 # nombre d'etat passé que le Model doit connaitre
n_future = 1 # nombre d'etats futur qu'il doit d'eterminer
n_features = 34
dataset = "normal_reel" #"Normal"

# variable globale pour la loss
k1=0
k2=1

# On peut pas utiliser directement les fichier csv, il faut transformer les données en un dataset de la forme:
# X1, Y1
# X2, Y2
# ...
# où Xi sera de la forme d'un vecteur de taille (n_past * n_features)
# et Yi de la forme d'un vecteur de taille (n_future * n_features)
# de cette maniere on peut le dataset dans un fichier
# Pour notre model il ne faudra pas donner les vecteurs mais des tableau 2D de la forme :
# X : (n_past, n_features)
# Y : (n_future, n_features)
# lors de l'entrainement le batch_size sera ajouté en tant que premier axe
# on aura donc X : (batch_size, n_past, n_features), Y : (batch_size, n_future, n_features)
# des tableaux en 3D donc
# Pour la preparation des données on aura aussi des tableau 3D :
# X : (taille du dataset, n_past, n_features)
# Y : (taille du dataset, n_future, n_features)

# reframed aura la forme :
# données passées, données futures
# n_past * n_feature, n_futur * n_feature
#        X          ,         Y
print("Mise en forme X, Y du dataset...")
reframed = read_csv(f"../data/import_export/{dataset}.csv", header=0, index_col=None) # chargement du fichier csv généré par aidmap_to_csv
columnsNames = reframed.columns # sauvegarde des noms des composants (et le temps en 1er position)
reframed["Temps"] = reframed["Temps"].map(timeConverter).values # conversion du temps en unité
reframed["Temps"] = (reframed[["Temps"]] - reframed[["Temps"]].shift()) # temps relatif, dans chaque ligne le temps donne à present le nombre de microseconde s'etant ecoulé depuis la derniere ligne 
reframed.loc[0, "Temps"] = 0 # la premier recois 0 (juste pour ne pas avoir nan)
reframed["Temps"] = reframed["Temps"].astype(int)
# on scale la colonne du temps entre 0 et 1, etape obligatoire pour ne pas avoir une valeur de l'erreur qui extreme pendant l'entrainement (et pour normaliser les donnés et pour matcher avec l'activation en sigmoid)
max_temps = max(reframed['Temps'])
print(f"Max temps = {max_temps}") # le temps est à sauvegarder pour pouvoir faire le calcule inverse du scaling
reframed[["Temps"]] = reframed[["Temps"]] / max_temps # le calcule inverse sur une valeur sera donc fait avec la multiplication de max_temps

# on forme les données du fichier csv en format x, y en fonction de n_past et  n_future
x, y = split_series(reframed.values, n_past, n_future)
reframed = pd.DataFrame(np.concatenate((x, y), axis=1).reshape(len(x), (n_past + n_future) * n_features))
# si on veux save dans un fichier pour ne pas refaire les calcul precedent :
# np.savetxt(f"../data/import_export/{dataset}.csv", reframed, delimiter=",", fmt='%f')
# pour le charger simplement lire le fichier :
# reframed = f"../data/import_export/{dataset}.csv"
# si save ne pas oublier de sauvegarder aussi la valeur de max_temps pour pouvoir refaire le calucle inverse sur les temps un jour

print("")

# Separation en TRAIN/TEST
# D'autre technique existent notament la "validaion croisé" (k-fold cross validation) qui pourait etre interessant de tester dans notre cas
# pour l'instant je serpare juste en selectionant 1 ligne sur 4 donc :
# 75% train 25% test
# train_df, test_df = reframed[1:int(reframed.shape[0] * 0.75)], reframed[int(reframed.shape[0] * 0.75):] # mauvaise maniere de faire !
split = 4
print(f"Separation de {len(reframed)} données en {100 - (100/split)}% entrainement {(100/split)}% test.")
random.seed(10)
reframed = reframed.sample(frac=1).reset_index(drop=True)
# on selctionne vraiement 1 ligne sur 4 pour les repartirs en train/test
train, test = reframed.iloc[[i for i in range(len(reframed)) if i % split != 0]], reframed.iloc[[i for i in range(len(reframed)) if i % split == 0]]
print(f"Entrainement : {train.shape}")
print(f"Test :{test.shape}")
print("")

print("Separtion X, Y des dataset d'entrainement et de test:")
# X_test, y_test = split_series(test.values, n_past, n_future)
X_test, y_test = np.hsplit(test, [n_features * n_past])
X_test = np.asarray(X_test).reshape(X_test.shape[0], n_past, n_features)
y_test = np.asarray(y_test).reshape(y_test.shape[0], n_future, n_features)

# X_train, y_train = split_series(train.values, n_past, n_future)
X_train, y_train = np.hsplit(train, [n_features * n_past])
X_train = np.asarray(X_train).reshape(X_train.shape[0], n_past, n_features)
y_train = np.asarray(y_train).reshape(y_train.shape[0], n_future, n_features)
print(f"X_train\t: {X_train.shape}")
print(f"y_train\t: {y_train.shape}")
print(f"X_test\t: {X_test.shape}")
print(f"y_test\t: {y_test.shape}")
print("")


# Architecture du Model
print("Creation du Model...")
inputs = Input(shape=(n_past, n_features))
layer = LSTM(50, return_sequences=True)(inputs)
layer = Dropout(0.2)(layer)
layer = LSTM(50, return_sequences=True)(layer)
layer = Dropout(0.2)(layer)
layer = LSTM(50, return_sequences=True)(layer)
layer = Dropout(0.2)(layer)
layer = LSTM(50, return_sequences=True)(layer)
layer = Dropout(0.2)(layer)
layer = LSTM(50, return_sequences=False)(layer)
layer = Dense(n_features * n_future, activation="sigmoid")(layer)
output = Reshape((n_future, n_features))(layer)
model = CustomModel(inputs, output)


# Compilation du Model
model.compile(run_eagerly=False, optimizer="RMSprop") # Adam(lr=1e-3, decay=1e-6) SGD(lr=0.01) optimizer="RMSprop"
print("")

# Plot du Model
model.summary()
plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=False, expand_nested=True, dpi=96*2)

# Entrainement
batch_size = int(X_train.shape[0]/50) 	# Nombre de batch à donner à chaaque epoque
epochs = 300							# Nombre d'epoques
callbacks=[PlotLosses()]				# Plot en live pendant l'entrainement 

history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)

# Sauvegarde des reslutats durant l'entrainements (accuracy de chaque composants, acc1, acc2, losses)
pd.DataFrame.from_dict(history.history).to_csv("../resultat/import_export/history.csv", index=False)


model.evaluate(X_test, y_test)

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################



plot_history(history.history)
pyplot.ion()
pyplot.ioff()
def simplePlot(name):
	pyplot.plot(history.history[name], label = "train "+name)
	pyplot.plot(history.history["val_"+name], label = "test "+name)
	pyplot.show()
simplePlot("composant_acc2")
simplePlot("composant_acc1")



# model.save('model_3nextstep')
# model = tf.keras.models.load_model('model_3laysertanhLSTM100n_RMSprop_1000it_100btch', custom_objects={ 'composant_loss': composant_loss, "composant_acc1":composant_acc1, "composant_acc2":composant_acc2})

# def make_prediction(X):
# 	return model.predict(np.array( [X,]))[0]

model.predict(np.array(X_test))

y_pred = model.predict(X_test[0].reshape(1, 50, 34))
a = y_test[0].reshape(1, 1, 34)
b = y_pred

np.round(a[:,0, 1:])
np.round(b[:,0, 1:])


composant_acc2(a, b)
composant_acc1(a, b)
composant_loss(a, b)


def plot_history(history):
	fig, axs = pyplot.subplots(int(len([m.name for m in model.metrics])), sharex=True, sharey=False)
	fig.suptitle('Résultats')
	
	if int(len([m.name for m in model.metrics])) == 1 :
		for x in [m.name for m in model.metrics]:
			axs.set_title(x)
			axs.plot(history.history[x], label='train')
			axs.plot(history.history[f"val_{x}"], label='test')
			axs.legend()
	else:
		i = 0
		for x in [m.name for m in model.metrics]:
			axs[i].set_title(x)
			axs[i].plot(history.history[x], label='train')
			axs[i].plot(history.history[f"val_{x}"], label='test')
			axs[i].legend()
			i = i + 1

	pyplot.show()




# max_temps = 4240000
# def predict_time(choix):
# 	expected = y_test[choix]
# 	pred = make_prediction(X_test[choix])
# 	return expected.reshape(34,)[:1][0] * max_temps, np.round(pred[:1][0] * max_temps)

# def predict_composant(choix):
# 	expected = y_test[choix]
# 	pred = make_prediction(X_test[choix])
# 	return expected.reshape(34,)[1:], np.round(pred[1:])





# print(predict_time(93))
# print(predict_composant(93))

# expect, pred = predict_composant(93)

# composant_acc1(expect, pred)

# K.mean(K.equal(expect, K.round(pred)),  axis=-1)

# y_pred = model.predict(np.array(X_test))
# y_pred.shape
# y_test.shape



# def live_diagnostic_prediction(choix):
# 	pred, expect = model.predict(np.array(X_test))[choix, 1:],  y_test[choix, 1:]
# 	print(f"match\tpred\tterrain\tconfiance\tvaleur")
# 	for i in range(expect.shape[0]):
# 		print(f"{int(np.round(pred[i], 0)) == int(expect[i])}\t{int(np.round(pred[i], 0))}\t{int(expect[i])}\t{np.round((1 - np.abs((pred[i] - np.round(pred[i]))))*100, 2) }%\t{pred[i]}")

# live_diagnostic_prediction(0)
# live_diagnostic_prediction(16)

# x = X_test[0].reshape(1, n_past, n_features)
# x.shape
# y = model.predict(x)
# x.reshape(50, 34)
# y.shape

# correct_prediction = tf.equal(tf.round(y_pred[:, 1:]), y_test[:, 1:])
# all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
# accuracy = tf.reduce_mean(all_labels_true)
# y_pred[:, 1:].shape
# y_test[:, 1:].shape
# #composant_acc2(y_test, y_pred).numpy()


# y_pred = model.predict(np.array(X_train))
# composant_acc2(y_train, y_pred).numpy()
# composant_acc1(y_train, y_pred).numpy()

# y_pred = model.predict(np.array(X_test))
# composant_acc2(y_test, y_pred).numpy()
# composant_acc1(y_test, y_pred).numpy()


def reverseTime(value):
	# dt = datetime.fromtimestamp( (value * max_temps) // 1000000000)
	# s = dt.strftime('%Y-%m-%d %H:%M:%S')
	# s += '.' + str(int((value * max_temps) % 1000000000)).zfill(9)
	return np.round( ((value * max_temps) / 10e6), 3) # response en secondes

def get_confiance(value):
	predicetion = np.round(value)
	indice = 1 - np.abs(value - predicetion)
	rescale = (indice / 0.5) - 1 # 0.5 => 0; 0.75 => 0.5; 1 => 1
	return np.round(rescale * 100, 2) # %


def diagostic():
	# live = read_csv("./data/import_export/import_export.csv", header=0, index_col=None)
	diag = pd.DataFrame(np.zeros((5, n_features), float), index=["match", "precedent", "terrain", "prediction", "confiance"], columns=columnsNames)

	lastTemps = timeConverter(read_csv(f"../data/import_export/{dataset}.csv", header=0, index_col=None)["Temps"][0])
	live = read_csv(f"../data/import_export/{dataset}.csv", header=0, index_col=None)
	window = np.empty((1, 0, n_features), float)

	for index, row in live.iterrows():

		currentTemps = timeConverter(row["Temps"])
		row["Temps"] = (currentTemps - lastTemps) / max_temps
		lastTemps = currentTemps
		window = np.append(window, row.values.reshape(1, 1, 34), 1)
		
		if(window.shape[1] > n_past):
			window = np.delete(window, 0, axis=1)
		
		if( (window.shape[1] == n_past) and ((index + 1) < live.shape[0])):
			print(index + 1)
			pred = model.predict(np.array(window).reshape(1, n_past, n_features).astype('float32'))
			expect = live.iloc[index + 1].copy()
			expect.loc["Temps"] = (timeConverter(expect["Temps"]) - lastTemps) / max_temps
			
			composant = expect.astype('float32')[1:]
			diag.loc["precedent"] = np.concatenate([live.iloc[index].copy()[:1], live.iloc[index].copy()[1:]]).reshape(34,)
			diag.loc["prediction"] = np.concatenate((np.array(reverseTime(pred[:, :, :1].item((0,0,0)))).reshape(1,), np.round(pred[:, :, 1:]).reshape(33,))).reshape(34,)
			diag.loc["terrain"] = np.concatenate( [ [reverseTime(expect.loc["Temps"])], np.round(composant).values.astype(int) ] )
			diag.loc["match"] = np.array(diag.loc["prediction"] == diag.loc["terrain"])
			diag.loc["confiance"] = np.round(np.concatenate( ([ np.array([np.nan]).reshape(1, 1, 1), get_confiance(pred[:,:,1:])]), axis=2), 3).reshape(34, )
			# print(diag)
			# input("Continuer...")

			if(not diag.loc["match"][1:34].values.all()):
				print(diag.loc[:, (diag.loc["match",]==False)].iloc[:, :].loc[["precedent", "terrain", "prediction", "confiance"]])
				input("\nContinuer...")
		

diagostic()

