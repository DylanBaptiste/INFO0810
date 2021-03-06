import numpy as np
import pandas as pd
import os
import sys
import csv
import re
from numpy.lib import math
import time

if len(sys.argv) < 2 :
	p = "../data/import_export/aidmap/"
	print("Nom du fichier (sans extention) parmis cette liste : ")
	inputFile = input([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
	if inputFile == "":
		inputFile = "Normal"
else:
	inputFile = sys.argv[1]

correspondance = pd.DataFrame({
	"Fermer la pince" : "FERMER",
	"Ouvrir la pince" : "OUVRIR",
	"Descendre la pince (axe Z)" : "ZDESC",
	"Monter la pince (axe Z)" : "ZMONT",
	"Déverrouiller le frein vertical (axe Z)" : "ZFREIN",
	"Mvt vers la station d'import/export (axe Y)" : "YSTA",
	# Mvt vers la station d'import/export (axe Y)
	"Mvt vers le convoyeur central (axe Y)" : "YCONV",
	"Mvt vers les glissières  (axe X)" : "XGLISS",
	"Mvt vers le convoyeur d'approvisionnement (axe X)" : "XCONV",
	# Mvt vers le convoyeur d'approvisionnement (axe X) 
	"Sortir le stoppeur intermédiaire (axe X )" : "XSTOPINT",
	"Sortir la butée de la glissière intermédiaire" : "BLOQINT",
	"Sortir la butée de la glissière externe" : "BLOQEXT",
	"Rotation du convoyeur d'approvisionnement" : "CONV",
	"Pince en position station import/export sur l'axe Y." : "y_sta",
	"Pince en position convoyeur central sur l'axe Y." : "y_conv",
	"Pince en position basse sur l'axe Z" : "z_bas",
	"Pince en position haute sur l'axe Z" : "z_haut",
	"Pince fermée." : "fermee",
	"Pince ouverte." : "ouvert",
	"Présence d'un 6packs sur la butée de la glissière externe." : "gliss_ext",
	"butée de la glisssière externe sortie" : "but_ext",
	# Butée de la glissière externe sortie.
	"Pince en position glissière externe sur l'axe X." : "x_ext",
	"Pince en position convoyeur d'approvisionnement sur l'axe X." : "x_conv",
	# Pince en position convoyeur d'approvisionnement sur l'axe X. 
	"Pince en position glissière intermédiaire sur l'axe X." : "x_inter",
	"Stoppeur intermédiaire rentrée" : "stop_in",
	# Stoppeur intermédiaire rentrée.
	"Stoppeur intermédiaire sortie" : "stop_out",
	# Stoppeur intermédiaire sortie.
	"Glissière intermédiaire pleine" : "gliss1",
	# Glissière intermédiaire pleine
	"Glissière externe pleine" : "gliss2",
	# # Glissière externe pleine
	"présence d'un 6packs à la fin du convyeur d'approvisionnement" : "conv_fin",
	# Présence d'un 6packs au début du convoyeur d'approvisionnement.
	"présence d'un 6packs au début du convoyeur d'approvisionnement" : "conv_debut",
	# Présence d'un 6packs au début du convoyeur d'approvisionnement.
	"Présence d'un 6packs sur la butée de la glissière intermédiaire" : "gliss_int",
	# Présence d'un 6packs sur la butée de la glissière intermédiaire 
	"butée de la glissière intermédiaire sortie" : "but_int",
	# Butée de la glissière intermédiaire sortie.

	"Arret d'urgence" : "EmStop"
}, index=[0])


def timeConverter(time):
	v = re.split('[^0-9]', time)
	# pas l'heure ???
	# print(v[0]+"-"+v[1]+"-"+v[2]+" "+"0"+":"+v[3]+":"+v[4]+"."+v[5])
	# yyyy-mm-dd hh:mm:ss[.fffffffff]
	# return v[2]+"-"+v[1]+"-"+v[0]+" "+"00"+":"+v[3]+":"+v[4]+"."+v[5]
	return v[2]+"-"+v[1]+"-"+v[0]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]

def valueConverter(value):
	return 0 if value == "FAUX" else 1

data = pd.read_csv(f"../data/import_export/aidmap/{inputFile}.txt", sep='\t')

lastTime = ""
init = True
initTime = data.iloc[0]["Date de départ"] + " " + data.iloc[0]["Heure de départ"]

output = pd.DataFrame({"Temps": [timeConverter(initTime)]})
i = 0

data['Commentaire'].replace(np.nan, "", inplace=True)
print("Traitement...")
start = time.time()
for index, row in data.iterrows():
	if(row["Commentaire"] != ""):
		currentTime = row["Date de départ"] + " " + row["Heure de départ"]
		if(initTime == currentTime):
			output[correspondance[row["Commentaire"]]] = row["Valeur"]
		else:
			if(lastTime != currentTime):
				lastTime = currentTime
				output = output.append(output.tail(1), ignore_index=True)
				i = i + 1
				output.iloc[i][correspondance[row["Commentaire"]]] = row["Valeur"]
				output.iloc[i]["Temps"] = timeConverter(currentTime)
			else:
				output.iloc[i][correspondance[row["Commentaire"]]] = row["Valeur"]


output.loc[:,output.columns != 'Temps'] = output.loc[:,output.columns != 'Temps'].applymap(valueConverter)
print(f"Fait en {(time.time() - start):.2f} secondes.")

output.to_csv(f"../data/import_export/{inputFile}.csv", sep=",", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)
print(f"le fichier {inputFile}.csv est généré.")
