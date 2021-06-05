import numpy as np
import pandas as pd
import os
import sys
import csv
import re
from numpy.lib import math
import time


default = "normal_reel"

if len(sys.argv) < 2 :
	p = "../data/import_export/aidmap/"
	print("Nom du fichier (sans extention) parmis cette liste : ")
	inputFile = input([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
	if inputFile == "":
		inputFile = default
else:
	inputFile = sys.argv[1]



def timeConverter(time):
	v = re.split('[^0-9]', time)
	return v[2]+"-"+v[1]+"-"+v[0]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]

def valueConverter(value):
	return 0 if value == "FAUX" else 1

data = pd.read_csv(f"../data/import_export/aidmap/{inputFile}.txt", sep='\t')

lastTime = ""
init = True
initTime = data.iloc[0]["Date de départ"] + " " + data.iloc[0]["Heure de départ"]

output = pd.DataFrame({"Temps": [timeConverter(initTime)]})
i = 0

data.replace(np.nan, "", inplace=True)

print("Traitement...")
start = time.time()
for index, row in data.iterrows():
	if(row["Commentaire"] != ""):
		currentTime = row["Date de départ"] + " " + row["Heure de départ"]
		if(initTime == currentTime):
			output[row["Variable"]] = row["Valeur"]
		else:
			if(lastTime != currentTime):
				lastTime = currentTime
				output = output.append(output.tail(1), ignore_index=True)
				i = i + 1
				output.iloc[i][row["Variable"]] = row["Valeur"]
				output.iloc[i]["Temps"] = timeConverter(currentTime)
			else:
				output.iloc[i][row["Variable"]] = row["Valeur"]


output.loc[:,output.columns != 'Temps'] = output.loc[:,output.columns != 'Temps'].applymap(valueConverter)
print(f"Fait en {(time.time() - start):.2f} secondes.")

output.to_csv(f"../data/import_export/{inputFile}.csv", sep=",", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)
print(f"le fichier {inputFile}.csv est généré.")
