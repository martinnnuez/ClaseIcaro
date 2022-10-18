import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

# Cargamos los datos y hacemos minimo preprocesamiento
df = pd.read_csv(f"{os.getcwd()}/data/iris.csv")

X = df.drop("variety", axis=1).values.copy()
y = df.variety.copy()

# Entrenamos el modelo predictivo
clf = RandomForestClassifier(max_depth=2)

clf.fit(X, y)

# Guardamos el modelo como un pickle
pickle.dump(clf, open(f"{os.getcwd()}/ml/model.pkl", 'wb'))