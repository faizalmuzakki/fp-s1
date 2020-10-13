import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def distance(x1, y1, x2, y2):
     dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     
     return dist

siswas = pd.read_csv("data.csv")
siswas = siswas.drop(["nik","lokasi","latsmp","lngsmp","jalur","jarak"], axis=1)

X = siswas.iloc[:, :-1].values
y = siswas.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=63)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))
print(set(y_test) - set(y_pred))