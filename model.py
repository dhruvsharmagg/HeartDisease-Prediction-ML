import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
#from sklearn.preprocessing import scale

#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from graphviz import Source


df = pd.read_csv("C:/Users/User/Desktop/1/framingham.csv")
df.head()
#print(df)

df = df.dropna()

X = df.iloc[:, 0:15]
y =  df.iloc[:, 15]

xgb_model = xgb.XGBClassifier()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [6,8,10, 20, 40],
                    'n_estimators': [10, 20, 50, 100, 200]},scoring='accuracy', verbose=1)

clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.2, random_state = 123 )
xgb_model = xgb.XGBClassifier(max_depth = 6, n_estimators = 20)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("test accuracy: ")
print(accuracy_score(y_test,y_pred))
print("train accuracy: ")
print(accuracy_score(y_train, xgb_model.predict(X_train)))



#Save the model
#model is saved from jupyter notebook
from sklearn.externals import joblib
joblib.dump(xgb_model, 'model.p')

# Load the model that you just saved
xgb = joblib.load('model.p')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.p')
#print(model_columns)
print("Models columns dumped!")
