import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib


dt=pd.read_csv('sample_data_intw.csv')
dt.dtypes
dt=dt.iloc[:,1:35]
dt=dt.drop('msisdn', axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
dt.iloc[:,1:33] = scaler.fit_transform(dt.iloc[:,1:33])
dt=dt.sample(frac=1)


X_train=dt.iloc[42000:,1:33].as_matrix()
Y_train=dt.iloc[42000:,0].as_matrix()
X_test=dt.iloc[0:42000,1:33].as_matrix()
Y_test=dt.iloc[0:42000,0].as_matrix()

np.random.seed(8)

model = xgb.XGBClassifier()
n_estimators = range(200,1000,200)
max_depth = range(4,10,2)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold, verbose=10)
grid_result = grid_search.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


######################
model = xgb.XGBClassifier(max_depth=6, n_estimators=600, learning_rate=0.05).fit(X_train, Y_train)


compare=pd.DataFrame(dt.iloc[0:42000,0])
compare['model']=model.predict(X_test)
compare.to_csv("compare_xgb_rocauc.csv")

compare1=pd.DataFrame(dt.iloc[0:42000,0])
compare1['model']=model.predict_proba(X_test)[:,1]
compare1.to_csv("compare_xgb_rocauc_prob.csv")

