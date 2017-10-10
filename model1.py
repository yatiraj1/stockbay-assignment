import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dt=pd.read_csv('sample_data_intw.csv')
dt.dtypes
dt=dt.iloc[:,1:35]
dt=dt.drop('msisdn', axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
dt.iloc[:,1:33] = scaler.fit_transform(dt.iloc[:,1:33])
#a=dt[dt.label == 0]
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)

dt=dt.sample(frac=1)
X_train=dt.iloc[0:168000,1:33]
Y_train=dt.iloc[0:168000,0]
X_test=dt.iloc[168000:209593,1:33]
Y_test=dt.iloc[168000:209593,0]


