import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense


dt=pd.read_csv('sample_data_intw.csv')
dt.dtypes
dt=dt.iloc[:,1:35]
dt=dt.drop('msisdn', axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
dt.iloc[:,1:33] = scaler.fit_transform(dt.iloc[:,1:33])
dt=dt.sample(frac=1)

##################### Upsampling part
#a=dt[dt.label == 0]
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)
#dt=dt.append(a, ignore_index=True)
####################################
X_train=dt.iloc[42000:,1:33].as_matrix()
Y_train=dt.iloc[42000:,0].as_matrix()
X_test=dt.iloc[0:42000,1:33].as_matrix()
Y_test=dt.iloc[0:42000,0].as_matrix()

np.random.seed(8)

#######################
#model1=Sequential()
#model1.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
#model1.add(Dense(15, kernel_initializer='normal', activation='relu'))
#model1.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

#model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###########
model1=Sequential()
model1.add(Dense(1, input_dim=32,kernel_initializer='normal',activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01),))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

############


print(model1.summary())

model1.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=250, batch_size=25000)



compare=pd.DataFrame(dt.iloc[0:42000,0])
compare['model']=model1.predict(X_test)
compare.to_csv("compare.csv")
