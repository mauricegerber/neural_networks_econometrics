import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from tensorflow import keras

import os
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

newmodel = tf.keras.models.load_model('mymodel.h5')
print(newmodel.summary())

n_steps = 200

df = si.get_data('^N225', '01.01.2018', '29.04.2021')
df = df[['adjclose', 'volume', 'open', 'high', 'low']]
df = df[-n_steps:]
print(df)

column_scaler = {}
for column in df.columns.values:
    scaler = preprocessing.MinMaxScaler()
    df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
    column_scaler[column] = scaler


df = np.array([df])

y_pred = newmodel.predict(df)
y_pred = np.squeeze(column_scaler["adjclose"].inverse_transform(y_pred))

print(y_pred)
#print(column_scaler)

# Wed. 5th = 28921.031
# Wed. 5th = 27989.307