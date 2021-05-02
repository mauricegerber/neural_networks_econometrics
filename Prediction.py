import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from tensorflow import keras

import os
import datetime
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

newmodel = tf.keras.models.load_model('prediction.h5')
#print(newmodel.summary())

# same n_staps as used to train the model
n_steps = 200
# Days into the future (y), same as used to train the model
lookup_step = 30 

start_date = '01.01.2018'
end_date = '29.04.2021'

df = si.get_data('^N225', start_date, end_date)
df = df[['adjclose', 'volume', 'open', 'high', 'low']]
df = df[-n_steps:]
index_data = df['adjclose'].copy()

#adjclose["date"] = adjclose.index
#adjclose.reset_index(inplace=True)

column_scaler = {}
for column in df.columns.values:
    scaler = preprocessing.MinMaxScaler()
    df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
    column_scaler[column] = scaler


df = np.array([df])

y_pred = newmodel.predict(df)
y_pred = np.squeeze(column_scaler["adjclose"].inverse_transform(y_pred))

print(y_pred)

index_data = index_data.reset_index()
index_data.reset_index(inplace=True)
index_data.columns = ['index', 'date', 'adjclose']
print(index_data)


# Wed. 5th = 28921.031
# Wed. 5th = 27989.307

#x_pred_date = pd.to_datetime(end_date) + datetime.timedelta(days = lookup_step)
#date_plot = pd.to_datetime(end_date) + datetime.timedelta(days = (lookup_step + 10))

plt.figure(figsize=(15, 8))
plt.xlim(index_data['date'][0], index_data['index'][n_steps-1] + lookup_step)
plt.plot(index_data['date'], index_data['adjclose'])
#plt.scatter(x = x_pred_date, y = y_pred, c = 'red')
plt.show()


#print(index_data['index'][n_steps-1] + lookup_step)
#plt.xlim(index_data['date'][0], index_data['index'][n_steps-1] + lookup_step)















