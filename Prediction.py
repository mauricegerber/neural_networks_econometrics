import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from tensorflow import keras
from matplotlib import rcParams


from time import time
import os
import datetime
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

newmodel = tf.keras.models.load_model('prediction.h5')

# Days into the future (y), same as used to train the model
lookup_step = 30 
# same n_staps as used to train the model
n_steps = 200

# M/D/Y
start_date = '01.01.2010'
end_date = '05.01.2021'
ticker = '^N225'

fig_size = (15,8)

df = si.get_data(ticker, start_date, end_date)
df = df[['adjclose', 'volume', 'open', 'high', 'low']]
df = df[-n_steps:]
index_data = df['adjclose'].copy()
acf_data = df['adjclose'].copy()
#print(df)


column_scaler = {}
for column in df.columns.values:
    scaler = preprocessing.MinMaxScaler()
    df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
    column_scaler[column] = scaler

df = np.array([df])

# prediction function
y_pred = newmodel.predict(df)
y_pred = np.squeeze(column_scaler["adjclose"].inverse_transform(y_pred))
y_pred = y_pred.astype(int)
#print(y_pred)

y_pred_1 = 27942;y_pred_2 = 29316;y_pred_3 = 29146;y_pred_4 = 29242
y_pred_5 = 28146;y_pred_6 = 28812;y_pred_7 = 29036;y_pred_8 = 28790
y_pred_9 = 28933;y_pred_10 = 28017

index_data = index_data.reset_index()
index_data.reset_index(inplace=True)
index_data.columns = ['index', 'date', 'adjclose']
#print(index_data)

# Business day function to calculate the date in the future
def business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date

# x value for y_pred 
x_pred_date = business_days(index_data['date'][n_steps-1], lookup_step)
#print(x_pred_date)

# Plot date range
start_plot = index_data['date'][0]
end_plot = x_pred_date

fig_size = (15,8)
size = 18 # text size
dpi = 500
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']


# Plot prediction
fig, ax = plt.subplots(figsize = fig_size)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
   label.set_fontsize(size)

ax.plot(index_data['date'], index_data['adjclose'], c = 'steelblue')
ax.scatter(x = x_pred_date, y = y_pred_1, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_2, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_3, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_4, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_5, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_6, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_7, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_8, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_9, c = 'orangered', alpha = 0.3, s = 100)
ax.scatter(x = x_pred_date, y = y_pred_10, c = 'orangered', alpha = 0.3, s = 100)
plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}", fontsize = size + 4, labelpad = 20)
plt.ylim(21000, 31500)
plt.legend(['^N225', 'Predicted price in JPY'], loc = 9, frameon = False, ncol = 2, fontsize = size)

plt.show()

# save plot
#plt.savefig(os.path.join('plots', f'{ticker}_prediction_data.png'), dpi = dpi)  
#plt.close()







# Plot prediction ARMA
#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#   label.set_fontsize(size)

#ax.plot(index_data['date'], index_data['adjclose'], c = 'steelblue')
#ax.scatter(x = x_pred_date, y = 29117.5312, c = 'green', s = 50)
#ax.scatter(x = x_pred_date, y = y_pred_1, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_2, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_3, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_4, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_5, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_6, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_7, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_8, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_9, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = y_pred_10, c = 'orangered', alpha = 0.3, s = 100)
#ax.scatter(x = x_pred_date, y = 29117.5312, c = 'green', s = 50)
#plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
#plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}", fontsize = size + 4, labelpad = 20)
#plt.ylim(21000, 31500)
#plt.legend(['^N225', 'ARMA prediction', 'LSTM predictions'], loc = 9, frameon = False, ncol = 3, fontsize = size)

#plt.show()

# save plot
#plt.savefig(os.path.join('plots', f'{ticker}_prediction_data_ARMA.png'), dpi = dpi)  
#plt.close()








