from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque


import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# TEST FILE


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


df = si.get_data("^N225", start_date = "01/01/2019", end_date = "01/01/2020")

result = {}
    # we will also return the original dataframe itself
result['df'] = df.copy()

df["date"] = df.index
df.reset_index(inplace=True)

feature_columns=['adjclose', 'volume', 'open', 'high', 'low']


column_scaler = {}
# scale the data (prices) from 0 to 1
for column in feature_columns:
    scaler = preprocessing.MinMaxScaler()
    df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
    column_scaler[column] = scaler # noch nicht sicher ob es gebraucht wird
# add the MinMaxScaler instances to the result returned
result["column_scaler"] = column_scaler


lookup_step = 1 
df['future'] = df['adjclose'].shift(-lookup_step) #time shift back


last_sequence = np.array(df[feature_columns].tail(lookup_step))
# drop NaNs
df.dropna(inplace=True) # cut last row 


n_steps = 5
sequence_data = []
sequences = deque(maxlen=n_steps)
for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
    sequences.append(entry)
    if len(sequences) == n_steps:
        sequence_data.append([np.array(sequences), target])

last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
last_sequence = np.array(last_sequence).astype(np.float32)
result['last_sequence'] = last_sequence


X, y = [], []
for seq, target in sequence_data:
    X.append(seq)
    y.append(target)

# convert to numpy arrays
X = np.array(X)
y = np.array(y)

test_size = 0.2
# split the dataset into training & testing sets by date (not randomly splitting)
train_samples = int((1 - test_size) * len(X))
result["X_train"] = X[:train_samples]
result["y_train"] = y[:train_samples]
result["X_test"]  = X[train_samples:]
result["y_test"]  = y[train_samples:]

shuffle = False

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

if shuffle:
    # shuffle the datasets for training (if shuffle parameter is set)
    shuffle_in_unison(result["X_train"], result["y_train"])
    shuffle_in_unison(result["X_test"], result["y_test"])



dates = result["X_test"][:, -1, -1] # print dates only

# retrieve test features from the original dataframe
result["test_df"] = result["df"].loc[dates] # add original data from test period to result

print(result["X_train"])
# remove dates from the training/testing sets & convert to float32
result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)