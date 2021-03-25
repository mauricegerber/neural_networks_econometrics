#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# 03988 = Stock price, Bank Of China, date and nominal price (closing price)
#quandl.ApiConfig.api_key = 'puJtYkz3w2mjsUvx_38R'
#dat1 = quandl.get('HKEX/03988', column_index='1')
#print(dat1)
# head-value; 2.84 / 2021-03-05, tail-value; 3.25 / 2014-02-21 
#plt.plot(dat1)
#plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
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


nikkei = si.get_data("^N225", start_date = "01/01/2010", end_date = "01/01/2021")
#print(nikkei)

nikkei = pd.DataFrame(data=nikkei)

dat1 = nikkei[['adjclose', 'volume', 'open', 'high', 'low']]

# add date as a column
dat1.index.names = ['date']
dat1.reset_index(inplace=True)
dat1.drop(columns=["date"], inplace=True)
#print(dat1)

# Training and test data
dat1_train = dat1.head(int(len(dat1)*(0.65)))
print(dat1_train)
dat1_test = dat1.tail(int(len(dat1)*(0.25)))
print(dat1_test)

# Scale data
scaler = MinMaxScaler()
dat1_train = scaler.fit_transform(dat1_train)
dat1_test = scaler.transform(dat1_test)

# Build X and y
x_train = dat1_train[:, 1:]
y_train = dat1_train[:, 0]
x_test = dat1_test[:, 1:]
y_test = dat1_test[:, 0]

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15
# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# LSTM neurons
UNITS = 10
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 5



# model name to save, making it as unique as possible based on parameters
ticker = "N225"
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
#{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"


# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# train the model and save the weights whenever we see 
# a new optimal model using ModelCheckpoint
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)









