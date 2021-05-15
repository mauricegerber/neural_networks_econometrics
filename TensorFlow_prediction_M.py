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
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# TEST FILE

#np.random.seed(1)
#tf.random.set_seed(1)
#random.seed(1)


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


## INPUT
# Data
ticker = "^N225"
start_date = "01.01.2010"
end_date = "05.01.2021"
# Days into the future (y)
lookup_step = 2 
# Days back (X), window size or the sequence length
n_steps = 5
# Test size
test_size = 0.2
# Feature column
feature_columns=['adjclose', 'volume', 'open', 'high', 'low']
# shuffle of training/test data
shuffle = False
# Layers 
n_layers = 4
# dropout
dropout = 0.3
# Optimizer
optimizer = "RMSprop"
# Loss
loss = "huber_loss"
# LSTM cell
cell = LSTM
# LSTM neurons
units = 256
# Batch size
batch_size = 100
# Epochs
epochs = 1

# Possible Optimizers
#class Adam: Optimizer that implements the Adam algorithm. OK
#class Adamax: Optimizer that implements the Adamax algorithm. OK
#class Nadam: Optimizer that implements the NAdam algorithm. OK
#class RMSprop: Optimizer that implements the RMSprop algorithm. OK

#class Adadelta: Optimizer that implements the Adadelta algorithm. NOT
#class Adagrad: Optimizer that implements the Adagrad algorithm. NOT
#class Ftrl: Optimizer that implements the FTRL algorithm. NOT
#class SGD: Gradient descent (with momentum) optimizer. NOT

# create folder to store data frames
#os.mkdir("inputs")

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(ticker, start_date, end_date, n_steps=50, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=feature_columns):
    # load it from yahoo_fin library
    df = si.get_data(ticker, start_date, end_date)
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # add date as a column
    df["date"] = df.index
    column_scaler = {}
    # scale the data (prices) to a value within 0 and 1
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # split the dataset into training & testing sets by date (not randomly splitting)
    train_samples = int((1 - test_size) * len(X))
    result["X_train"] = X[:train_samples]
    result["y_train"] = y[:train_samples]
    result["X_test"]  = X[train_samples:]
    result["y_test"]  = y[train_samples:]
    if shuffle:
        # shuffle the datasets for training (if shuffle parameter is set)
        shuffle_in_unison(result["X_train"], result["y_train"])
        shuffle_in_unison(result["X_test"], result["y_test"])
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result



def create_model(sequence_length, n_features, units, cell, n_layers, dropout, loss, optimizer):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, 
                batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

# load the data
data = load_data(ticker, start_date, end_date, n_steps, shuffle, lookup_step, test_size, feature_columns)

# date now
date_now = time.strftime("%Y-%m-%d")

# model name to save, making it as unique as possible based on parameters
## create these folders if they does not exist
#if not os.path.isdir("results"):
#    os.mkdir("results")
#if not os.path.isdir("logs"):
#    os.mkdir("logs")

model_name = f"{date_now}_{ticker}-{loss}-{optimizer}-{cell.__name__}-seq-{n_steps}-step-\
{lookup_step}-layers-{n_layers}-units-{units}"

# construct the model
model = create_model(n_steps, len(feature_columns), loss=loss, units=units, 
    cell=cell, n_layers=n_layers, dropout=dropout, optimizer=optimizer)
# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), 
    save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# train the model and save the weights whenever we see 
# a new optimal model using ModelCheckpoint
model.fit(data["X_train"], data["y_train"], batch_size=batch_size, epochs=epochs, 
    validation_data=(data["X_test"], data["y_test"]),
    callbacks=[checkpointer, tensorboard],
    verbose=1)


model.save('prediction.h5')

def get_final_df(model, data):
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{lookup_step}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{lookup_step}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    return test_df


# load optimal model weights from results folder
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# get the final dataframe for the testing set
final_df = get_final_df(model, data)

# calculate the mean absolute error (MAE)
mae = mean_absolute_error(final_df[f'true_adjclose_{lookup_step}'], 
    final_df[f'adjclose_{lookup_step}'])
mae = round(mae,1)

# calculate the annualized Sharpe ratio
# daily returns original data and predicted data
daily_return_orig = final_df['adjclose'].pct_change()
sharpe_ratio_orig = daily_return_orig.mean() / daily_return_orig.std()
sharpe_ratio_orig = sharpe_ratio_orig * (252**0.5)
print(sharpe_ratio_orig)

daily_return_pred = final_df[f'adjclose_{lookup_step}'].pct_change()
sharpe_ratio_pred = daily_return_pred.mean() / daily_return_pred.std()
sharpe_ratio_pred = sharpe_ratio_pred * (252**0.5)
print(sharpe_ratio_pred)

# save different input data frames### look_up
#different_input = final_df[f'adjclose_{lookup_step}'].copy()
#different_input.to_csv(os.path.join('inputs', f'{ticker}_{n_steps}_{sharpe_ratio_pred}.csv'))

### batch size
#different_input = final_df[f'adjclose_{lookup_step}'].copy()
#different_input.to_csv(os.path.join('inputs', f'batch_size_{batch_size}_{sharpe_ratio_pred}.csv'))

### final model
#different_input = final_df[f'adjclose_{lookup_step}'].copy()
#different_input.to_csv(os.path.join('inputs', f'final_model_{sharpe_ratio_pred}.csv'))
# orig price
#different_input = final_df[f'adjclose'].copy()
#different_input.to_csv(os.path.join('inputs', f'final_model_orig_price.csv'))

# save predicted price with different optimizations for later display
#different_optim = final_df[f'adjclose_{lookup_step}'].copy()
#different_optim.to_csv(os.path.join('optimizer', f'{ticker}_{optimizer}.csv'))



# start and end date from test data set used for plot
start_plot = final_df.index[0]
end_plot = final_df.index[-1]


# plot true/pred prices graph
def plot_graph(test_df):
    plt.figure(figsize=(15, 8))
    plt.plot(test_df[f'true_adjclose_{lookup_step}'], c='steelblue')
#    plt.plot(test_df[f'adjclose'], c='yellow')
    plt.plot(test_df[f'adjclose_{lookup_step}'], c='firebrick')
    plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}")
    plt.ylabel("Adjusted closing price in JPY")
    plt.legend(["Actual", f"Predicted [MAE:{mae}]"], loc = 9, 
        frameon = False, ncol = 2)
    # save the plot
#    plt.savefig(os.path.join('plots', 
#        f'{ticker}_{start_date}_{end_date}_layers:{n_layers}_ls:{lookup_step}_ns:{n_steps}_ep:{epochs}_batch:{batch_size}_units:{units}.png'), dpi = 600)  
#    plt.close()
    plt.show()


## print Output
print(model.summary())
#print("Mean Absolute Error:", mae)
#plot_graph(final_df)
#print(final_df)

# 2
# 793,857
# 1.615100266570783

# 3
# 1,319,169
# 1.457351208906496

# 4
# 1,844,481
# 1.4119590700191704

# 5
# 2,369,793
# 1.1976629511095334



















