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


n_steps = 50
sequence_data = []
sequences = deque(maxlen=n_steps)
for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
    sequences.append(entry)
    if len(sequences) == n_steps:
        sequence_data.append([np.array(sequences), target])

print(last_sequence)
last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
last_sequence = np.array(last_sequence).astype(np.float32)



'''
print(df)
         index      open      high       low         close  adjclose  \
0   2019-01-04  0.000000  0.000000  0.000000  19561.960938  0.000000   
1   2019-01-07  0.065257  0.130416  0.142884  20038.970703  0.105904   
2   2019-01-08  0.128391  0.148990  0.181906  20204.039062  0.142552   
3   2019-01-09  0.160318  0.182281  0.229190  20427.060547  0.192067   
4   2019-01-10  0.138808  0.148536  0.180975  20163.800781  0.133619  

       volume ticker       date    future  
0    0.787618  ^N225 2019-01-04  0.105904  
1    0.700774  ^N225 2019-01-07  0.142552  
2    0.742906  ^N225 2019-01-08  0.192067  
3    0.625967  ^N225 2019-01-09  0.133619  
4    0.633706  ^N225 2019-01-10  0.177112 
'''


''' 
output of sequence_data.append
jedes element aus der Liste besteht aus einer kleineren Liste, das erste Element der kleineren Liste
ist ein np.array, welches n_steps Spalten des Dataframes enthaelt
Das zweite Element dieser kleineren Liste ist der future Wert

[[array([[0.0, 0.7876182287188306, 0.0, 0.0, 0.0,
        Timestamp('2019-01-04 00:00:00')],
       [0.10590430977625243, 0.7007738607050731, 0.06525682604901473,
        0.1304161715626453, 0.14288360622878482,
        Timestamp('2019-01-07 00:00:00')]], dtype=object), 0.14255230301513855], [array([[0.10590430977625243, 0.7007738607050731, 0.06525682604901473,
        0.1304161715626453, 0.14288360622878482,
        Timestamp('2019-01-07 00:00:00')],
       [0.14255230301513855, 0.7429062768701633, 0.1283905696747043,
        0.14899034302937952, 0.18190634149226614,
        Timestamp('2019-01-08 00:00:00')]], dtype=object), 0.19206687917284704], [array([[0.14255230301513855, 0.7429062768701633, 0.1283905696747043,
        0.14899034302937952, 0.18190634149226614,
        Timestamp('2019-01-08 00:00:00')],
       [0.19206687917284704, 0.6259673258813413, 0.1603182598297126,
        0.1822808709567978, 0.2291898734593154,
        Timestamp('2019-01-09 00:00:00')]], dtype=object), 0.13361871777337697]]
'''
'''
output of df[feature_columns + ["date"]].values
[[0.0 0.7876182287188306 0.0 0.0 0.0 Timestamp('2019-01-04 00:00:00')]
 [0.10590430977625243 0.7007738607050731 0.06525682604901473
  0.1304161715626453 0.14288360622878482 Timestamp('2019-01-07 00:00:00')]
 [0.14255230301513855 0.7429062768701633 0.1283905696747043
  0.14899034302937952 0.18190634149226614
  Timestamp('2019-01-08 00:00:00')]
 ...
 [0.9371136653673746 0.27171109200343935 0.937436873301027
  0.9394640976318485 0.9550716494320293 Timestamp('2019-12-25 00:00:00')]
 [0.9686513633886049 0.3465176268271711 0.9316003779456556
  0.9637130936585834 0.9535009808463863 Timestamp('2019-12-26 00:00:00')]
 [0.9492916484723963 0.38521066208082544 0.9690330244546335
  0.9718225866898278 0.9666068629673807 Timestamp('2019-12-27 00:00:00')]]
  '''




'''
output of:
np.expand_dims(df['close'].values, axis=1)

[[10654.79003906]
 [10681.83007812]
 [10731.45019531]
 ...
 [26854.02929688]
 [27568.15039062]
 [27444.16992188]]

after scale:

 [[0.12874963]
 [0.13121962]
 [0.13151123]
 ...
 [0.96490293]
 [0.97826252]
 [1.        ]]
'''



'''


nikkei = pd.DataFrame(data=nikkei)

dat1 = nikkei[['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']]

# add date as a column
#dat1.index.names = ['date']
#dat1.reset_index(inplace=True)
#dat1.drop(columns=["date"], inplace=True)
dat1 = dat1.assign(ticker='1')
dat1['ticker'] = dat1['ticker'].astype(float)
# print(dat1)


#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


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

# TEST FILE

# set seed, so we can get the same results after rerunning several times
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)


pd.set_option('display.max_columns', None)


nikkei = si.get_data("^N225", start_date = "01/01/2010", end_date = "01/01/2021")
#print(nikkei)

nikkei = pd.DataFrame(data=nikkei)

dat1 = nikkei[['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']]

# add date as a column
dat1.index.names = ['date']
#dat1.reset_index(inplace=True)
#dat1.drop(columns=["date"], inplace=True)
dat1 = dat1.assign(ticker='1')
dat1['ticker'] = dat1['ticker'].astype(float)
#print(dat1)

# Training and test data
dat1_train = dat1.head(int(len(dat1)*(0.8)))
print(dat1_train)
dat1_test = dat1.tail(int(len(dat1)*(0.2)))
print(dat1_test)

# Scale data
scaler = MinMaxScaler()
dat1_train = scaler.fit_transform(dat1_train)
dat1_test = scaler.transform(dat1_test)



print(dat1_train)

 

'''
