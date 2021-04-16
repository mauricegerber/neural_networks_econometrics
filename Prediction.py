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


newmodel = tf.keras.models.load_model('mymodel.h5')
print(newmodel.summary())

# load optimal model weights from results folder

a = np.array([[[0.99044895, 0.99312127, 0.936754,   0.9906605,  0.9535704 ], 
    [0.9747412,  0.45829752, 0.9693597,  0.98753715, 0.9902297],
    [1, 0.55030096, 1,         1, 1        ],
    [0.9477063 , 0.32588133, 0.94320554, 0.94599146, 0.9579066 ],
    [0.9477063 , 0.953501, 0.94320554, 0.94599146, 0.9637131 ],
    [0.9477063 , 0.32588133, 0.20554, 0.4599146, 0.9579066 ],
    [0.9477063 , 0.588133, 0.9637131, 0.9146, 0.9579066 ],
    [0.9477063 , 0.8133, 0.94320554, 0.94599146, 0.9579066 ],
    [0.93711364, 0.27171108, 0.9374369,  0.9394641,  0.9550716 ],
    [0.96865135, 0.34651762, 0.9316004,  0.9637131,  0.953501  ]]])

y_pred = newmodel.predict(a)

print(y_pred)


