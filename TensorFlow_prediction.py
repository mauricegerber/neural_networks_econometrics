import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# 03988 = Stock price, Bank Of China, date and nominal price (closing price)
quandl.ApiConfig.api_key = 'puJtYkz3w2mjsUvx_38R'
dat1 = quandl.get('HKEX/03988', column_index='1')
print(dat1)
# head-value; 2.84 / 2021-03-05, tail-value; 3.25 / 2014-02-21 
#plt.plot(dat1)
#plt.show()

dat1 = pd.DataFrame(data=dat1)
print(dat1)

# Dimensions of dataset
n = dat1.shape[0]
p = dat1.shape[1]
# Make data a numpy array / Drop date variable
dat1.reset_index(inplace=True)
dat1.drop(columns=["Date"], inplace=True)
#dat1 = dat1.values
print(dat1)
plt.plot(dat1)
#plt.show()

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
X_train = dat1_train[:, 1:]
y_train = dat1_train[:, 0]
X_test = dat1_test[:, 1:]
y_test = dat1_test[:, 0]




