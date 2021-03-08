import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#train_start = 0
#train_end = int(np.floor(0.8*n))
#test_start = train_end
#test_end = n
#data_train = dat1[np.arange(train_start, train_end), :]
#data_test = dat1[np.arange(test_start, test_end), :]


dat1.tail(5)

#train = head(dat1,round(0.65*nrow(dat1))) 
#test = tail(dat1,round(0.25*nrow(dat1)))






