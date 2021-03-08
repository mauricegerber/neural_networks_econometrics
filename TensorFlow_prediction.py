import quandl
import pandas as pd
import matplotlib.pyplot as plt

# 03988 = Stock price, Bank Of China, date and nominal price (closing price)
quandl.ApiConfig.api_key = 'puJtYkz3w2mjsUvx_38R'
dat1 = quandl.get('HKEX/03988', column_index='1')
print(dat1)
# head-value; 2.84 / 2021-03-05, tail-value; 3.25 / 2014-02-21 
plt.plot(dat1)

dat1 = pd.DataFrame(data=dat1)
print(dat1)

# Dimensions of dataset
n = dat1.shape[0]
p = dat1.shape[1]
# Make data a numpy array / Drop date variable
#dat1.reset_index(inplace=True)
#dat1.drop(columns=["Date"], inplace=True)
dat1 = dat1.values
print(dat1)









