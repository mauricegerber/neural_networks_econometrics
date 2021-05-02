from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import pathlib

#os.mkdir("plots")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

start_date = '01.01.2000'
end_date = '01.01.2021'
ticker = '^N225'

fig_size = (15,8)

df = si.get_data(ticker, start_date, end_date)
df = df['adjclose']
print(df)

# start and end date from data set used for plot
start_plot = df.index[0]
end_plot = df.index[-1]

## Plot price graph
#plt.figure(figsize = fig_size)
#plt.plot(df, c='steelblue', linewidth = 0.9)
#plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}")
#plt.ylabel("Adjusted closing price in JPY")
#plt.legend(["Nikkei 225 Index [^N225]"], loc = 9, frameon = False)
#plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', 
#	f'{ticker}_{start_date}_{end_date}_price_data.png'), dpi = 600)  
#plt.close()


## Plot daily returns graph
#daily_returns = df.pct_change()

#plt.figure(figsize = fig_size)
#plt.ylabel("Daily returns in %")
#plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}")
#plt.plot(daily_returns, c='steelblue', linewidth = 0.9)
#plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', 
#	f'{ticker}_{start_date}_{end_date}_daily_returns.png'), dpi = 600)  
#plt.close()





# load different optimizer
adam = pd.read_csv('optimizer/^N225_Adam.csv')
adamax = pd.read_csv('optimizer/^N225_Adamax.csv')
nadam = pd.read_csv('optimizer/^N225_Nadam.csv')
RMSprop = pd.read_csv('optimizer/^N225_RMSprop.csv')
SGD = pd.read_csv('optimizer/^N225_SGD.csv')

## Plot different optimizer graphs
plt.figure(figsize = fig_size)
plt.plot(adam['adjclose_5'], c='olivedrab', linewidth = 0.9)
plt.plot(adamax['adjclose_5'], c='gold', linewidth = 0.9)
plt.plot(nadam['adjclose_5'], c='deepskyblue', linewidth = 0.9)
plt.plot(RMSprop['adjclose_5'], c='brown', linewidth = 0.9)
plt.plot(SGD['adjclose_5'], c='crimson', linewidth = 0.9)
plt.ylabel("Adjusted closing price in JPY")
plt.xlabel("Time steps for test set")
plt.legend(['Adam','Adamax','Nadam','RMSprop','SGD'],loc = 9, frameon = False, ncol = 5)
plt.show()

#plt.savefig(os.path.join('plots', f'{ticker}_different_optimizer.png'), dpi = 600)  
#plt.close()












