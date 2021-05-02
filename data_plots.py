from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os

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
plt.figure(figsize = fig_size)
plt.plot(df, c='steelblue', linewidth = 0.9)
plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}")
plt.ylabel("Adjusted closing price in JPY")
plt.legend(["Nikkei 225 Index [^N225]"], loc = 9, frameon = False)
plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', 
#	f'{ticker}_{start_date}_{end_date}_price_data.png'), dpi = 600)  
#plt.close()


## Plot daily returns graph
daily_returns = df.pct_change()
plt.figure(figsize = fig_size)
plt.ylabel("Daily returns in %")
plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}")
plt.plot(daily_returns, c='steelblue', linewidth = 0.9)
plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', 
#	f'{ticker}_{start_date}_{end_date}_daily_returns.png'), dpi = 600)  
#plt.close()





