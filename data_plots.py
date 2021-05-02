from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
import pandas as pd
import os

#os.mkdir("plots")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

start_date = '01.01.2020'
end_date = '01.01.2021'
ticker = '^N225'

df = si.get_data(ticker, start_date, end_date)
df = df['adjclose']
print(df)

# plot true/pred prices graph
plt.figure(figsize=(15, 8))
plt.plot(df, c='dodgerblue')
plt.xlabel(f"Date from {start_date} to {end_date}")
plt.ylabel("Adjusted closing price in JPY")
plt.legend(["Nikkei 225 Index [N225]"], loc = 8, frameon = False)
plt.show()


plt.savefig(os.path.join('plots', f'{ticker}_{start_date}_{end_date}_price_data.png'))  
