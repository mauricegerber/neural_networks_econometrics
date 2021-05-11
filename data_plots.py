from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
import pathlib
from matplotlib import rcParams
import statistics
from functools import reduce

#os.mkdir("plots")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#start_date = '2018.11.06' if input data from "01.01.2010" to "01.01.2021"
#end_date = '2020.12.29'

#start_date = '2015.12.02' if input data from "01.01.1995" to "01.01.2021"
#end_date = '2020.12.29'


start_date = '12.02.2015'
end_date = '12.29.2020'
ticker = '^N225'

fig_size = (15,8)
size = 18 # text size
dpi = 500
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']

df = si.get_data(ticker, start_date, end_date)
orig_price = df.copy()
orig_price.reset_index(inplace=True)
df = df['adjclose']
#print(orig_price)

# start and end date from data set used for plot
start_plot = df.index[0]
end_plot = df.index[-1]

## Plot index price graph
#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	label.set_fontsize(size)

#ax.plot(df, c='steelblue', linewidth = 0.9)
#plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}", fontsize = size + 4, labelpad = 20)
#plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
#ax.legend(["Nikkei 225 Index [^N225]"], loc = 9, frameon = False, fontsize = size)
#plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', f'{ticker}_{start_date}_{end_date}_price_data.png'), dpi = dpi)  
#plt.close()


## Plot daily returns graph
#daily_returns = df.pct_change()
#daily_returns = list(daily_returns)
#del daily_returns[0]
#mean = statistics.mean(daily_returns)
#mean = round(mean, 6)
#std = statistics.stdev(daily_returns)
#lower = mean - std * 3
#upper = mean + std * 3
#print(mean)

#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	label.set_fontsize(size)

#ax.plot(daily_returns, c='steelblue', linewidth = 0.9)
#ax.axhline(mean, c='orangered')
#ax.axhline(lower, c='sandybrown', ls = ':')
#ax.axhline(upper, c='sandybrown', ls = ':')
#plt.ylabel("Daily returns in %", fontsize = size + 4, labelpad = 20)
#plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}", fontsize = size + 4, labelpad = 20)
#plt.legend(['^N225 returns [%]',f'mean:{mean}', 'upper and lower band'],loc = 9, frameon = False, ncol = 6, fontsize = size)
#plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', f'{ticker}_{start_date}_{end_date}_daily_returns.png'), dpi = dpi)  
#plt.close()





# load different optimizer
#adam = pd.read_csv('optimizer/^N225_Adam.csv')
#adamax = pd.read_csv('optimizer/^N225_Adamax.csv')
#nadam = pd.read_csv('optimizer/^N225_Nadam.csv')
#RMSprop = pd.read_csv('optimizer/^N225_RMSprop.csv')
#SGD = pd.read_csv('optimizer/^N225_SGD.csv')

## Plot different optimizer graphs

#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	label.set_fontsize(size)

#ax.plot(orig_price['adjclose'], c='steelblue', linewidth = 0.9)
#ax.plot(adam['adjclose_1'], c='crimson', linewidth = 0.9)
#ax.plot(adamax['adjclose_1'], c='gold', linewidth = 0.9)
#ax.plot(nadam['adjclose_1'], c='olive', linewidth = 0.9)
#ax.plot(RMSprop['adjclose_1'], c='brown', linewidth = 0.9)
#ax.plot(SGD['adjclose_1'], c='deepskyblue', linewidth = 0.9)
#plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
#plt.xlabel("Time steps for test set", fontsize = size + 4, labelpad = 20)
#plt.legend(['^N225','Adam','Adamax','Nadam','RMSprop','SGD'],loc = 9, frameon = False, ncol = 6, fontsize = size)
#plt.show()

#plt.savefig(os.path.join('plots', f'{ticker}_different_optimizer.png'), dpi = dpi)  
#plt.close()


# Plot different inputs plot

# n_steps
# load different n_step
#df_5 = pd.read_csv('inputs/^N225_5_0.5309752031266569.csv')
#df_25 = pd.read_csv('inputs/^N225_25_0.97889434867139.csv')
#df_50 = pd.read_csv('inputs/^N225_50_1.1617199826213775.csv')
#df_100 = pd.read_csv('inputs/^N225_100_1.5143371860802408.csv')


#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	label.set_fontsize(size)

#ax.plot(orig_price['adjclose'], c='steelblue', linewidth = 0.9)
#ax.plot(df_5['adjclose_1'], c='olive', linewidth = 0.9)
#ax.plot(df_25['adjclose_1'], c='lightseagreen', linewidth = 0.9)
#ax.plot(df_50['adjclose_1'], c='gold', linewidth = 0.9)
#ax.plot(df_100['adjclose_1'], c='crimson', linewidth = 0.9)
#plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
#plt.xlabel("Time steps for test set", fontsize = size + 4, labelpad = 20)
#plt.legend(['^N225','n_step = 5','n_step = 25','n_step = 50','n_step = 100'],
#	loc = 9, frameon = False, ncol = 5, fontsize = size)
#plt.show()

#plt.savefig(os.path.join('plots', f'{ticker}_different_n_step.png'), dpi = dpi)  
#plt.close()

# batch_size
# load different batch_size
#df_5 = pd.read_csv('inputs/batch_size_5_1.4735569169843192.csv')
#df_50 = pd.read_csv('inputs/batch_size_50_1.5455034104023815.csv')
#df_100 = pd.read_csv('inputs/batch_size_100_1.5686194127492417.csv')


#fig, ax = plt.subplots(figsize = fig_size)
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	label.set_fontsize(size)

#ax.plot(orig_price['adjclose'], c='steelblue', linewidth = 0.9)
#ax.plot(df_5['adjclose_1'], c='olive', linewidth = 0.9)
#ax.plot(df_50['adjclose_1'], c='gold', linewidth = 0.9)
#ax.plot(df_100['adjclose_1'], c='crimson', linewidth = 0.9)
#plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
#plt.xlabel("Time steps for test set", fontsize = size + 4, labelpad = 20)
#plt.legend(['^N225','batch_size = 5','batch_size = 50','batch_size = 100'],
#	loc = 9, frameon = False, ncol = 5, fontsize = size)
#plt.show()

#plt.savefig(os.path.join('plots', f'{ticker}_different_batch_size.png'), dpi = dpi)  
#plt.close()


# final model (several runs)
# load different models
df_1 = pd.read_csv('inputs/final_model_0.98757384906139.csv')
df_2 = pd.read_csv('inputs/final_model_0.876493997041864.csv')
df_3 = pd.read_csv('inputs/final_model_0.8340478935634371.csv')
df_4 = pd.read_csv('inputs/final_model_0.8372230854291572.csv')
df_5 = pd.read_csv('inputs/final_model_0.8453413742642574.csv')
df_6 = pd.read_csv('inputs/final_model_0.8564342617517773.csv')
df_7 = pd.read_csv('inputs/final_model_0.8720014660194579.csv')
df_8 = pd.read_csv('inputs/final_model_0.8790596623010138.csv')
df_9 = pd.read_csv('inputs/final_model_0.8792852113916764.csv')
df_10 = pd.read_csv('inputs/final_model_0.8808236052156138.csv')
df_11 = pd.read_csv('inputs/final_model_0.9017172334599305.csv')
df_12 = pd.read_csv('inputs/final_model_0.9373521565329678.csv')
df_13 = pd.read_csv('inputs/final_model_0.9764776498934619.csv')
df_14 = pd.read_csv('inputs/final_model_0.9947950864659025.csv')
df_15 = pd.read_csv('inputs/final_model_0.9949503806221899.csv')
df_16 = pd.read_csv('inputs/final_model_0.9969162177673935.csv')
df_17 = pd.read_csv('inputs/final_model_0.9987860161807575.csv')
df_18 = pd.read_csv('inputs/final_model_1.0085844505886985.csv')
df_19 = pd.read_csv('inputs/final_model_1.0095629338750394.csv')
df_20 = pd.read_csv('inputs/final_model_1.0197653932927881.csv')
df_21 = pd.read_csv('inputs/final_model_0.8769765561870677.csv')
# original price
df_orig = pd.read_csv('inputs/final_model_orig_price.csv')


data_frames = [df_1['adjclose_1'], df_2['adjclose_1'], df_3['adjclose_1'], df_4['adjclose_1'], 
df_5['adjclose_1'], df_6['adjclose_1'], df_7['adjclose_1'], df_8['adjclose_1'], 
df_9['adjclose_1'], df_10['adjclose_1'], df_11['adjclose_1'], 
df_12['adjclose_1'], df_13['adjclose_1'], df_14['adjclose_1'], 
df_15['adjclose_1'], df_16['adjclose_1'], df_17['adjclose_1'],
df_18['adjclose_1'], df_19['adjclose_1'], df_20['adjclose_1'],
df_21['adjclose_1']]

adjclose = pd.concat(data_frames, axis=1, keys=["1", "2", "3", "4",
	"5", "6", "7","8", "9", "10","11", "12", "13","14", "15", "16",
	"17", "18", "19", "20", "21"])

col = adjclose.loc[: , "1":"21"]

# calculate mean values
adjclose['adjclose_mean'] = col.mean(axis=1)


fig, ax = plt.subplots(figsize = fig_size)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(size)

ax.plot(df_orig['adjclose'], c='steelblue', linewidth = 1)
ax.plot(adjclose['adjclose_mean'], c = 'springgreen', linewidth = 1)
ax.plot(adjclose['1'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['2'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['3'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['4'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['5'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['6'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['7'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['8'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['9'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['10'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['11'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['12'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['13'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['14'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['15'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['16'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['17'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['18'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['19'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['20'], c='orangered', linewidth = 0.6, alpha = 0.6)
ax.plot(adjclose['21'], c='orangered', linewidth = 0.6, alpha = 0.6)
plt.ylabel("Adjusted closing price in JPY", fontsize = size + 4, labelpad = 20)
plt.xlabel(f"Date from {start_plot.strftime('%Y-%m-%d')} to {end_plot.strftime('%Y-%m-%d')}", fontsize = size + 4, labelpad = 20)
plt.legend(['^N225', 'adjclose_mean','model $n_{1,...,21}$'],loc = 9, frameon = False, ncol = 6, fontsize = size)
plt.show() # for saving plot, dont show it

#plt.savefig(os.path.join('plots', f'{ticker}_final_models_21.png'), dpi = dpi)  
#plt.close()





