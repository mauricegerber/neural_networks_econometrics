from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from yahoo_fin import stock_info as si
from matplotlib import pyplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# M/D/Y
start_date = '01.01.2015'
end_date = '05.01.2021'
ticker = '^N225'

fig_size = (15,8)

df = si.get_data(ticker, start_date, end_date)
df = df[['adjclose']]
df.dropna(inplace = True)

diff_data = df.diff()[1:]
print(diff_data)


# ACF plot
plot_acf(diff_data, lags = 20)
pyplot.show()

# PACF plot
plot_pacf(diff_data, lags = 20)
#pyplot.show()


# ARMA
model = ARMA(diff_data, order = (2,2))
model_fit = model.fit()


#summary of the model
print(model_fit.summary())

print(len(diff_data))

predictions = model_fit.forecast(40)
print(predictions[0])


#print(df['adjclose'].iloc[-1])


x, x_diff = df['adjclose'].iloc[-1], predictions[0]
pred_values = np.r_[x, x_diff].cumsum().astype(float)

y_pred = pred_values[-1]
print(y_pred)


# Final prediction
# 29117.531240121705






