library(keras)
library(Quandl)
Quandl.api_key("puJtYkz3w2mjsUvx_38R")


dat = Quandl(code = "BITSTAMP/USD",type="xts")

head(dat)
tail(dat)
