library(keras)
library(Quandl)
Quandl.api_key("puJtYkz3w2mjsUvx_38R")


dat = Quandl(code = "HKEX/03988",type="xts")

head(dat)
tail(dat)