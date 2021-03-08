library(keras)
library(Quandl)
library(caTools)
library(dplyr)
library(tensorflow)

Quandl.api_key("puJtYkz3w2mjsUvx_38R")

# 03988 = Stock price, Bank Of China, date and nominal price (closing price)
dat = Quandl(code = "HKEX/03988", column_index='1')
colnames(dat)[2] = "price"

head(dat)
tail(dat)

plot(dat, type = "l")

train = head(dat,round(0.65*nrow(dat))) 
test = tail(dat,round(0.25*nrow(dat)))

#data1 = sample.split(dat$price,SplitRatio = 0.2)
##subsetting into Train data
#train =subset(dat$price,data1==TRUE)
#subsetting into Test data
#test =subset(dat$price,data1==FALSE)

head(test)
tail(test)
head(train)
tail(train)

# scale
test = scale(test, center = TRUE, scale = TRUE)
train = scale(train, center = TRUE, scale = TRUE)

plot(test, type = "l")
plot(train, type = "l")


# TensorFlow

input1 <- tf$placeholder(tf$float32)
input2 <- tf$placeholder(tf$float32)
print(input1)

ouput <- tf$multiply(input1, input2)

initiz <- tf$global_variables_initializer()
sess <- tf$Session()
print(sess$run(ouput, feed_dict = dict(input1=7, input2 = 2)))
sess$close()





