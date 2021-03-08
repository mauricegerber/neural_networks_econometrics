import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

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
dat1_train = dat1.head(int(len(dat1)*(0.65)))
print(dat1_train)
dat1_test = dat1.tail(int(len(dat1)*(0.25)))
print(dat1_test)


# Scale data
scaler = MinMaxScaler()
dat1_train = scaler.fit_transform(dat1_train)
dat1_test = scaler.transform(dat1_test)
# Build X and y
X_train = dat1_train[:, 1:]
y_train = dat1_train[:, 0]
X_test = dat1_test[:, 1:]
y_test = dat1_test[:, 0]

# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)

# Define the addition
c = tf.add(a, b)
print(c)

# Initialize the graph
graph = tf.Session()

# Run the graph
graph.run(c, feed_dict={a: 5, b: 4})

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
#Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Model architecture parameters
#n_stocks = 500
#n_neurons_1 = 1024
#n_neurons_2 = 512
#n_neurons_3 = 256
#n_neurons_4 = 128
#n_target = 1
# Layer 1: Variables for hidden weights and biases
#W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
#bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
#W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
#bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
#W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
#bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
#W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
#bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
#W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
#bias_out = tf.Variable(bias_initializer([n_target]))


