import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
tf.disable_v2_behavior()


df = pd.read_csv("FuelConsumptionCo2.csv")

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])
a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b
loss = tf.reduce_mean(tf.square(y - train_y))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_values = []
train_data = []
for step in range(300):
    _, loss_val, a_val, b_val = sess.run([optimizer, loss, a, b])
    loss_values.append(loss_val)
    if step % 3 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])
plt.plot(loss_values, 'ro')
plt.show()

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])
plt.show()