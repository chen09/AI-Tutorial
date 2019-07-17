# 最简单的AI例子，一元一次函数 y = ax + b 的回归计算

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据准备
x_data = np.linspace(-1, 1, 10, dtype=np.float)
print(x_data)
y_data = np.dot(2.5, x_data) + 0.5 + np.random.randn(10) / 5
print(y_data)

# 模型准备
# 偏移量
b = tf.Variable(tf.zeros([1]) + 0.1)
# 权重（斜率）
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 目标函数
y = tf.multiply(w, x_data) + b
# 损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化器
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(0, 10000):
        sess.run(train)
        if step % 1000 == 0:
            print(sess.run(w), sess.run(b), sess.run(loss))
    y_data_new = sess.run(y)

plt.scatter(x_data, y_data)
plt.plot(x_data, y_data_new, color='red')
plt.show()
