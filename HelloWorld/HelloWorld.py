import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

const = tf.constant("hello tf")

with tf.Session() as sess:
    result = sess.run(const)
    print(result)
