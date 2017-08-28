#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')

# a = tf.random_normal([2, 20])
# sess = tf.Seesion()
# out = sess.run()
# x, y = out

# plt.scatter(x, y)
# plt.show()

a = tf.constant(5, name = "input_a")
b = tf.constant(3, name = "input_b")
c = tf.multiply(a, b, name = "mul_c")
d = tf.add(a, b, name = "add_c")
e = tf.add(c, d, name = "add_e")

sess = tf.Session()
output = sess.run(e)
print(output)
writer = tf.summary.FileWriter('./my_graph', sess.graph)

writer.close()
sess.close()

# 运行后控制台执行下面的代码
# tensorboard --logdir="my_graph"
