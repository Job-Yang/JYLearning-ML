#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant([5, 3], name = "input_a")
b = tf.reduce_prod(a, name = "prod_b")
c = tf.reduce_sum(a, name = "sum_c")
d = tf.add(b, c, name = "add_d")

sess = tf.Session()
output = sess.run(d)
print(output)
writer = tf.summary.FileWriter('./my_graph', sess.graph)

writer.close()
sess.close()

# 运行后控制台执行下面的代码
# tensorboard --logdir="my_graph"
