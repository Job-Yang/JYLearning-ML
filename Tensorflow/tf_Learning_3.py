#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#创建一个长度为2，数据类型为int32的占位向量
a = tf.placeholder(tf.int32, shape=[2], name="my_input")

#将该占位向量视为其他任意Tensor对象，加以使用
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")

#完成数据流图的定义
d = tf.add(b, c, name="add_d")

#创建一个TensorFlow Session对象
sess = tf.Session()

#创建一个字典feed_dict
#key为：a占位符对象
#Value：[5, 3]
input_dict = {a : np.array([5, 3], dtype=np.int32)}
output = sess.run(d, feed_dict=input_dict)
print(output)

sess.close()

