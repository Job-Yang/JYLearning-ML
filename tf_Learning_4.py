#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#2*2的零矩阵
zeros = tf.zeros([2, 2])

#长度为6的向量
ones = tf.ones([6])

#3*3*3的张量，其元素服从0~10的均匀分布
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)

#3*3*3张量,其元素服从0均值、标准差为2的正态分布
normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)

#生成在一定标准差情况下的数
#该对象的值域为[3, 7]
trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)



#创建一个可变对象
my_var = tf.Variable(1)

#变量赋值
my_var_times_two = my_var.assign(my_var * 2)

init = tf.global_variables_initializer()

#创建多个TensorFlow Session对象
sess1 = tf.Session()
sess2 = tf.Session()

#初始化所有的变量
sess1.run(init)
print(sess1.run(my_var_times_two))
print(sess1.run(my_var_times_two))
print(sess1.run(my_var.assign_add(5)))

sess2.run(init)
print(sess2.run(my_var_times_two))
print(sess2.run(my_var_times_two))
print(sess2.run(my_var_times_two))
print(sess2.run(my_var.assign_add(3)))

#重新初始化所有的变量
sess1.run(init)
sess2.run(init)

sess1.close()
sess2.close()
