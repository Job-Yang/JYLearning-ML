#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# 模型定义的相关代码
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


# 读取训练数据
def inputs():
	# Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
	weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
	blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
	return tf.to_float(weight_age), tf.to_float(blood_fat_content)

# 计算推断模型
def inference(X):
	return tf.matmul(X, W) + b

# 计算相对期望输出的损失
def loss(X, Y):
	Y_predicted = inference(X)
	return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

# 调整模型参数
def train(total_loss):
	learing_rate = 0.0000001
	return tf.train.GradientDescentOptimizer(learing_rate).minimize(total_loss)

# 评估训练得到的模型
def evaluate(sess, X, Y):
	print(sess.run(inference([[80.,25.]]))) # ~303
	print(sess.run(inference([[65.,25.]]))) # ~256
	return

# 启动会话，运行训练闭环
with tf.Session() as sess:
	tf.initialize_all_variables().run()

	X, Y = inputs()

	total_loss = loss(X, Y)
	train_op = train(total_loss)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# 实际的训练迭代次数
	training_steps = 10000
	for step in range(training_steps):
		sess.run([train_op])
		# 处于调试和学习的目的，查看损失在训练过程中递减的情况
		if step % 10 == 0:
			print ("loss: ", sess.run([total_loss]))

	evaluate(sess, X, Y)

	coord.request_stop()
	coord.join(threads)
	sess.close()

