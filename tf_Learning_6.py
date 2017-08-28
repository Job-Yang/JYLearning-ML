#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# 初始化变量和模型参数，定义训练闭环中的运算


# 读取训练数据
def inputs():
	# 读取或生成训练数据X及其期望输出Y

# 计算推断模型
def inference(X):
	# 计算推断模型在数据X上的输出，并将结果返回

# 计算相对期望输出的损失
def loss(X, Y):
	# 依据训练数据X及其期望输出Y计算损失

# 调整模型参数
def train(total_loss):
	# 依据计算的总损失训练数据或调整模型参数

# 评估训练得到的模型
def evaluate(sess, X, Y):
	# 对训练得到的模型进行评估

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
	for step in xrange(training_steps):
		sess.run([train_op])
		# 处于调试和学习的目的，查看损失在训练过程中递减的情况
		if step % 10 = 0:
			print "loss:",sess.run([total_loss])

	evaluate(sess, X, Y)

	coord.request_stop()
	coord.join(threads)
	sess.close()
