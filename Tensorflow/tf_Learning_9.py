#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'对数几率回归'

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# 模型定义的相关代码
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# 读CSV文件
def read_csv(batch_size, file_name, record_defaults):
	filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
	reader = tf.TextLineReader(skip_header_lines=1)
	key, value = reader.read(filename_queue)

	# decode_csv会将字符串（文本行）转换到具有指定默认值的由张量列构成的元组中
	# 他还会为每一列数据设置数据类型
	decoded = tf.decode_csv(value, record_defaults=record_defaults)

	# 实际上会读取一个文件，并加载一个长林中的batch_size行
	return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)


# 读取训练数据
def inputs():
	# Data from https://www.kaggle.com/c/titanic/data
	passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
		read_csv(100, "data/train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
	# 转换属性数据
	is_first_class = tf.to_float(tf.equal(pclass, [1]))
	is_second_class = tf.to_float(tf.equal(pclass, [2]))
	is_third_class = tf.to_float(tf.equal(pclass, [3]))
	gender = tf.to_float(tf.equal(sex, ["female"]))

	# 最终将所有的特征排列在一个矩阵中，然后对该矩阵转置，使其每行对应一个样本，每列对应一种特征
	features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
	survived = tf.reshape(survived, [100, 1])

	return features, survived

# 值合并
def combine_inputs(X):
	return tf.matmul(X, W) + b

# 计算推断模型
def inference(X):
	return tf.sigmoid(combine_inputs(X))

# 计算相对期望输出的损失
def loss(X, Y):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

# 调整模型参数
def train(total_loss):
	learing_rate = 0.01
	return tf.train.GradientDescentOptimizer(learing_rate).minimize(total_loss)

# 评估训练得到的模型
def evaluate(sess, X, Y):
	predicted = tf.cast(inference(X) > 0.5, tf.float32)
	print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))
	return

# 启动会话，运行训练闭环
with tf.Session() as sess:
	tf.initialize_all_variables().run()

	X, Y = inputs()

	total_loss = loss(X, Y)
	train_op = train(total_loss)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess, coord)

	# 实际的训练迭代次数
	training_steps = 1000
	for step in range(training_steps):
		sess.run([train_op])
		# 处于调试和学习的目的，查看损失在训练过程中递减的情况
		if step % 10 == 0:
			print ("loss: ", sess.run([total_loss]))

	evaluate(sess, X, Y)

	coord.request_stop()
	coord.join(threads)
	sess.close()

