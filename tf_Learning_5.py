#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 显示的创建一个Graph对象
graph = tf.Graph()

with graph.as_default():

	with tf.name_scope("Variables"):
		# 追踪数据流图运行次数的 Variables对象
		global_step = tf.Variables(0, dtype=tf.init32, trainable=False, name="global_step")
		# 追踪所以输出随着时间的累加和的 Variables对象
		total_output = tf.Variables(0.0, dtype=tf.float32, trainable=False, name="total_output")

	#主要的变换Op
	with tf.name_scope("transformation"):

		# 独立的输入层
		with tf.name_scope("input"):
			# 创建可接受一个向量的占位符
			a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

		# 独立的中间层
		with tf.name_scope("intermediate_layer"):
			b = tf.redece_prod(a, name="product_b")
			c = tf.redece_sum(a, name="sum_c")

		# 独立的输出层
		with tf.name_scope("output"):
			output = tf.add(b, c, name="output")

	with tf.name_scope("update"):
		# 用最新的输出更新 Variable对象的 total_output
		update_total = total_output.assign_add(output)

		# 将前面的 Variable对象的global_step增加1，只要数据流图运行，该操作便需要进行
		increment_step = global_step.assign_add(1)

	with tf.name_scope("summaries"):
		avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")

		# 为输出节点创建汇总数据
		tf.scalar_summary(b'Output', output, name="output_summary")
		tf.scalar_summary(b'Sum of outputs over time', update_total, name="total_summary")
		tf.scalar_summary(b'Average of outputs over time', avg, name="average_summary")

	# 全局Variable对象和Op
	with tf.name_scope("global_ops"):
		# 初始化Op
		init = tf.initialize_all_variables()
		# 将所有汇总数据合并到一个Op中
		merged_summaries = tf.merge_all_summaries()

	# 用明确创建的 Graph对象启动一个会话
	sess = tf.Session(graph=graph)

	# 开启一个SummaryWriter对象，保存汇总数据
	writer = tf.train.SummaryWriter('./improved_gragh', graph)

	# 初始化Variable对象
	sess.run(init)

def run_gragh(input_tensor):
	"""
	辅助工具：用给定的输入张量运行数据流图，并保存汇总数据
	"""
	feed_dict = {a: input_tensor}
	_, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
	writer.add_summary(summary, global_step=step)


	# 用不同的输入运行该数据流图
	run_gragh([2,8])
	run_gragh([3,1,3,3])
	run_gragh([8])
	run_gragh([1,2,3])
	run_gragh([11,4])
	run_gragh([4,1])
	run_gragh([7,3,1])
	run_gragh([6,3])
	run_gragh([0,2])
	run_gragh([4,5,6])

	# 将汇总数据写入磁盘
	writer.flush()

	# 关闭SummaryWriter对象
	writer.close()

	# 关闭Session对象
	sess.close()


	





