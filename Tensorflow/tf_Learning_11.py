#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'卷积神经网络基础'

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()


# 图片结构相关
def image_shape():
	image_batch = tf.constant([
		[  # 第一张图
			[[0, 255, 0], [0, 255, 0], [0, 255, 0]],
			[[0, 255, 0], [0, 255, 0], [0, 255, 0]]
		],
		[  # 第二张图
			[[0, 0, 255], [0, 0, 255], [0, 0, 255]],
			[[0, 0, 255], [0, 0, 255], [0, 0, 255]]
		]
	])
	shape = image_batch.get_shape()
	print(shape)
	image_pixels = image_batch[0][0][0]
	print(sess.run(image_pixels))
	print("-------------------------")


# 输入和卷积核
def convolution():
	input_batch = tf.constant([
		[  # 第一个输入
			[[0.0], [1.0]],
			[[2.0], [3.0]]
		],
		[  # 第二个输入
			[[2.0], [4.0]],
			[[6.0], [8.0]]
		]
	])

	# 卷积核
	kernel = tf.constant([
			[
				[[1.0, 2.0]]
			]
		])
	conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding="SAME")
	print(sess.run(conv2d))

	lower_right_image_pixel = sess.run(input_batch)[0][1][1]
	lower_right_kernel_pixel = sess.run(conv2d)[0][1][1]
	print(lower_right_image_pixel)
	print(lower_right_kernel_pixel)
	print("-------------------------")


# 跨度
def strides():
	input_batch = tf.constant([
		[  # 第一个输入 (6x6x1)
			[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
			[[0.1], [1.1], [2.1], [3.1], [4.1], [5.1]],
			[[0.2], [1.2], [2.2], [3.2], [4.2], [5.2]],
			[[0.3], [1.3], [2.3], [3.3], [4.3], [5.3]],
			[[0.4], [1.4], [2.4], [3.4], [4.4], [5.4]],
			[[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]],
		],
	])

	# 卷积核 (3x3x1)
	kernel = tf.constant([
		[[[0.0]], [[0.5]], [[0.0]]],
		[[[0.0]], [[1.0]], [[0.0]]],
		[[[0.0]], [[0.5]], [[0.0]]]
	])

	# 注意：strides参数的尺寸变化
	conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 3, 3, 1], padding='SAME')
	print(sess.run(conv2d))
	print("-------------------------")

if __name__ == '__main__':
	image_shape()
	convolution()
	strides()




