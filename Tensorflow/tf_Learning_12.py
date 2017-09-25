#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'深入探讨卷积核'

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


kernel1 = tf.constant([
	[
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
	],
	[
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[  8., 0., 0.], [ 0.,  8., 0.], [ 0., 0.,  8.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
	],
	[
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
	]
])


kernel2 = tf.constant([
	[
		[[  0., 0., 0.], [ 0.,  0., 0.], [ 0., 0.,  0.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[  0., 0., 0.], [ 0.,  0., 0.], [ 0., 0.,  0.]]
	],
	[
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[  5., 0., 0.], [ 0.,  5., 0.], [ 0., 0.,  5.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
	],
	[
		[[  0., 0., 0.], [ 0.,  0., 0.], [ 0., 0.,  0.]],
		[[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
		[[  0., 0., 0.], [ 0.,  0., 0.], [ 0., 0.,  0.]]
	]
])

def read_image(name):

	# match_filenames_once可以接收一个正则表达式，用于批量匹配多张图片，然后生成队列
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(name))

	# 读取所需的整个图像文件，因为它们是JPEG，如果图像太大了，可以提前分割成较小的文件或使用固定读取器来分割文件。
	image_reader = tf.WholeFileReader()

	# 从队列中读取整个文件，元组中的第一个返回值是filename，我们忽略。
	image_name, image_file= image_reader.read(filename_queue)

	# 将图像解码为JPEG文件，这样就可以将它变成一个Tensor，我们可以这样做然后用于训练。
	image = tf.image.decode_jpeg(image_file)
	return image


def image_convolution(image_batch, kernel):
	conv2d = tf.nn.conv2d(image_batch, kernel, [1, 1, 1, 1], padding="SAME")
	activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d), 255))
	print(activation_map[0])


with tf.Session() as sess:
	tf.local_variables_initializer().run()

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	image = read_image("./images/dog.jpg")
	image_batch = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32, saturate=False)
	image_convolution(image_batch, kernel1)
	image_convolution(image_batch, kernel2)


	coord.request_stop()
	coord.join(threads)





