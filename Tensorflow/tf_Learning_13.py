#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'图像相关操作'

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()


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


# 读取图片并保存为TFRecord
def write_tfrecord(image, file_name):
	# 将张量转换为字节型，这个操作会加载整个图像文件
	image_label = b'\x01' #假设标签数据位于一个one_hot编码表示中（00000001）
	image_loaded = sess.run(image)
	image_bytes = image_loaded.tobytes()

	# 导出TFRecord
	writer = tf.python_io.TFRecordWriter(file_name)

	# 在样本文件中不保存图像的宽度，高度或通道数，以便节省空间
	example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
			'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
		}))

	# 将样本保存到一个文本文件tfrecord中
	writer.write(example.SerializeToString())
	writer.close()


# 加载 TFRecord文件
def load_tfrecord(file_name):
	tf_record_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(file_name))

	# 注意这个不同的记录读取器，他的设计意图是能够使用可能包含多个样本的 TFRecord文件
	tf_record_reader = tf.TFRecordReader()
	_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

	# 标签和图像都按字节存储，也可以按int64或float64存储于序列化的tf.Example protobuf文件中
	tf_record_features = tf.parse_single_example(tf_record_serialized, features={
			'label': tf.FixedLenFeature([], tf.string),
			'image': tf.FixedLenFeature([], tf.string),
		})

	# 使用tf.uint8类型，因为所有的通道信息都处于[0, 255]
	tf_record_image = tf.decode_raw(tf_record_features['image'], tf.uint8)
	image_height, image_width, image_channels = tf_record_image.shape

	# 调整图像尺寸，使其也保存的图像类似，但这并非是必须的
	# 用实值表示图像的高度，宽度和通道，因为必须对输入的形状进行调整
	tf_record_image = tf.reshape(tf_record_image, [image_height, image_width, image_channels])

	tf_record_label = tf.cast(tf_record_features['label'], tf.string)
	return tf_record_label, tf_record_image

if __name__ == '__main__':

	# 复用之前的图像
	image = read_image("./images/dog.jpg")
	print(image)
	file_name = "./output/training-image.tfrecord"
	write_tfrecord(image, file_name)
	tf_record_label, tf_record_image = load_tfrecord(file_name)

	sess.run(tf.equal(image, tf_record_image))

	# 检查标签是否一致
	sess.run(tf_record_label)

