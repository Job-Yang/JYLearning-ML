#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'CNN'

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from itertools import groupby
from collections import defaultdict


sess = tf.InteractiveSession()

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# 将文件名分解成其品种和相应的文件名。 通过获取目录名称找到该品种
image_filename_with_breed = map(lambda filename:(filename.split("/")[2], filename), image_filenames)

# 依据品种（上述返回的元组的第0个分量）对图像进行分组
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
	# 枚举每个品种的图像，并将大致20%的图像划入测试集
	for i, breed_images in enumerate(breed_images):
		if  i % 5 == 0:
			testing_dataset[dog_breed].append(breed_image1[1])
		else:
			training_dataset[dog_breed].append(breed_image[1])

	# 检查每个品种的测试图像是否至少有全部的18%
	breed_training_count = len(training_dataset[dog_breed])
	breed_testing_count = len(testing_dataset[dog_breed])

	assert round(breed_testing_count / (breed_training_count + breed_testing_count), 2) > 0.18, "Not enough testing images"



def write_records_file(dataset, record_location):
	"""
	用dataset中的图像填充一个TFRecord文件，并将其类别包含进来

	参数
	------------
	dataset:dict(list)
		这个字典的键对应于其值中文件名列表对应的标签

	record_location
		存储TFRecord输出路径
	"""

	writer = None

	# 枚举dataset，因为当前索引用于对文件进行划分，每隔100幅图像，许梿样本的信息就被写入到一个新的TFRecord文件中，以加快写操作的进程
	current_index = 0

	for breed, images_filenames in dataset.items():
		for images_filename in images_filenames:
			if current_index % 100 == 0
			if writer:
				writer.close()

				record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,
					current_index=current_index)

				writer = tf.python_io.TFRecordWriter(record_filename)
				current_index += 1

				image_file = tf.read_file(image_filename)

				# 在ImageNet的狗的图像中， 有少量无法被TensorFlow识别为Jpeg的图像，利用try/catch可将这些图像忽略
				try:
					image = tf.image.decode_jpeg(image_file)
				except:
					print(images_filename)
					continue

				# 转换为灰度图可减少处理的计算量和内存占用，单着并不是必须的
				grayscale_image = tf.image.rgb_to_grayscale(image)
				resized_image = tf.image.resize_images(grayscale_image, 250, 151)

				# 这里之所以使用tf.cast，是因为虽然尺寸更改后的图像数据类型是浮点型，单RGB值尚未转换到[0, 1)区间内
				image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

				# 将标签按字符串存储较为高效，退件的做法是将其转换为整数索引或者独热编码的秩为1张量
				image_label = breed.encode("utf-8")

			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
				'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
			}))

			writer.write(example.SerializeToString())
    		writer.close()

write_records_file(testing_dataset, "./output/testing-images/testing-image")
write_records_file(training_dataset, "./output/training-images/training-image")

if __name__ == '__main__':

