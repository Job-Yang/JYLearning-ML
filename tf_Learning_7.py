#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Job Yang'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# 模型定义的相关代码

# 创建一个Saver对象
saver = tf.train.Saver()


# 启动会话，运行训练闭环
with tf.Session() as sess:

	inital_step = 0

	# 验证之前是否已经保存了检查点文件
	ckpt = tf.trian.get_checkpoint_state(os.path.dirname(__file__))
	if ckpt and ckpt.model_checkpoint_path: 
		# 从检查点恢复模型参数
		saver.restore(sess, ckpt.model_checkpoint_path)
		inital_step = int(ckpt.model_checkpoint_path.rslist('-', 1)[1])

	# 实际的训练迭代次数
	for step in range(inital_step, training_steps):
		sess.run([train_op])

		# 处于调试和学习的目的，查看损失在训练过程中递减的情况
		if step % 10 = 0:
			print "loss:",sess.run([total_loss])

		# 保存训练检查点
		if step % 1000 = 0:
			saver.save(sess, 'my_model', global_step=step)


	saver.save(sess, 'my_model', global_step=training_steps)

	evaluate(sess, X, Y)

	coord.request_stop()
	coord.join(threads)
	sess.close()
