# -*- coding:utf-8 -*-

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃　　　　　　　    ┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛
"""
import time
import cv2
import os
import numpy as np

from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import keras

from course_4_week_4 import fr_utils
from course_4_week_4.inception_blocks_v2 import *

# 获取模型
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

# 打印模型的总参数数量
print("参数数量：" + str(FRmodel.count_params()))


# def triplet_loss(y_true, y_pred, alpha=0.2):
#     """
#     三元组损失函数
#     :param y_true:
#     :param y_pred:
#     :param alpha:
#     :return:
#     """
#     # 获取anchor、positive、negative图像编码
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#     # 计算anchor和positive的编码距离
#     """
#     reduce_sum: 计算张量的和
#     square: 是对a里的每一个元素求平方
#     subtract: 返回x-y
#     """
#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
#     # 计算anchor和negative的编码距离
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
#     # 计算距离
#     basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#     # 计算代价
#     loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
#
#     return loss
#
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
#     loss = triplet_loss(y_true, y_pred)
#
#     print("loss = " + str(loss.eval()))