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

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from course_2_week_3 import tf_utils
import time

np.random.seed(1)

# # 计算损失
# # 定义y_hat为36
# y_hat = tf.constant(36, name='y_hat')
# # 定义y为39
# y = tf.constant(39, name='y')
#
# # 定义损失函数
# loss = tf.Variable((y - y_hat) ** 2, name='loss')
#
# # 初始化参数
# init = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# output 9


# 使用placeholder占位符
# x = tf.placeholder(tf.int64, name="x")
# with tf.Session() as session:
#     print(session.run(2 * x, feed_dict={x: 3}))


# output 6


# 计算线性函数
# def linear_function():
#     """
#     实现一个计算线性函数的功能
#     :return:
#     """
#     X = np.random.randn(3, 1)
#     W = np.random.randn(4, 3)
#     b = np.random.randn(4, 1)
#
#     Y = tf.add(tf.matmul(W, X), b)
#     # 这俩是一样的，因为如果tf会重载运算符
#     Y = tf.matmul(W, X) + b
#
#     with tf.Session() as session:
#         result = session.run(Y)
#         session.close()
#     return result
#
#
# print("result = " + str(linear_function()))

"""
output
result = [[-2.15657382]
 [ 2.95891446]
 [-1.08926781]
 [-0.84538042]]
"""


# sigmoid
# def sigmoid(z):
#     """
#     sigmoid
#     :param z:
#     :return:
#     """
#     x = tf.placeholder(tf.float32, name='x')
#     sigmoid = tf.sigmoid(x)
#     with tf.Session() as session:
#         result = session.run(sigmoid, feed_dict={x: z})
#     return result
#
#
# print("sigmoid(0) = " + str(sigmoid(0)))
# print("sigmoid(12) = " + str(sigmoid(12)))

# output
# sigmoid(0) = 0.5
# sigmoid(12) = 0.9999938


# one-hot矩阵
# def one_hot_matrix(labels, C):
#     """
#     生成onehot矩阵
#     :param labels:标签向量
#     :param C:分类数
#     :return:
#     """
#     C = tf.constant(C, name='C')
#     one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
#     with tf.Session() as session:
#         result = session.run(one_hot_matrix)
#         session.close()
#     return result
#
#
# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = one_hot_matrix(labels, C=4)
# print(str(one_hot))

"""
output 
[[0. 0. 0. 1. 0. 0.]
 [1. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 0.]]
"""


