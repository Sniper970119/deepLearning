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
from course_1_week_4 import testCases
from course_1_week_4.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from course_1_week_4 import lr_utils

# 指定随机数种子
np.random.seed(1)


def init_two_layer_parameters(n_x, n_h, n_y):
    """
    随机初始化参数（一个两层的神经网络）
    :param n_x: 输入层结点个数
    :param n_h: 隐藏层结点个数
    :param n_y: 输出层结点个数
    :return:
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters


def init_deep_layers_parameters(layers_dim):
    """
    初始化多层神经网络参数
    :param layers_dim: 每层节点数的列表
    :return:
    """
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layers_dim)):
        parameters['W' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layers_dim[i], 1))

    return parameters


def linear_forward(A, W, b):
    """
    前向传播（只算wx+b，不激活）
    :param A:
    :param W:
    :param b:
    :return:
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    前向传播的激活
    :param A_prev:上一层的激活值
    :param W:
    :param b:
    :param activation: 激活函数
    :return:
    """
    assert activation in ['sigmoid', 'relu']
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation is 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    多层模型的传播
    :param X:
    :param parameters:
    :return:
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A, cache = linear_activation_forward(A, parameters['W' + str(i)], parameters['b' + str(i)], 'relu')
        caches.append(cache)
    AL, cahce = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cahce)

    return AL, caches


def compute_cost(AL, Y):
    """
    计算成本
    :param AL:
    :param Y:
    :return:
    """
    m = Y.shape[1]

