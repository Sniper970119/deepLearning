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
from course_5_week_1 import rnn_utils


def rnn_cell_forward(xt, a_prev, parameters):
    """
    rnn cell前向传播
    :param xt:
    :param a_prev:
    :param parameters:
    :return:
    """
    # 从parameters”获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # 使用上面的公式计算当前单元的输出
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    # 保存反向传播需要的值
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(X, a0, parameters):
    """
    rnn 前向传播
    :param X:
    :param a0:
    :param parameters:
    :return:
    """
    caches = []

    # 获取x和Wya的维度信息
    n_x, m, T_x = X.shape
    n_y, n_a = parameters['Wya'].shape

    # 使用0来初始化a和y
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # 初始化next
    a_next = a0

    # 遍历所有的序列
    for i in range(T_x):
        # cell前向传播
        a_next, yt_pred, cache = rnn_cell_forward(X[:, :, i], a_next, parameters)
        # 使用 a 来保存“next”隐藏状态（第 i）个位置。
        a[:, :, i] = a_next
        # 保存预测值
        y_pred[:, :, i] = yt_pred

        caches.append(cache)
    caches = (caches, X)

    return a, y_pred, caches

