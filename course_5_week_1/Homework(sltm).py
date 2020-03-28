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


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    lstm cell 前向传播
    :param xt:
    :param a_prev:
    :param c_prev:
    :param parameters:
    :return:
    """
    # 从“parameters”中获取相关值
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取x_t、Wy维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 链接a_prev与xt
    contact = np.zeros([n_a + n_x, m])
    contact[:n_a, :] = a_prev
    contact[n_a:, :] = xt

    # 根据公式计算
    # 遗忘门，公式1
    ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)

    # 更新门，公式2
    ut = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)

    # 更新单元，公式3
    cct = np.tanh(np.dot(Wc, contact) + bc)

    # 更新单元，公式4
    c_next = ft * c_prev + ut * cct
    # 输出门，公式5
    ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)

    # 输出门，公式6
    # a_next = np.multiply(ot, np.tan(c_next))
    a_next = ot * np.tanh(c_next)
    # 3.计算LSTM单元的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    # 保存包含了反向传播所需要的参数
    cache = (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    lstm前向传播
    :param x:
    :param a0:
    :param parameters:
    :return:
    """
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    # 使用0来初始化a,c,y
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    # 初始化anext  cnext
    a_next = a0
    c_next = np.zeros([n_a, m])

    # 遍历所有的时间步
    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

        # 保存新的下一个隐藏状态到变量a中
        a[:, :, t] = a_next

        # 保存预测值到变量y中
        y[:, :, t] = yt_pred

        # 保存下一个单元状态到变量c中
        c[:, :, t] = c_next

        # 把cache添加到caches中
        caches.append(cache)

    # 保存反向传播需要的参数
    caches = (caches, x)

    return a, y, c, caches

