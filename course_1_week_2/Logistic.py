# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
from course_1_week_2.lr_utils import load_dataset

# 训练集图片， 训练集标签， 测试集图片，测试集标签，分类标签文本描述
# train_set_x_pic, train_set_y, test_set_x_pic, test_set_y, classes = load_dataset()
#
# # 查看一下图片
# plt.imshow(train_set_x_pic[0])
# plt.show()
#
# print('训练集大小:', train_set_x_pic.shape[0])
# print('测试集大小:', test_set_x_pic.shape[0])
# print('图片的大小:', train_set_x_pic.shape[1:])
#
# train_set_x_flatten = train_set_x_pic.reshape(train_set_x_pic.shape[0], -1).T
# test_set_x_flatten = test_set_x_pic.reshape(test_set_x_pic.shape[0], -1).T
# # 标准化RGB颜色，因此原值范围较大，标准化到0~1之间
# train_set_x = train_set_x_flatten / 255
# test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def init_weight(dim):
    """
    初始化维度
    :param dim:
    :return:
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    """
    前后传播以及成本计算
    :param w:
    :param b:
    :param X:
    :param Y:
    :return:
    """
    n = X.shape[1]
    # 计算代价
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.mean(-Y * np.log(A) - (1 - Y) * np.log(1 - A))
    # 反向传播
    dw = (1 / n) * (X @ (A - Y).T)
    db = (1 / n) * np.sum(A - Y)
    return dw, db, cost


def optimize(w, b, X, Y, iterations_times, learning_rate, print_cost=False):
    """
    运行梯度下降来优化w和b
    :param w:
    :param b:
    :param X:
    :param Y:
    :param iterations_times:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    costs = []
    for i in range(iterations_times):
        wb, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * wb
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print('迭代次数：%i，误差值：%f' % (i, cost))

    return w, b, costs


def predict(w, b, X):
    """
    预测
    :param w:
    :param b:
    :param X:
    :return:
    """
    n = X.shape[1]
    Y_prediction = np.zeros((1, n))

    A = sigmoid((w.T @ X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.01, print_cost=True):
    """
    总体模型
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    w, b = init_weight(X_train.shape[0])
    wb, db, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_test = predict(wb, db, X_test)

    print('训练集准确率:', (1 - (np.mean(np.abs(Y_prediction_test - Y_test)))) * 100, '%')
    return wb, db, costs


# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
