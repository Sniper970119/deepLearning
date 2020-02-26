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

    #    进行xw+b(按顺序)的各个参数  # 进行激活的Z
    cache = linear_cache, activation_cache
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
    # caches 中为 每一次传播过程的（进行xw+b的各个参数，进行激活的Z）
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
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

    # 矩阵转数组(单个数字)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    """
    反向传播
    :param dZ:
    :param cache:
    :return:
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation='relu'):
    """
    反向传播的激活
    :param dA:
    :param cache:
    :param activation:
    :return:
    """
    assert activation in ['sigmoid', 'relu']
    linear_cache, activation_cache = cache
    if activation is "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation is "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    多层网络的反向传播
    :param AL:
    :param Y:
    :param caches:
    :return:
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 2)], current_cache, 'relu')
        grads['dA' + str(l + 1)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    更新参数
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2
    for i in range(L):
        parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - learning_rate * grads['dW' + str(i + 1)]
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * grads['db' + str(i + 1)]

    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    """
    实现一个两层的神经网络
    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iteration:
    :param print_cost:
    :param isPlot:
    :return:
    """
    np.random.seed(1)

    grads = {}
    costs = []
    # n_x, n_h, n_y = layers_dims

    # parameters = init_two_layer_parameters(n_x, n_h, n_y)
    parameters = init_deep_layers_parameters(layers_dims)

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    for i in range(num_iterations):
        # 前向传播
        # A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        # A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        # 反向传播
        # dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        # dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        # dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        grads = L_model_backward(AL, Y, caches)

        # 向后传播完成后的数据保存到grads
        # grads["dW1"] = dW1
        # grads["db1"] = db1
        # grads["dW2"] = dW2
        # grads["db2"] = db2

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        # W1 = parameters["W1"]
        # b1 = parameters["b1"]
        # W2 = parameters["W2"]
        # b2 = parameters["b2"]

        # 打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 是否打印成本值
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))
    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    # 返回parameters
    return parameters


def predict(X, y, parameters):
    """
    预测结果
    :param X:
    :param y:
    :param parameters:
    :return:
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


# 预处理数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# RGB标准化
train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

parameters = two_layer_model(train_x, train_set_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True,
                             isPlot=True)
"""
cost： 0.05336140348560556
cost： 0.048554785628770185
train准确度为: 1.0
test准确度为: 0.72

cost为： 0.14781357997051983
cost： 0.12935258942424563
train准确度为: 1.0
test准确度为: 0.74
"""

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


def print_mislabeled_images(classes, X, y, p):
    """
	绘制预测和实际不同的图像。
	    X - 数据集
	    y - 实际的标签
	    p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
    plt.show()


print_mislabeled_images(classes, test_x, test_y, pred_test)
