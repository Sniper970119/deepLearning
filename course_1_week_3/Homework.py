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
import matplotlib.pyplot as plt
from course_1_week_3.testCases import *
import sklearn.linear_model
from course_1_week_3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# 加载数据
X, Y = load_planar_dataset()
print('X的维度:', X.shape)
print('Y的维度:', Y.shape)
plt.title('show point')
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()

# 使用逻辑回归处理
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X.T, Y.T)
# planar_util里的方法   绘制决策边界
plot_decision_boundary(lambda x: clf.predict(x), X, Y[0, :])
plt.title('Logistic Regression')
LR_predictions = clf.predict(X.T)
print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                               np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      "% " + "(正确标记的数据点所占的百分比)")
plt.show()

# 使用之前写的逻辑回归思维的神经网络进行计算
from course_1_week_2.Logistic import model
d = model(X.T, Y.T, X.T, Y.T)


# 神经网络

# (5,3)    (2,3)
X_asses, Y_asses = layer_sizes_test_case()


def layer_sizes(X, Y):
    """
    返回神经网络各层的个数
    :param X: 输入数据集
    :param Y: 标签
    :return:
    """
    # 输入层个数
    n_x = X.shape[0]
    # 隐藏层个数
    n_h = 4
    # 输出层个数
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def init_parameters(n_x, n_h, n_y):
    """
    初始化参数
    :param n_x: 输入层个数
    :param n_h: 隐藏层个数
    :param n_y: 输出层个数
    :return:
    """
    # 初始化一个种子，以保证大家的结果一样
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    return parameters


def forward_propagation(X, parameters):
    """
    前向传播
    :param X:
    :param parameters:
    :return:
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    # 这里别忘了使用sigmoid进行激活
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    assert (A2.shape == (1, X.shape[1]))
    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    计算代价
    :param A2: 计算结果
    :param Y: 标签
    :param parameters: 网络参数
    :return:
    """
    m = Y.shape[1]
    if m == 0:
        print()
    # ??? 干嘛的
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # 计算成本
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    反向传播
    :param parameters:
    :param cache:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    更新权重
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = init_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if print_cost:
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))
    return parameters


def predict(parameters, X):
    """
    预测函数
    :param parameters:
    :param X:
    :return:
    """
    A2, cache = forward_propagation(X, parameters)
    # 四舍五入
    predictions = np.round(A2)

    return predictions


parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=10000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
plt.show()
"""
隐藏层的节点数量： 1  ，准确率: 67.25 %
隐藏层的节点数量： 2  ，准确率: 67.0 %
隐藏层的节点数量： 3  ，准确率: 90.75 %
隐藏层的节点数量： 4  ，准确率: 90.5 %
隐藏层的节点数量： 5  ，准确率: 91.0 %
隐藏层的节点数量： 20  ，准确率: 91.25 %
隐藏层的节点数量： 50  ，准确率: 90.75 %
"""