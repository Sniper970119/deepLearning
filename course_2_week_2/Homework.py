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
import math

from course_2_week_2 import opt_utils
from course_2_week_2 import testCases

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载数据
train_X, train_Y = opt_utils.load_dataset()
plt.show()


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    更新数值
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2  # 神经网络的层数

    # 更新每个参数
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    创建随机batch列表
    :param X:
    :param Y:
    :param mini_batch_size:
    :param seed:
    :return:
    """

    np.random.seed(seed)  # 指定随机种子
    m = X.shape[1]
    mini_batches = []

    # 第一步：打乱顺序
    permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:, permutation]  # 将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 第二步，分割
    # 计算要分割多少份数据集
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 处理没法被batch大小处理的部分数据
    if m % mini_batch_size != 0:
        # 获取最后剩余的部分
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_velocity(parameters):
    """
    初始化速度
    :param parameters:
    :return:
    """
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        #  zeros_like是初始化一个0矩阵，后台调用实际上还是zeros，只不过这里可以传矩阵而不是shape
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return v


def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):
    """
    使用动量更新参数(momentum算法）
    :param parameters:
    :param grads:
    :param v:
    :param beta:
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2
    for l in range(L):
        # 计算速度
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


def initialize_adam(parameters):
    """
    初始化Adam算法使用的v和s
    :param parameters:
    :return:
    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return (v, s)


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    更新参数
    :param parameters:
    :param grads:
    :param v:
    :param s:
    :param t:
    :param learning_rate:
    :param beta1:
    :param beta2:
    :param epsilon:
    :return:
    """
    L = len(parameters) // 2
    v_corrected = {}  # 偏差修正后的值
    s_corrected = {}  # 偏差修正后的值

    for l in range(L):
        # 计算v
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # 修正误差
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # 计算s
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # 修正误差
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return (parameters, v, s)


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    定义神经网络模型
    :param X:
    :param Y:
    :param layers_dims:
    :param optimizer:
    :param learning_rate:
    :param mini_batch_size:
    :param beta:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param num_epochs:
    :param print_cost:
    :param is_plot:
    :return:
    """
    L = len(layers_dims)
    costs = []
    t = 0  # 每学习完一个minibatch就增加1
    seed = 10  # 随机种子

    # 初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 使用动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用Adam优化
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # 前向传播
            A3, cache = opt_utils.forward_propagation(minibatch_X, parameters)
            # 计算误差
            cost = opt_utils.compute_cost(A3, minibatch_Y)
            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)
            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
        # 记录误差值
        if i % 100 == 0:
            costs.append(cost)
            # 是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters


# 使用普通的梯度下降
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)
#
# # 预测
# preditions = opt_utils.predict(train_X, train_Y, parameters)
#
# # 绘制分类图
# plt.title("Model with Momentum optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)


#使用动量的梯度下降
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, beta=0.9,optimizer="momentum",is_plot=True)
#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)

#绘制分类图
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)


# Adam
layers_dims = [train_X.shape[0], 5, 2, 1]
# 使用Adam优化的梯度下降
parameters = model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)
# 预测
preditions = opt_utils.predict(train_X, train_Y, parameters)

# 绘制分类图
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
