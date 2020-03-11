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

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

from course_4_week_1 import cnn_utils

np.random.seed(1)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
# index = 6
# plt.imshow(X_train_orig[index])
# print("y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()

# 数据标准化
X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T

conv_layers = {}


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    创建占位符
    :param n_H0:
    :param n_W0:
    :param n_C0:
    :param n_y:
    :return:
    """
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


def initialize_parameters():
    """
    初始化权值矩阵，这里我们把权值矩阵硬编码：
    :return:
    """

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    """
    实现前向传播
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    :param X:
    :param parameters:
    :return:
    """
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A1 = tf.nn.relu(Z1)
    # Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    # Conv2d : 步伐：1，填充方式：“SAME”
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A2 = tf.nn.relu(Z2)
    # Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    # 全连接层（FC）：使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3


def compute_cost(Z3, Y):
    """
    计算代价
    :param Z3:
    :param Y:
    :return:
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True, isPlot=True):
    """
    创建模型
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param learning_rate:
    :param num_epochs:
    :param minibatch_size:
    :param print_cost:
    :param isPlot:
    :return:
    """
    tf.get_default_graph()
    seed = 3
    tf.set_random_seed(1)
    # m个图片  h*w大小  c个卷积层
    m, n_H0, n_W0, n_C0 = X_train.shape
    # 有多少种结果
    n_y = Y_train.shape[1]
    costs = []
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算成本
    cost = compute_cost(Z3, Y)
    # 反向传播
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 保存数据
    saver = tf.train.Saver()
    # 全局初始化变量
    init = tf.global_variables_initializer()
    # 开始运行
    with tf.Session() as session:
        # 初始化参数
        session.run(init)
        # 遍历数据
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _, temp_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # 保存参数
            saver.save(session, './model/my_model', global_step=5)

            # 打印成本
            if print_cost:
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            # 记录成本
            costs.append(minibatch_cost)

        # 绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 开始预测
        predict_op = tf.argmax(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, 'float',name='accuracy'))
        print("corrent_prediction accuracy= " + str(accuracy))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuracy))

        return train_accuracy, test_accuracy, parameters


_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)
"""
当前是第 0 代，成本值为：1.9079020842909813
当前是第 5 代，成本值为：1.8338764905929565
当前是第 10 代，成本值为：1.3530711755156517
当前是第 15 代，成本值为：1.0513618439435959
当前是第 20 代，成本值为：0.7954236306250095
当前是第 25 代，成本值为：0.6776404492557049
当前是第 30 代，成本值为：0.6062654200941324
当前是第 35 代，成本值为：0.5560709461569786
当前是第 40 代，成本值为：0.50390131957829
当前是第 45 代，成本值为：0.45013799145817757
当前是第 50 代，成本值为：0.43954286724328995
当前是第 55 代，成本值为：0.3888242533430457
当前是第 60 代，成本值为：0.37428596522659063
当前是第 65 代，成本值为：0.35892391856759787
当前是第 70 代，成本值为：0.33011145144701004
当前是第 75 代，成本值为：0.3231000145897269
当前是第 80 代，成本值为：0.31615445110946894
当前是第 85 代，成本值为：0.3158330311998725
当前是第 90 代，成本值为：0.26991996727883816
当前是第 95 代，成本值为：0.2658209102228284
当前是第 100 代，成本值为：0.2894991096109152
当前是第 105 代，成本值为：0.24726102221757174
当前是第 110 代，成本值为：0.2515424760058522
当前是第 115 代，成本值为：0.3262146320194006
当前是第 120 代，成本值为：0.23388438019901514
当前是第 125 代，成本值为：0.23200181871652603
当前是第 130 代，成本值为：0.2730541517958045
当前是第 135 代，成本值为：0.22053561871871352
当前是第 140 代，成本值为：0.25617739744484425
当前是第 145 代，成本值为：0.19133262312971056
corrent_prediction accuracy= Tensor("Mean_1:0", shape=(), dtype=float32)
训练集准确度：0.93796295
测试集准确度：0.81666666
"""