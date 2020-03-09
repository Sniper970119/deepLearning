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
import tensorflow as tf
from course_2_week_3 import tf_utils
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

# plt.imshow(X_train_orig[11])
# print("Y = " + str(np.squeeze(Y_train_orig[:, 11])))
# plt.show()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# 转换onehot矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)


def create_placeholders(n_x, n_y):
    """
    为TensorFlow会话创建占位符
    :param n_x:
    :param n_y:
    :return:
    """
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
    return X, Y


# 初始化参数
def initialize_parameters():
    """
    初始化参数
    :return:
    """

    W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())

    W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())

    W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }


def forward_propagation(X, parameters):
    """
    前向传播
    :param X:
    :param parameters:
    :return:
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(W3, A2) + b3

    return Z3


def compute_cost(Z3, Y):
    """
    计算代价
    :param Z3:
    :param Y:
    :return:
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1300, minibatch_size=32,
          print_cost=True,
          is_plot=True):
    """
    神经网络模型
    LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param learning_rate:
    :param num_epochs:
    :param minbatch_size:
    :param print_cost:
    :param is_plot:
    :return:
    """
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # 创建placeholder
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前行传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播,Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()

    # 初始化所有的变量
    init = tf.global_variables_initializer()

    # 开始会话并计算
    with tf.Session() as session:
        session.run(init)

        # 循环
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minbatches = int(m / minibatch_size)
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                # 开始运行
                _, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                # 计算代价
                epoch_cost = epoch_cost + minibatch_cost / num_minbatches

            # 记录代价
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

            saver.save(session, './model/my_model', global_step=100)

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存参数
        parameters = session.run(parameters)

        # 计算预测结果
        current_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(current_prediction, 'float'))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


# 开始时间
start_time = time.clock()
# 开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
# 结束时间
end_time = time.clock()
# 计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")

"""
epoch = 0    epoch_cost = 1.8670177423592766
epoch = 100    epoch_cost = 1.0305518833073701
epoch = 200    epoch_cost = 0.7247187675851763
epoch = 300    epoch_cost = 0.49531238006822986
epoch = 400    epoch_cost = 0.33661927914980694
epoch = 500    epoch_cost = 0.2284338934855028
epoch = 600    epoch_cost = 0.17487916463252268
epoch = 700    epoch_cost = 0.11813201111826033
epoch = 800    epoch_cost = 0.08827614908417065
epoch = 900    epoch_cost = 0.06095291397562531
epoch = 1000    epoch_cost = 0.04344996804315032
epoch = 1100    epoch_cost = 0.11286204381648339
epoch = 1200    epoch_cost = 0.04022079737236102
epoch = 1300    epoch_cost = 0.03377958937463435
epoch = 1400    epoch_cost = 0.07520042944022201
训练集的准确率： 0.9962963
测试集的准确率: 0.825
CPU的执行时间 = 433.66559440000003 秒
"""
