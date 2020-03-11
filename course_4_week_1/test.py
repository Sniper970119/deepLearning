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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    P = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)
    return Z3


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


def compute_cost(Z3, Y):
    """
    计算代价
    :param Z3:
    :param Y:
    :return:
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


my_image1 = "./test_imgs/10.png"
fileName1 = my_image1
image1 = mpimg.imread(fileName1)
plt.imshow(image1)
# plt.show()
my_image1 = image1.reshape(1, 64, 64, 3)

gragh = tf.get_default_graph()
saver = tf.train.import_meta_graph('./model/my_model-5.meta')
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]
X, Y = create_placeholders(64, 64, 3, 6)
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    saver.restore(session, tf.train.latest_checkpoint('.\model'))
    parameters = session.run(parameters)
    a = session.run(Z3, feed_dict={X: my_image1})
    res = a.tolist()[0]
    print(res)
    print('预测结果：' +str(res.index(max(res))))

    # predict_op = tf.argmax(Z3, 1)
    # corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(corrent_prediction, 'float'))
    # print("corrent_prediction accuracy= " + str(accuracy))
    # train_accuracy = accuracy.eval({X: my_image1, Y: np.array(['0', '0', '0', '0', '0', '0']).T})
    # print("训练集准确度：" + str(train_accuracy))
    pass
