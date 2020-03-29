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
import random
import time

from course_5_week_1 import cllm_utils as utils

# 获取名称
data = open("dinos.txt", "r").read()
# 转化为小写字符
data = data.lower()
# 转化为无序且不重复的元素列表
chars = list(set(data))
# 获取大小信息
data_size, vocab_size = len(data), len(chars)
# print(chars)
# print("共计有%d个字符，唯一字符有%d个" % (data_size, vocab_size))
#
# # 映射
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}


#
# print(char_to_ix)
# print(ix_to_char)


def clip(gradients, maxValue):
    """
    梯度修剪
    :param gradients:
    :param maxValue:
    :return:
    """
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']
    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {
        'dWaa': dWaa,
        'dWax': dWax,
        'dWya': dWya,
        'db': db,
        'dby': dby
    }
    return gradients


def sample(parameters, char_to_ix, seed):
    """
    取样
    :param parameters:
    :param char_to_is:
    :param seed:
    :return:
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 创建one hot 向量
    x = np.zeros((vocab_size, 1))
    # 初始化a0
    a_prev = np.zeros((n_a, 1))
    # 输出列表
    output = []
    # 换行标记
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    # 如果不是换行符或者输出不超过50个
    while idx != newline_character and counter < 50:
        # 前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = utils.softmax(z)

        # 随机数种子
        np.random.seed(counter + seed)

        # 从随机概率分布中获取字符
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # 添加索引
        output.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # 更新a_prev
        a_prev = a

        seed += 1
        counter += 1

    if counter == 50:
        output.append(char_to_ix['\n'])

    return output


def optimize(X, y, a_prev, parameters, learning_rate=0.01):
    """
    rnn的单步优化
    :param X:
    :param y:
    :param a_prev:
    :param parameters:
    :param learning_rate:
    :return:
    """
    # 前向传播
    loss, cache = utils.rnn_forward(X, y, a_prev, parameters)
    # 反向传播
    gradients, a = utils.rnn_backward(X, y, parameters, cache)
    # 梯度修剪
    gradients = clip(gradients, 5)
    # 更新参数
    parameters = utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]  # 代价  参数  隐藏层


def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    """
    模型
    :param data:
    :param ix_to_char:
    :param char_to_ix:
    :param num_iterations:
    :param n_a:
    :param dino_names:
    :param vocab_size:
    :return:
    """

    # 从vocab_size中获取n_x、n_y
    n_x, n_y = vocab_size, vocab_size

    # 初始化参数
    parameters = utils.initialize_parameters(n_a, n_x, n_y)

    # 初始化损失
    loss = utils.get_initial_loss(vocab_size, dino_names)

    # 构建恐龙名称列表
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

    # 初始化LSTM隐藏状态
    a_prev = np.zeros((n_a, 1))

    # 循环
    for j in range(num_iterations):
        # 定义一个训练样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        # 选择学习率为0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = utils.smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters


# 开始时间
start_time = time.clock()
# 开始训练
parameters = model(data, ix_to_char, char_to_ix, num_iterations=3500)
# 结束时间
end_time = time.clock()
# 计算时差
minium = end_time - start_time
print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")
