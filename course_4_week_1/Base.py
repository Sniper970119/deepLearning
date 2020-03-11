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

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


# padding
# arr3D = np.array([[[1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4]],
#
#                   [[0, 1, 2, 3, 4, 5],
#                    [0, 1, 2, 3, 4, 5],
#                    [0, 1, 2, 3, 4, 5]],
#
#                   [[1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4]]])
#
# print('constant:  \n' + str(np.pad(arr3D, ((0, 0), (1, 1), (2, 2)), 'constant')))


def zero_pad(X, pad):
    """
    对图片X进行填充
    :param X:
    :param pad:
    :return:
    """
    X_paded = np.pad(X, (
        (0, 0),  # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),  # 通道数，不填充
                     'constant', constant_values=0)  # 连续一样的值填充

    return X_paded


# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_paded = zero_pad(x, 2)
#
# # 绘制图
# fig, axarr = plt.subplots(1, 2)  # 一行两列
# axarr[0].set_title('x')
# axarr[0].imshow(x[0, :, :, 0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0, :, :, 0])
# plt.show()

def conv_single_step(a_slice_prev, W, b):
    """
    进行卷积操作
    :param a_slice_prev:
    :param W:
    :param b:
    :return:
    """
    s = np.multiply(a_slice_prev, W) + b

    Z = np.sum(s)

    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    卷积前向传播
    :param A_prev:
    :param W:
    :param b:
    :param hparameters:
    :return:
    """
    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape
    # 获取超参值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # 使用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    # 自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 执行单步卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    # 数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))

    # 存储一些缓存值，以便于反向传播使用
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)


def pool_forward(A_prev, hparameters, mode="max"):
    """
    池化层
    :param A_prev:
    :param hparameters:
    :param mode:
    :return:
    """

    # 获取输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取超参数的信息
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # 池化完毕，校验数据格式
    assert (A.shape == (m, n_H, n_W, n_C))

    # 校验完毕，开始存储用于反向传播的值
    cache = (A_prev, hparameters)

    return A, cache


def conv_backward(dZ, cache):
    """
    卷积层反向传播
    :param dZ:
    :param cache:
    :return:
    """
    # 获取cache的值
    (A_prev, W, b, hparameters) = cache

    # 获取A_prev的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取dZ的基本信息
    (m, n_H, n_W, n_C) = dZ.shape

    # 获取权值的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取hparaeters的值
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    # 初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # 现在处理数据
    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # 数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)



def create_mask_from_window(x):
    """
    从输入矩阵中创建掩码，以保存最大值的矩阵的位置。
    :param x:
    :return:
    """
    mask = x == np.max(x)

    return mask


def distribute_value(dz, shape):
    """
    平均池化的反向传播
    :param dz:
    :param shape:
    :return:
    """
    # 获取矩阵的大小
    (n_H, n_W) = shape

    # 计算平均值
    average = dz / (n_H * n_W)

    # 填充入矩阵
    a = np.ones(shape) * average

    return a


def pool_backward(dA, cache, mode="max"):
    """
    实现池化层的反向传播
    :param dA:
    :param cache:
    :param mode:
    :return:
    """
    # 获取cache中的值
    (A_prev, hparaeters) = cache

    # 获取hparaeters的值
    f = hparaeters["f"]
    stride = hparaeters["stride"]

    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    # 开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 选择反向传播的计算方式
                    if mode == "max":
                        # 开始切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 创建掩码
                        mask = create_mask_from_window(a_prev_slice)
                        # 计算dA_prev,就是只在最大值的位置进行加减
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    # 数据处理完毕，开始验证格式
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev
