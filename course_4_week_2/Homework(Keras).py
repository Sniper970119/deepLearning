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

import keras
from course_4_week_2 import kt_utils
import keras.backend as K

K.set_image_data_format('channels_last')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# 标准化数据
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


def model(input_shape):
    """
    总体模型
    :param input_shape:
    :return:
    """
    # 定义输出的placeholder
    X_input = keras.layers.Input(input_shape)

    # 用0填充
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU
    X = keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)
    X = keras.layers.Activation('relu')(X)

    # 最大池化层
    X = keras.layers.MaxPool2D((2, 2), name='max_pool')(X)

    # 降维，矩阵转换为向量+全连接
    X = keras.layers.Flatten()(X)  # 多维的输入一维化
    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)  # 输出一个神经元 sigmoid激活

    model = keras.models.Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


# 创建一个模型实体
happy_model = model(X_train.shape[1:])
# 编译模型
happy_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
# 训练模型
happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
# 评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)  # verbose 日志级别
# 保存模型
happy_model.save('happy_model.h5')
print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))
"""
150/150 [==============================] - 1s 7ms/step
误差值 = 0.1285133997599284
准确度 = 0.9333333373069763
"""
