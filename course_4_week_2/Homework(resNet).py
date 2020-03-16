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
from keras.models import Model
from keras.initializers import glorot_uniform

from course_4_week_2 import resnets_utils

import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    实现最小模块
    :param X: 输入数据
    :param f: CONV窗口维度
    :param filters: 每层过滤器数量
    :param stage: 层数（用来命名）
    :param block: 字符串，用来命名
    :return:
    """
    # 定义命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 获取过滤器数量
    F1, F2, F3 = filters

    # 保存输入数据，用于捷径
    X_shortcut = X

    # 第一部分
    # 卷积层
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                            kernel_initializer=glorot_uniform(seed=0))(X)

    # 归一化
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)

    # 激活
    X = keras.layers.Activation('relu')(X)

    # 第二部分
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # 第三部分
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = keras.layers.Activation('relu')(X)

    # 将捷径加进来
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    卷积块
    :param X:
    :param f:
    :param fliters:
    :param stage:
    :param block:
    :param s:
    :return:
    """
    # 定义命名规则
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 获取过滤器数量
    F1, F2, F3 = filters

    # 保存输入数据
    X_shortcut = X

    # 第一部分
    # 卷积层
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)

    # 第二部分
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # 第三部分
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = keras.layers.Activation('relu')(X)

    # 捷径
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                     name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    50层的ResNet
    :param input_shape:
    :param classes:
    :return:
    """
    # 输入tensor
    X_input = keras.layers.Input(input_shape)

    # padding
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)

    # stage1
    X = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
                            kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # stage5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 均值池化层
    X = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # 输出层
    X = keras.layers.Flatten()(X)  # 多维的输入一维化
    X = keras.layers.Dense(classes, activation="softmax", name="fc" + str(classes),
                           kernel_initializer=glorot_uniform(seed=0))(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

# 标准化
X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

model.fit(X_train, Y_train, epochs=2, batch_size=32)
preds = model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)  # verbose 日志级别
# 保存模型
model.save('resnet_model.h5')
print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))

"""
 32/120 [=======>......................] - ETA: 4s
 64/120 [===============>..............] - ETA: 2s
 96/120 [=======================>......] - ETA: 0s
120/120 [==============================] - 4s 37ms/step
误差值 = 0.6602797726790111
准确度 = 0.7666666666666667
"""