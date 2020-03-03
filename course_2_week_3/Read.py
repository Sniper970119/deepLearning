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
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

# 这是博主自己拍的图片
from course_2_week_3 import tf_utils

my_image1 = "./test_imgs/10.png"  # 定义图片名称
fileName1 = my_image1  # 图片地址
image1 = mpimg.imread(fileName1)  # 读取图片
plt.imshow(image1)  # 显示图片
plt.show()
my_image1 = image1.reshape(1, 64 * 64 * 3).T  # 重构图片

with tf.Session() as session:
    saver = tf.train.import_meta_graph('./model/my_model-100.meta')
    saver.restore(session, tf.train.latest_checkpoint('./model'))
    parameters = {
        'W1': session.run('W1:0'),
        'b1': session.run('b1:0'),
        'W2': session.run('W2:0'),
        'b2': session.run('b2:0'),
        'W3': session.run('W3:0'),
        'b3': session.run('b3:0'),
    }
    my_image_prediction = tf_utils.predict(my_image1, parameters)  # 开始预测
    print("预测结果: y = " + str(np.squeeze(my_image_prediction)))
