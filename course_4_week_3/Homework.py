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

import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import keras

from course_4_week_3.yad2k.models.keras_yolo import *
from course_4_week_3 import yolo_utils


def yolo_filter_boxs(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    通过阈值来过滤对象和分类的置信度
    :param box_confidence: tensor张量，（19，19，5，1） 所有anchor box的pc
    :param boxes: tensor张量，（19，19，5，4），所有anchor box的x y h w
    :param box_class_probs: tensor张量，（19，19，5，80） 80个特征的c
    :param threshold: 阈值
    :return:
    """
    # 计算anchor box 得分
    box_scores = box_confidence * box_class_probs

    # 找到anchor box的最大值索引以及对应的得分
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # 根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)

    # 对scores、boxes、classes使用掩码
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """
    计算iou交并比
    :param box1:
    :param box2:
    :return:
    """
    # 计算两个盒子相交部分的面积
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1 - xi2) * (yi1 - yi2)

    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    anchor box 的非最大值抑制
    :param scores:tensor张量，yolo filter boxes的输出
    :param boxes:tensor张量，yolo filter boxes的输出
    :param classes:tensor张量，yolo filter boxes的输出
    :param max_boxes:整数，预测anchor box的最大值
    :param iou_threshold:交并比阈值
    :return:
    """
    # 定义变量并初始化
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # 下面两步是进行iou计算，实际上我们并不需要使用自己实现的iou函数

    # 获取与我们保留的anchor box相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # 选择保留的anchor box
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    将yolo的输出（很多anchor box）转换为预测框以及他们的分数、坐标和分类
    :param yolo_outputs:编码模型的输出
    :param image_shape:tensor张量，输出图像的维度（608，608）
    :param max_boxes:预测anchor box的最大值
    :param score_threshold:可能性阈值
    :param iou_threshold:iou阈值
    :return:
    """
    # 获取yolo模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 可信度分值过滤（选出有东西框）
    scores, boxes, classes = yolo_filter_boxs(box_confidence, boxes, box_class_probs, score_threshold)

    # 缩放anchor box
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # 非最大值抑制（过滤重复框）
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    预测部分
    :param sess:
    :param image_file:
    :param is_show_info:
    :param is_plot:
    :return:
    """
    # 图像预处理
    image, image_data = yolo_utils.preprocess_image('images/' + image_file, model_image_size=(608, 608))

    # 运行会话
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    # 指定边框颜色
    colors = yolo_utils.generate_colors(class_names)

    # 绘制边框
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存图片
    image.save(os.path.join("out", image_file), quality=100)

    # 打印出已经绘制了边界框的图
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        plt.imshow(output_image)
        plt.show()

    return out_scores, out_boxes, out_classes


# 测试yolo
with K.get_session() as sess:
    class_names = yolo_utils.read_classes('model_data/coco_classes.txt')
    anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)
    yolo_model = keras.models.load_model("model_data/yolov2.h5")
    # yolo_model.summary()

    # 将yolo输出转换为边界框
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

    out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
    # for i in range(1, 121):
    #     # 计算需要在前面填充几个0
    #     num_fill = int(len("0000") - len(str(1))) + 1
    #     # 对索引进行填充
    #     filename = str(i).zfill(num_fill) + ".jpg"
    #     print("当前文件：" + str(filename))
    #
    #     # 开始绘制，不打印信息，不绘制图
    #     out_scores, out_boxes, out_classes = predict(sess, filename, is_show_info=False, is_plot=False)
    #
    # print("绘制完成！")
