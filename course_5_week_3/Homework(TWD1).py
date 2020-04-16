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
from course_5_week_3.td_utils import *

# x = graph_spectrogram("audio_examples/example_train.wav")
# plt.show()


Tx = 5511  # 从频谱图输入到模型的时间步数
n_freq = 101  # 在频谱图的每个时间步输入模型的频率数

Ty = 1375  # 我们模型输出中的时间步数

# 使用pydub加载音频片段
activates, negatives, backgrounds = load_raw_audio()


def get_random_time_segment(segment_ms):
    """
    获取 10,000 ms音频剪辑中时间长为 segment_ms 的随机时间段。

    参数：
    segment_ms -- 音频片段的持续时间，以毫秒为单位("ms" 代表 "毫秒")

    返回：
    segment_time -- 以ms为单位的元组（segment_start，segment_end）
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # 确保段不会超过10秒背景
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    检查段的时间是否与现有段的时间重叠。

    参数：
    segment_time -- 新段的元组（segment_start，segment_end）
    previous_segments -- 现有段的元组列表（segment_start，segment_end）

    返回：
    如果时间段与任何现有段重叠，则为True，否则为False
    """

    segment_start, segment_end = segment_time

    # 第一步：将重叠标识 overlap 初始化为“False”标志 (≈ 1 line)
    overlap = False

    # 第二步：循环遍历 previous_segments 的开始和结束时间。
    # 比较开始/结束时间，如果存在重叠，则将标志 overlap 设置为True (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    在随机时间步骤中在背景噪声上插入新的音频片段，确保音频片段与现有片段不重叠。

    参数：
    background -- 10秒背景录音。
    audio_clip -- 要插入/叠加的音频剪辑。
    previous_segments -- 已放置的音频片段的时间

    返回：
    new_background -- 更新的背景音频
    """

    # 以ms为单位获取音频片段的持续时间
    segment_ms = len(audio_clip)

    # 第一步：使用其中一个辅助函数来选择要插入的随机时间段
    # 新的音频剪辑。 (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # 第二步：检查新的segment_time是否与previous_segments之一重叠。
    # 如果重叠如果是这样，请继续随机选择新的 segment_time 直到它不重叠。(≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # 第三步： 将新的 segment_time 添加到 previous_segments 列表中 (≈ 1 line)
    previous_segments.append(segment_time)

    # 第四步： 叠加音频片段和背景
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    更新标签向量y。段结尾的后面50个输出的标签应设为 1。
    严格来说，我们的意思是 segment_end_y 的标签应该是 0，而随后的50个标签应该是1。

    参数：
    y -- numpy数组的维度 (1, Ty), 训练样例的标签
    segment_end_ms -- 以ms为单位的段的结束时间

    返回：
    y -- 更新标签
    """

    # 背景持续时间（以频谱图时间步长表示）
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # 将1添加到背景标签（y）中的正确索引
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1

    return y


def create_training_example(background, activates, negatives):
    """
    创建具有给定背景，正例和负例的训练示例。

    参数：
    background -- 10秒背景录音
    activates --  "activate" 一词的音频片段列表
    negatives -- 不是 "activate" 一词的音频片段列表

    返回：
    x -- 训练样例的频谱图
    y -- 频谱图的每个时间步的标签
    """

    # 设置随机种子
    np.random.seed(18)

    # 让背景更安静
    background = background - 20

    # 第一步：初始化 y （标签向量）为0 (≈ 1 line)
    y = np.zeros((1, Ty))

    # 第二步：将段时间初始化为空列表 (≈ 1 line)
    previous_segments = []

    # 从整个 "activate" 录音列表中选择0-4随机 "activate" 音频片段
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # 第三步： 循环随机选择 "activate" 剪辑插入背景
    for random_activate in random_activates:
        # 插入音频剪辑到背景
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # 从 segment_time 中取 segment_start 和 segment_end
        segment_start, segment_end = segment_time
        # 在 "y" 中插入标签
        y = insert_ones(y, segment_end_ms=segment_end)

    # 从整个负例录音列表中随机选择0-2个负例录音
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # 第四步： 循环随机选择负例片段并插入背景中
    for random_negative in random_negatives:
        # 插入音频剪辑到背景
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # 标准化音频剪辑的音量
    background = match_target_amplitude(background, -20.0)

    # 导出新的训练样例
    file_handle = background.export("train" + ".wav", format="wav")
    print("文件 (train.wav) 已保存在您的目录中。")

    # 获取并绘制新录音的频谱图（正例和负例叠加的背景）
    x = graph_spectrogram("train.wav")

    return x, y


x, y = create_training_example(backgrounds[0], activates, negatives)
