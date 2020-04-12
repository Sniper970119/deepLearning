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
from course_5_week_2 import w2v_utils

# 加载词向量
words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')

# 查看hello的词向量
print(word_to_vec_map['hello'])


def cosine_similarity(u, v):
    """
    余弦相似度
    :param u:
    :param v:
    :return:
    """
    distance = 0
    # 计算u与v的内积
    dot = np.dot(u, v)
    # 计算u的第2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    # 计算v的第二范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))
    # 计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)
    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    a与b相比就类似于d与____相比一样
    :param word_a:
    :param word_b:
    :param word_to_vec_map:
    :return:
    """
    # 单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    # 获取词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    # 获取全部单词
    words = word_to_vec_map.keys()
    # 初始化
    max_cosine_sim = -100
    best_word = None

    # 遍历整个数据集
    for word in words:
        if word in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[word] - e_c)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word
    return best_word



