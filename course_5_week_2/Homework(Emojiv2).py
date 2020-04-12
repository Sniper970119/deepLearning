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

import numpy as np

from course_5_week_2 import emo_utils


def sentences_to_indices(X, word_to_index, max_len):
    """
    将句子扩充到max len长度
    :param X:
    :param word_to_index:
    :param max_len:
    :return:
    """
    m = X.shape[0]  # 训练集数量
    # 使用0初始化X_indices
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        # 将第i个居住转化为小写并按单词分开。
        sentences_words = X[i].lower().split()

        # 初始化j为0
        j = 0

        # 遍历这个单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j] = word_to_index[w]

            j += 1

    return X_indices


word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """

    :param word_to_vec_map:
    :param word_to_index:
    :return:
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    # 初始化嵌入矩阵(dict to list)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # 将嵌入矩阵的每行的“index”设置为词汇“index”的词向量表示
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # 定义Keras的embbeding层
    embedding_layer = keras.layers.embeddings.Embedding(vocab_len, emb_dim, trainable=False)

    # 构建embedding层。
    embedding_layer.build((None,))

    # 将嵌入层的权重设置为嵌入矩阵。
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

def model(input_shape, word_to_vec_map, word_to_index):
    """
    算法实现
    :param input_shape:
    :param word_to_vec_map:
    :param word_to_index:
    :return:
    """
    # 定义sentence_indices为计算图的输入，维度为(input_shape,)，类型为dtype 'int32'
    sentence_indices = keras.Input(input_shape, dtype='int32')

    # 创建embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # 通过嵌入层传播sentence_indices，你会得到嵌入的结果
    embeddings = embedding_layer(sentence_indices)

    # 通过带有128维隐藏状态的LSTM层传播嵌入
    # 需要注意的是，返回的输出应该是一批序列。
    X = keras.layers.LSTM(128, return_sequences=True)(embeddings)
    # 使用dropout，概率为0.5
    X = keras.layers.Dropout(0.5)(X)
    # 通过另一个128维隐藏状态的LSTM层传播X
    # 注意，返回的输出应该是单个隐藏状态，而不是一组序列。
    X = keras.layers.LSTM(128, return_sequences=False)(X)
    # 使用dropout，概率为0.5
    X = keras.layers.Dropout(0.5)(X)
    # 通过softmax激活的Dense层传播X，得到一批5维向量。
    X = keras.layers.Dense(5)(X)
    # 添加softmax激活
    X = keras.layers.Activation('softmax')(X)

    # 创建模型实体
    model = keras.Model(inputs=sentence_indices, outputs=X)

    return model


X_train, Y_train = emo_utils.read_csv('./data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('./data/test.csv')
max_len = 10
model = model((max_len,), word_to_vec_map, word_to_index)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)
Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C=5)
model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=max_len)
Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C=5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)

print("Test accuracy = ", acc)

C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if num != Y_test[i]:
        print('正确表情：' + emo_utils.label_to_emoji(Y_test[i]) + '   预测结果： ' + X_test[i] + emo_utils.label_to_emoji(
            num).strip())
