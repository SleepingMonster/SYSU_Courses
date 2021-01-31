import os
# import math
import csv
import numpy as np
import pandas as pd
# import heapq


def read_file_csv(file_name):
    """
    :param file_name: 读取train_set,validation_set,test_set三个文件的内容(如果是前两者，则将情感也读出来）
    :return: 返回文件内容&情感，即文件的第0列和第1列
    """
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        labels = []
        lines = []
        if file_name == 'train_set.csv' or file_name == 'validation_set.csv':
            for row in reader:
                labels.append(row[1])
                lines.append(row[0])
            del(lines[0])   # 由于reader没有index函数，所以就只能在构建完list之后再删除第一行
            del(labels[0])
            return lines, labels
        else:
            for row in reader:
                lines.append(row[0])
            del (lines[0])
            return lines


def count_words(train_lines, valid_lines, test_lines):
    """
    :param train_lines: 训练集的词表
    :param valid_lines: 验证集的词表
    :param test_lines: 测试集的词表
    :return: 去重后的大词表list
    """
    word_set = set()    # 使用set来进行查找，快！
    word_list = []
    for row in train_lines:
        word_temp = row.split(' ')
        for i in word_temp:
            if i not in word_set:
                word_set.add(i)
                word_list.append(i)
    for row in valid_lines:
        word_temp = row.split(' ')
        for i in word_temp:
            if i not in word_set:
                word_set.add(i)
                word_list.append(i)
    for row in test_lines:
        word_temp = row.split(' ')
        for i in word_temp:
            if i not in word_set:
                word_set.add(i)
                word_list.append(i)
    return word_list


def tf_idf(train_lines, valid_lines, test_lines, word_list):
    """
    这样子只用遍历一遍，即可同时得到tf和idf矩阵，并通过点乘求得tf-idf矩阵
    :param train_lines: 训练集文本
    :param valid_lines: 验证集文本
    :param test_lines: 测试集文本
    :param word_list: 去重后的大单词表
    :return: 三者的TF-IDF矩阵
    """
    word_num = len(word_list)
    file_num = len(train_lines) + len(valid_lines) + len(test_lines)
    idf = np.zeros(word_num, dtype=float)    # 创建一维idf数组，大小为word_list长度，初始化为0
    train_tf = np.zeros((len(train_lines), word_num), dtype=float)     # 创建三种二维的tf矩阵，初始化为0
    valid_tf = np.zeros((len(valid_lines), word_num), dtype=float)
    test_tf = np.zeros((len(test_lines), word_num), dtype=float)
    for i, row in enumerate(train_lines):
        word_temp = row.split(' ')
        total = len(word_temp)
        for word in word_temp:
            index = word_list.index(word)
            if train_tf[i][index] == 0:     # 更新idf矩阵
                idf[index] += 1
            train_tf[i][index] = (train_tf[i][index]*total+1)/total
    for i, row in enumerate(valid_lines):
        word_temp = row.split(' ')
        total = len(word_temp)
        for word in word_temp:
            index = word_list.index(word)
            if valid_tf[i][index] == 0:     # 更新idf矩阵
                idf[index] += 1
            valid_tf[i][index] = (valid_tf[i][index]*total+1)/total
    for i, row in enumerate(test_lines):
        word_temp = row.split(' ')
        total = len(word_temp)
        for word in word_temp:
            index = word_list.index(word)
            if test_tf[i][index] == 0:     # 更新idf矩阵
                idf[index] += 1
            test_tf[i][index] = (test_tf[i][index]*total+1)/total
    idf = np.log10(file_num/(idf+1))
    # idf1 = np.repeat(idf, len(train_lines),axis=0)      # 扩展行数，使得可以使用
    train_tfidf = idf * train_tf    # 利用点乘操作
    valid_tfidf = idf * valid_tf
    test_tfidf = idf * test_tf
    return train_tfidf, valid_tfidf, test_tfidf


def calculate_distance(train_tfidf, row, flag):
    """
    这里，运用了numpy进行举例的计算，一次性将1个test和所有的训练集进行了距离计算，加速！！
    :param train_tfidf: 训练集
    :param row: 验证集
    :param flag: 表示用的是什么距离
    :return: 距离的list
    """
    row = row.reshape(1, -1)     # ？？row自己变成了列向量，要转回成行向量
    temp = np.repeat(row, train_tfidf.shape[0], axis=0)   # 将行向量扩展成矩阵
    if flag is True:  # 曼哈顿距离
        temp1 = np.abs(temp - train_tfidf)  # np求相减之后的绝对值
        sum_list = np.sum(temp1, axis=1)  # 行内求和
    else:  # 欧氏距离
        temp1 = np.square(temp - train_tfidf)  # np求相减之后的平方
        sum_list = np.sum(temp1, axis=1)  # 行内求和
        sum_list = np.sqrt(sum_list)
    return sum_list


def KNN_predict(train_tfidf, valid_tfidf, k, train_labels):
    """
    :param train_tfidf: 训练集
    :param valid_tfidf: 验证集
    :param k: k值
    :param train_labels: 训练集的情感集合
    :return: 返回预测的情感集合
    """
    result = [0 for i in range(valid_tfidf.shape[0])]
    for index in range(0, valid_tfidf.shape[0]):      # ！！按照下标来访问
        row = valid_tfidf[index]
        sum_list = calculate_distance(train_tfidf, row, True)     # 这个test和测试集的距离集合
        sort_index = np.argsort(sum_list)    # 返回排序之前最小的下标
        dict1 = {}
        i = 0
        while i < k:
            if train_labels[sort_index[i]] not in dict1:
                dict1[train_labels[sort_index[i]]] = 1
            else:
                dict1[train_labels[sort_index[i]]] += 1
            i += 1
        result[index] = max(dict1, key=lambda x: dict1[x])   # 找出字典中值最大对应的键： lambda函数
    return result


def calculate_accuracy(valid_real, valid_predict):
    """
    :param valid_real: 真实的验证集标签
    :param valid_predict: 预测的验证集标签
    :return: 返回准确率
    """
    total = len(valid_real)
    count = 0
    for i in range(total):
        if valid_real[i] == valid_predict[i]:
            count += 1
    return count/total


def main():
    train_lines, train_labels = read_file_csv('train_set.csv')
    valid_lines, valid_labels = read_file_csv('validation_set.csv')
    test_lines = read_file_csv('classification_simple_test.csv')
    word_list = count_words(train_lines, valid_lines, test_lines)
    train_tfidf, valid_tfidf, test_tfidf = tf_idf(train_lines, valid_lines, test_lines, word_list)
    ''' 调参：k
    k = 3
    while k < 20:
        valid_predict = KNN_predict(train_tfidf, valid_tfidf, k, train_labels)
        accuracy = calculate_accuracy(valid_labels, valid_predict)
        print('k = '+str(k)+', accuracy = '+str(accuracy));
        k += 1
    '''


    test_predict = KNN_predict(train_tfidf, test_tfidf, 13, train_labels)
    test_output = pd.DataFrame({'Words (split by space)': test_lines, 'label': test_predict})
    test_output.to_csv('KNN_classification_sample.csv', index=None, encoding='utf8')    # 参数index设为None则输出的文件前面不会再加上行号

    # print(accuracy)


main()
