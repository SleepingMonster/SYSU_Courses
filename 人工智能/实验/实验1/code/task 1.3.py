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
        first_row = next(reader)    # next(reader)用来跳过第一行！！
        label_num = len(first_row)-1
        # labels = np.zeros((label_num, file_num), dtype=float)   # numpy矩阵来存储6种情感，每种情感有len(reader)-1列
        lines = []
        labels = [[]for i in range(label_num)]  # ！！后面的是行数label_num
        if file_name == 'train_set1.csv' or file_name == 'validation_set1.csv':
            for i, row in enumerate(reader):
                for j in range(label_num):
                    labels[j].append(float(row[j+1]))    # 第j行第i-1列是row的第j+1个情感，记得转成数字！！！
                lines.append(row[0])
            return lines, labels
        else:
            for row in reader:
                lines.append(row[0])
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
    :param flag: 标志是曼哈顿距离还是欧氏距离
    :return: 距离的list
    """
    row = row.reshape(1, -1)     # ？？row自己变成了列向量，要转回成行向量
    temp = np.repeat(row, train_tfidf.shape[0], axis=0)   # 将行向量扩展成矩阵
    if flag is True:        # 曼哈顿距离
        temp1 = np.abs(temp - train_tfidf)  # np求相减之后的绝对值
        sum_list = np.sum(temp1, axis=1)  # 行内求和
    else:       # 欧氏距离
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
    label_num = len(train_labels)
    result = [[0.0 for j in range(valid_tfidf.shape[0])]for i in range(label_num)]
    for index in range(0, valid_tfidf.shape[0]):      # ！！遍历验证集/测试集的test
        row = valid_tfidf[index]
        sum_list = calculate_distance(train_tfidf, row, True)     # 这个test和测试集的距离集合
        sort_index = np.argsort(sum_list)    # 返回排序之前最小的下标
        i = 0
        pro_sum = 0
        while i < k:
            for j in range(label_num):
                if sum_list[sort_index[i]] == 0:
                    sum_list[sort_index[i]] = 0.01
                result[j][index] += train_labels[j][sort_index[i]] / float(sum_list[sort_index[i]])
                pro_sum += train_labels[j][sort_index[i]] / float(sum_list[sort_index[i]])
            i += 1
        for i in range(label_num):
            result[i][index] /= pro_sum     # 归一化
    return result



def calculate_cor1(valid_real, valid_predict):
    """
    :param valid_real: 真实的验证集标签
    :param valid_predict: 预测的验证集标签
    :return: 返回相关系数
    """
    valid_real = np.array(valid_real)       # 转成np矩阵
    valid_predict = np.array(valid_predict)
    valid_real_average = np.mean(valid_real, axis=1)     # 行内求均值。注意：这里得到的是数组[中间无逗号]！！要将他转化成列向量！！
    # valid_real_average = valid_real_average.reshape(-1, 1)      # 转化成列向量
    valid_real_average = np.repeat(valid_real_average.reshape(6,1), valid_real.shape[1], axis=1)     # 将列向量扩展成矩阵

    valid_predict_average = np.mean(valid_predict, axis=1)
    # valid_predict_average = valid_predict_average.reshape(-1, 1)
    valid_predict_average = np.repeat(valid_predict_average.reshape(6,1), valid_predict.shape[1], axis=1)  # 将列向量扩展成矩阵
    valid_real = valid_real - valid_real_average
    # print(valid_real.shape)
    valid_predict = valid_predict - valid_predict_average
    numerator = np.sum(valid_real * valid_predict, axis=1)      # 对应为相乘，然后行内求和
    # numerator = numerator.reshape(-1,1)
    temp1 = np.sqrt(np.sum(np.square(valid_real), axis=1))        # ！！！！np.sum返回的是数组，不是矩阵！！！所以要进行转换！！
    # temp1 = temp1.reshape(-1, 1)     # 转化成列向量！！
    # print(valid_real.shape)
    temp2 = np.sqrt(np.sum(np.square(valid_real), axis=1))
    # temp2 = temp2.reshape(-1, 1)  # 转化成列向量！！
    denominator = temp1 * temp2
    cor = numerator / denominator   # numpy的点乘来计算各相关系数
    cor1 = np.mean(cor.reshape(6,1), axis=0)     # 列内求均值
    return cor1



def calculate_cor(valid_real, valid_predict):
    """
    :param valid_real: 真实的验证集标签
    :param valid_predict: 预测的验证集标签
    :return: 返回相关系数
    """
    valid_real = np.array(valid_real)       # 转成np矩阵
    valid_predict = np.array(valid_predict)
    label_num = valid_predict.shape[0]
    test_num = valid_predict.shape[1]
    correlation = np.corrcoef(valid_real, valid_predict)
    cor_sum = 0
    for i in range(label_num):
        cor_sum += correlation[i][label_num+i]
    cor = 0
    cor = cor_sum / label_num
    return cor


def main():
    train_lines, train_labels = read_file_csv('train_set1.csv')
    valid_lines, valid_labels = read_file_csv('validation_set1.csv')
    test_lines = read_file_csv('regression_simple_test.csv')
    word_list = count_words(train_lines, valid_lines, test_lines)
    train_tfidf, valid_tfidf, test_tfidf = tf_idf(train_lines, valid_lines, test_lines, word_list)
    ''' 调参：k
    k = 3
    while k < 20:
        
        valid_predict = KNN_predict(train_tfidf, valid_tfidf, k, train_labels)
        cor = calculate_cor(valid_labels, valid_predict)
        print('k = ' + str(k) + ', Correlation coefficient = ' + str(cor))
        k += 1
    '''

    test_predict = KNN_predict(train_tfidf, test_tfidf, 14, train_labels)
    test_output = pd.DataFrame({'Words (split by space)': test_lines, 'anger': test_predict[0], 'disgust': test_predict[1],
                                'fear': test_predict[2], 'joy': test_predict[3], 'sad': test_predict[4], 'surprise': test_predict[5]})
    test_output.to_csv('KNN_regression_sample.csv', index=None, encoding='utf8')    # 参数index设为None则输出的文件前面不会再加上行号

    # print(accuracy)


main()
