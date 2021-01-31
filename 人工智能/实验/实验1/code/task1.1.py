import os
import math

f = open("semeval.txt")
lines = f.readlines()       # 将所有的行一次性读出来，这样可以知道总文档数
file_num = len(lines)    # num是文档数（即文件的行数）
dictionary = {}
word_num = [0 for i in range(file_num)]      # word_num列表存储各行中的单词个数
for line in lines:    # 按行读取文件
    str1 = line.split('\t', 2)[2]     # 按照\t来分隔出最后的单词，且取第二个单元
    str1 = str1.strip('\n')     # 去掉行末的换行符
    word_temp = str1.split(' ')  # 也可不加参数，默认为空格

    word_num[lines.index(line)] = len(word_temp)

    for word in word_temp:
        if word not in dictionary:   # 如果字典里面没有这个单词(without "()")
            temp_list = [0 for i in range(file_num+1)]     # 创建一个长度为num+1的临时列表
            temp_list[0] = 1    # 第0单元代表该单词在文件中的多少个文档中出现过
            temp_list[lines.index(line)+1] = 1    # 在列表的对应项置为1，代表在对应的文档中有1个这个单词
            dictionary[word] = temp_list    # 将这个键值对插入到字典中
        else:       # 如果字典里面有这个单词
            if dictionary[word][lines.index(line)+1] == 0:  # 原来这一行前面没有这个单词
                dictionary[word][0] = dictionary[word][0]+1     # 文档数+1
            dictionary[word][lines.index(line)+1] = 1 + dictionary[word][lines.index(line)+1]   # 无论原来这一行前面有没有这个单词，数量都加1
# 排序：
# dic_order = sorted(dictionary.items(), key=lambda dictionary: dictionary[0], reverse=False)      # 对字典进行排序,成为元组，且一定要重新赋值！！并不是原地做的！！
dic_order = sorted(dictionary)      # ！！用这个方法，这样就不会转成元组，只需要知道键值排序就好！然后再去访问原来的字典去访问对应的列表.

distinct_word_num = len(dictionary)      # distinct_word_num存储的是文件中不同的单词个数
result = [[0 for i in range(distinct_word_num)]for i in range(file_num)]    # 结果二维矩阵

for i in range(len(dic_order)):
    key = dic_order[i]      # 获得了键值：也就是单词
    idf = 0
    for j in range(len(dictionary[key])):   # 遍历每个单词对应的列表
        val = dictionary[key][j]
        if j == 0:
            idf = math.log(file_num/(val+1), 10)      # 求出对应单词的idf
        else:
            if val != 0:  # 只对不为0的进行处理
                result[j-1][i] = val/(word_num[j-1])*idf

file = open('TFIDF.txt', 'w')
file.write('file_num'+' ')
for i in range(len(result)):
    file.write(dic_order[i]+' ')
file.write('\n')
for i in range(len(result)):
    file.write(str(i+1)+'\t')     # str函数将数字转成字符串
    s = str(result[i])+'\n'
    file.write(s)
file.close()

