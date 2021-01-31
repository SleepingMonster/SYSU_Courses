import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readfile(filename):
    dataset = pd.read_csv(filename, header=None)    # 加上后面的none参数表示不把第一行作为列属性
    dataset = dataset.values    # numpy数组，而不是list，所以不能用+拼接。
    return dataset


def PLA(dataset, iteration, learning_rate):
    """
    返回固定迭代次数中计算的w和b。
    :param dataset: 数据集
    :param iteration: 迭代次数
    :param learning_rate: 学习率
    :return:
    """
    n = len(dataset[0]) - 1
    w = np.zeros(n)
    b = 0
    i = 0
    dataset1 = np.array(dataset)    # 在外面转换，提速！！
    while i < iteration:
        flag = True
        for data in dataset1:
            x = data[:-1]   # 左闭右开，即去除最后的label
            temp = x.dot(w) + b
            if data[-1] == 0:   # 这里一定要转换0位-1，否则无法更新w,b！！
                data[-1] = -1
            if np.sign(temp) != data[-1]:   # 出现误判点，注意-1在csv中为0！！！
                flag = False
                w = w + learning_rate * data[-1] * x
                b = b + learning_rate * data[-1]
                break
        if flag is True:    # 意味着此时没有误判点
            break
        i += 1
    return w, b


def k_fold(dataset, k, i):
    """
    将数据集划分成训练集和验证集
    :param dataset:数据集
    :param k: 划分成k分
    :param i: 取第i份为验证集
    :return: 返回训练集和验证集
    """
    total = len(dataset)
    step = total // k   # 这样可以返回下取整: 步长
    start1 = i * step
    end1 = start1 + step
    # 这里的dataset是numppy数组，不是list,所以要vstack用于联结矩阵，要用2个括号。（list用+即可）
    train_set = np.vstack((dataset[:start1], dataset[end1:]))
    valid_set = dataset[start1:end1]
    return train_set, valid_set


def validation(valid_set, w, b):
    """
    返回w,b下，验证集的准确率
    :param valid_set: 验证集
    :param w: 训练集得到的w
    :param b: 训练集得到的b
    :return: 返回准确率
    """
    total = len(valid_set)
    cnt = 0
    valid_set1 = np.array(valid_set)
    for data in valid_set1:
        x = data[:-1]  # 左闭右开，即去除最后的label
        temp = x.dot(w) + b
        if data[-1] == 0:  # 这里一定要转换0位-1，否则无法更新w,b！！
            data[-1] = -1
        if np.sign(temp) == data[-1] or temp == 0:   # temp=0也是预测正确的
            cnt += 1
    return cnt/total


def main():
    dataset = readfile("train.csv")
    x = []
    y = []
    """
        for k in range(3, 20):  # 交叉验证
        temp = 0
        for i in range(k):  # 对不同的验证集取均值作为一个k的结果
            train_set, valid_set = k_fold(dataset, k, i)
            w, b = PLA(train_set, 200, 1)
            temp += validation(valid_set, w, b)
        print("对数据集进行%s折划分后的准确率为%s" % (k, temp / k))
    x.append(k)
    y.append(temp / k)
    """

    k = 10
    j = 100
    while j < 1010:
        temp = 0
        for i in range(k):  # 对不同的验证集取均值作为一个k的结果
            train_set, valid_set = k_fold(dataset, k, i)
            w, b = PLA(train_set, j, 1)
            temp += validation(valid_set, w, b)
        print("PLA在迭代次数为%s时，对数据集进行%s折划分后的准确率为%s" % (j, k, temp / k))
        x.append(j)
        y.append(temp / k)
        j += 10
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy')
    plt.title('PLA (when %s-fold, learning_rate=1)' % k)
    plt.show()


if __name__ == "__main__":
    main()
