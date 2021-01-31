import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readfile(filename):
    dataset = pd.read_csv(filename, header=None)    # 加上后面的none参数表示不把第一行作为列属性
    dataset = dataset.values
    return dataset


"""
def h(w, x):
    return 1/(1 + np.exp(-(w.dot(x))))


def logistic(dataset, iteration, learning_rate):
    n = len(dataset[0])
    w = np.zeros(n)
    i = 0
    diff = 1e-4     # 模型收敛的判据
    dataset1 = np.array(dataset)
    while i < iteration:
        sum = 0
        for data in dataset1:
            x = np.append(data[:-1], np.array([1]))    # 在最后加上一个1
            temp = h(w, x)
            sum += (data[-1] - temp) * x
        w_new = w + learning_rate * sum
        diff1 = np.linalg.norm(w_new - w)   # 二范数：求两者的欧式距离（倒数第一次为真实值，倒数第二次为预测值）
        if diff1 <= diff:
            break
        w = w_new
        i += 1
    return w


def validation(valid_set, w):
    total = len(valid_set)
    cnt = 0
    valid_set1 = np.array(valid_set)
    for data in valid_set1:
        x = np.append(data[:-1], np.array([1]))    # 在最后加上一个1
        temp = h(w, x)
        if (temp >= 0.5 and data[-1] == 1) or (temp < 0.5 and data[-1] == 0):
            cnt += 1
    return cnt/total
"""


def h(w, x):
    """
    计算逻辑回归的预测函数 π(x)
    """
    w = w.reshape((-1, 1))   # w原为数组，通过reshape(1,-1)转成行向量; reshape(-1,1)转成列向量.而且要赋值回去！！
    temp = np.dot(x, w)    # numpy也有dot！！！
    '''
     # 防止溢出：
    for i, data in enumerate(temp):
        if data >= 0:
            temp[i] = 1.0 / (1 + np.exp(-1 * data))
        else:
            temp[i] = np.exp(data) / (1 + np.exp(data))
    '''
    temp = 1 / (1 + np.exp(-temp))
    return temp


def logistic(dataset, iteration, learning_rate):
    """
    逻辑回归函数
    :param dataset: 数据集
    :param iteration: 迭代次数
    :param learning_rate: 学习率
    :return: 返回w
    """
    n = len(dataset[0])
    w = np.zeros(n)
    w = w.reshape((-1, 1))      # 弄成列向量，否则后面会出错
    i = 0
    diff = 1e-3     # 模型收敛的判据
    dataset1 = np.array(dataset)
    # print(dataset1.shape)
    while i < iteration:
        dataset_copy = dataset1
        dataset_copy = np.delete(dataset_copy, dataset_copy.shape[1] - 1, axis=1)  # 删除最后一列
        temp_one = np.ones(dataset_copy.shape[0])
        # x_copy = x[:, :-1]      # 取出二维矩阵的前n-1列（去掉最后的1）
        dataset2 = np.insert(dataset_copy, dataset_copy.shape[1], values=temp_one, axis=1)  # 插入一列1到最后一列
        pi = h(w, dataset2)     # 返回的是列向量（在列向量的基础上做numpy）
        y = dataset1[:, dataset.shape[1] - 1]   # 取出最后一列，这时是数组
        y = y.reshape((-1, 1))   # 转成列向量
        temp = y - pi
        temp = np.repeat(temp, dataset2.shape[1], axis=1)    # 沿着列扩展成二维矩阵
        sum = np.sum(temp * dataset2, axis=0)   # 沿着行求和，即对x的各分量求和
        sum = sum.reshape((-1, 1))   # 转成列向量
        w_new = w + learning_rate * sum
        diff1 = np.linalg.norm(w_new - w)   # 二范数：求两者的欧式距离（倒数第一次为真实值，倒数第二次为预测值）
        if diff1 <= diff:
            print("梯度下降已收敛")
            break
        w = w_new
        i += 1
    return w


def validation(valid_set, w):
    """
    返回w,b下，验证集的准确率
    :param valid_set: 验证集
    :param w: 训练集得到的w
    :return: 返回准确率
    """
    total = len(valid_set)
    cnt = 0
    valid_set1 = np.array(valid_set)
    valid_copy = valid_set1
    valid_copy = np.delete(valid_copy, valid_copy.shape[1] - 1, axis=1)
    temp_one = np.ones(valid_copy.shape[0])
    valid_set2 = np.insert(valid_copy, valid_copy.shape[1], values=temp_one, axis=1)
    pi = h(w, valid_set2)
    i = 0
    for i in range(pi.shape[0]):
        if (pi[i][0] >= 0.5 and valid_set1[i][-1] == 1) or (pi[i][0] < 0.5 and valid_set1[i][-1] == 0):
            cnt += 1
    return cnt/total


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
    # train_set = dataset[:start1] + dataset[end1:]
    train_set = np.vstack((dataset[:start1], dataset[end1:]))   # vstack用于联结矩阵，要用2个括号
    valid_set = dataset[start1:end1]
    return train_set, valid_set


def main():
    dataset = readfile("train.csv")
    x = []
    y = []
    """
        for k in range(3, 20):  # 交叉验证
        temp = 0
        for i in range(k):  # 对不同的验证集取均值作为一个k的结果
            train_set, valid_set = k_fold(dataset, k, i)
            w = logistic(train_set, 100, 1)
            temp += validation(valid_set, w)
        print("对数据集进行%s折划分后的准确率为%s" % (k, temp / k))
        x.append(k)
        y.append(temp / k)
    """

    k = 10
    j = 1
    while j < 100:
        temp = 0
        for i in range(k):  # 对不同的验证集取均值作为一个k的结果
            train_set, valid_set = k_fold(dataset, k, i)
            w = logistic(train_set, j, 0.00001)
            temp += validation(valid_set, w)
        print("LR在迭代次数为%s时，对数据集进行%s折划分后的准确率为%s" % (j, k, temp / k))
        x.append(j)
        y.append(temp / k)
        j += 1
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy')
    plt.title('LR (when %s-fold,learning_rate=0.00001)' % k)
    plt.show()


if __name__ == "__main__":
    main()
