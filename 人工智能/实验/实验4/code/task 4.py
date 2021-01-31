import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy


def readfile(filename):
    dataset = pd.read_csv(filename, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])     # 第一行作为列属性，取第2列开始的列
    # dataset = dataset.values    # return numpy 数组，去掉第一列的属性
    return dataset


def split_data(dataset):

    total = len(dataset)
    step = total // 10
    train_set = dataset[:step * 7]
    valid_set = dataset[step*7:]

    '''
    train_set = dataset[:-5]
    valid_set = dataset[-5:]
    '''
    return train_set, valid_set


def one_hot(dataset):
    """
    利用pd.get_dummies函数，对season, mnth, weathersit, hr, weekday进行one-hot编码。
    原因：如对于season来说，四季编码为0,1,2,3的数字大小没有实际意义，所以需要进行one-hot编码
    :param dataset:数据集
    :return:返回one-hot编码完的数据集
    """

    # 需要进行one-hot处理的特征组合
    dummy_features = ['season', 'mnth', 'weathersit', 'hr', 'weekday']
    for feature in dummy_features:
        dummies = pd.get_dummies(dataset[feature], prefix=feature, drop_first=False)    # prefix表示前缀
        # 合并进dataset中
        dataset = pd.concat([dataset, dummies], axis=1)     # 用concat来合并，而不用join来合并。join是dataframe的函数
    # 把原来的列去掉（加上atemp，两个温度差不多）
    features_to_drop = ['season', 'mnth', 'weathersit', 'hr', 'weekday', 'atemp']
    dataset = dataset.drop(features_to_drop, axis=1)
    # 把cnt这一列移到最后
    cnt_temp = dataset['cnt'].values        # 取了values之后才可以用reshape函数转成列向量
    cnt_temp = cnt_temp.reshape(len(cnt_temp), 1)
    dataset = dataset.drop('cnt', axis=1)
    '''
    for i in range(len(cnt_temp)-5):
        cnt_temp[i] = float(cnt_temp[i])
    '''
    dataset['cnt'] = cnt_temp   # 这样就可以插入到最后一列，且带标签名！！
    return dataset


def normalization(dataset):
    """
    归一化函数：对temp,hum,windspeed,cnt进行归一化处理
    :param dataset: 数据集
    :return: 返回数据集，且去掉了第一列（之后不需要属性名了）
    """
    # 需要进行归一化处理的属性：
    # nor_features = ['temp', 'hum', 'windspeed', 'cnt']
    nor_features = ['temp', 'hum', 'windspeed']
    data_num = dataset.shape[0]
    list = []
    for feature in nor_features:
        list.append(dataset[feature])   # 这里提取出来会变成行向量
    list = np.array(list)   # 转成numpy数组
    list = list.reshape(-1, len(nor_features))   # 转置
    mean = np.mean(list, axis=0)
    min = np.min(list, axis=0)      # axis=0表示0维消失
    max = np.max(list, axis=0)
    temp = max - min
    for i, feature in enumerate(nor_features):
        dataset.loc[:, feature] = (dataset[feature] - min[i]) / temp[i]
    '''
    这样不行，因为弄出来不是-1,1的
    mean = np.average(list, axis=1)     # numpy并行计算均值和标准差
    std = np.std(list, axis=1)
    for i, feature in enumerate(nor_features):
        dataset.loc[:, feature] = (dataset[feature] - mean[i]) / std[i]
    '''

    # pd.set_option('display.max_columns', 60)  # 这样可以输出的时候输出全部的行（max=60）
    # print(dataset.head())
    dataset = dataset.values
    return dataset


def make_matrix(row_num, col_num):
    """
    生成矩阵，大小为row_num*col_num， 数值随机
    :param row_num: 行数
    :param col_num: 列数
    :return: 返回矩阵list
    """
    result = []
    for i in range(row_num):
        temp = []
        for j in range(col_num):
            temp.append(random.uniform(-1.0, 1.0))    # 产生一个随机数在（-1,1）之间
            # temp.append(1.0)  # 产生一个随机数在（-1,1）之间
        result.append(temp)
    return result


def init_para(dataset, hnode_num):
    """
    初始化各参数
    :param dataset:
    :param hnode_num:
    :return: w_hidden, b_hidden, w_output, b_output
    """
    feature_num = dataset.shape[1] - 1  # 第一维-1等于变量数
    w_hidden = np.array(make_matrix(feature_num, hnode_num))
    b_hidden = np.array(make_matrix(1, hnode_num))
    w_output = np.array(make_matrix(hnode_num, 1))
    b_output = random.uniform(0, 1.0)  # b_output初始化为[-1,1]之间的一个数
    return w_hidden, b_hidden, w_output, b_output


def sigmoid(x):
    '''
    temp = [[0] * len(x[0]) for i in range(len(x))]     # 二维数组初始化
    for i, data in enumerate(x):
        for j, data_each in enumerate(data):
            if data_each >= 0:
                temp[i][j] = 1.0 / (1 + np.exp(-data_each))
            else:
                temp[i][j] = np.exp(data_each) / (1 + np.exp(data_each))
    result = np.array(temp)
    return result
    '''
    return 1.0 / (1.0 + np.exp(-x))


def activation_func(func_type, x):
    if func_type == "sigmoid":
        # 跑测试集的时候要加这句话！
        # x1 = np.array(x, dtype=np.float64)  # 否则会出错
        return sigmoid(x)
    elif func_type == "tanh":
        return 2 * sigmoid(2*x) - 1
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif func_type == "relu":
        return np.maximum(0.01*x, x)


def derivation(func_type, hidden_output):   # 激活函数的求导
    if func_type == "sigmoid":
        return hidden_output * (1 - hidden_output)
    elif func_type == "tanh":
        return 1 - np.square(hidden_output)
    elif func_type == "relu":
        temp = [[0.0] * len(hidden_output[0]) for i in range(len(hidden_output))]     # 二维数组初始化
        for i, data in enumerate(hidden_output):
            for j, data_each in enumerate(data):
                if data_each < 0:
                    temp[i][j] = 0.01
                else:
                    temp[i][j] = 1
        result = np.array(temp)
        return result


def forward_pass(dataset, w_hidden, b_hidden, w_output, b_output, func_type):
    """
    前向传播
    :param dataset: 数据集
    :param w_hidden : 隐藏层 w
    :param b_hidden: 隐藏层b（偏置theta)
    :param w_output: 输出层 w
    :param b_output: 输出层 b
    :param func_type: 激活函数类型
    :return: output, hidden_output
    """
    data_num = dataset.shape[0]     # 样本数
    # 隐藏层计算
    b_hidden1 = np.repeat(b_hidden, data_num, axis=0)   # 将b_hidden沿着行扩展
    dataset1 = dataset[:, 0:dataset.shape[1] - 1]  # 剔除掉最后一列的label
    hidden_input = np.dot(dataset1, w_hidden)    # 隐藏层节点
    hidden_input = hidden_input + b_hidden1   # 加上偏置theta
    hidden_output = activation_func(func_type, hidden_input)    # 激活函数
    # 输出层计算
    output = np.dot(hidden_output, w_output)    # 输出层预测输出
    b_output1 = np.array([b_output] * data_num)     # 这样可以形成数组，且各单元值相同。因为这只是一个数，所以用按行repeat不会形成列向量；多个数就会
    b_output1 = b_output1.reshape(-1, 1)      # 转成列向量
    output = output + b_output1     # 加上偏置量
    # output = activation_func(func_type, output)

    # print(output[:5])
    # print(dataset[:5, -1])
    return output, hidden_output


def backward_pass(dataset, output, hidden_output, learning_rate, w_hidden, b_hidden, w_output, b_output, func_type):
    feature_num = dataset.shape[1] - 1  # 第一维-1等于变量数
    hnode_num = hidden_output.shape[1]  # 隐藏层节点数
    data_num = dataset.shape[0]  # 样本数
    dataset1 = dataset[:, 0:dataset.shape[1] - 1]  # 剔除掉最后一列的label
    label = (dataset[:, dataset.shape[1]-1]).reshape(data_num, 1)   # 取出最后一列，并转成列向量
    # 计算输出层误差err_output
    err_output = label - output     # 此时激活函数 y=x的导数为1
    # 计算err_output * w_output
    temp = np.repeat(err_output, hnode_num, axis=1)  # 沿着列扩展err_k
    temp_w_output = np.repeat(w_output.reshape(1, hnode_num), data_num, axis=0)   # 沿着行扩展w_output，原来是（hnode_num,1）
    temp = temp * temp_w_output

    # 计算隐藏层误差 err_hidden
    err_hidden = derivation(func_type, hidden_output) * temp
    # 求出所有样本的平均误差：输出层平均
    err_output1 = np.repeat(err_output, hnode_num, axis=1)
    err_output1 = err_output1 * hidden_output   # 先乘法
    err_output_mean_w = np.mean(err_output1, axis=0)    # 更新w时的误差*Oj的均值
    err_output_mean_b = np.mean(err_output)     # 更新b时的误差均值
    # 更新输出层的w,b：
    for i in range(hnode_num):
        w_output[i] += learning_rate * err_output_mean_w[i]
    b_output += learning_rate * err_output_mean_b

    # 求出所有样本的平均误差：隐藏层平均
    err_hidden_mean_b = np.mean(err_hidden, axis=0)
    err_hidden_mean_w = []
    for i in range(hnode_num):
        err_hidden_each = (err_hidden[:, i]).reshape(-1, 1)  # 取出一个隐藏层节点的误差
        err_hidden_each1 = np.repeat(err_hidden_each, feature_num, axis=1)  # 沿着列扩展
        err_hidden_each1 = err_hidden_each1 * dataset1
        err_hidden_mean_w.append(np.mean(err_hidden_each1, axis=0))
    err_hidden_mean_w = np.array(err_hidden_mean_w)     
    # 更新隐藏层的w,b：
    for j in range(hnode_num):
        for i in range(feature_num):
            w_hidden[i][j] += learning_rate * err_hidden_mean_w[j][i]
        b_hidden[0][j] += learning_rate * err_hidden_mean_b[j]     # b_hidden是一个行向量，(1,hnode_num)
    return w_hidden, b_hidden, w_output, b_output


def bpnn(hnode_num, dataset, iteration, learning_rate, func_type):
    w_hidden, b_hidden, w_output, b_output = init_para(dataset, hnode_num)
    i = 0
    for i in range(iteration):
        # 前向传播：得到输出层和隐藏层的输出
        print("在迭代次数为%s时，预测值和真实值分别为：" % i)
        output, hidden_output = forward_pass(dataset, w_hidden, b_hidden, w_output, b_output, func_type)
        # 后向传播：更新两组w,b
        w_hidden, b_hidden, w_output, b_output = backward_pass(dataset, output, hidden_output, learning_rate,
                                                               w_hidden, b_hidden, w_output, b_output, func_type)
    return w_hidden, b_hidden, w_output, b_output


def validation(dataset, func_type, w_hidden, b_hidden, w_output, b_output, threshold):     # 预测验证集的输出
    output, hidden_output = forward_pass(dataset, w_hidden, b_hidden, w_output, b_output, func_type)
    label = dataset[:, dataset.shape[1] - 1].reshape(-1, 1)    # 取出最后一列
    diff = output - label
    loss = np.mean(np.square(diff)) / 2
    cnt = 0
    for i in range(diff.shape[0]):
        if diff[i][0] < threshold:
            cnt += 1
    accuracy = cnt / diff.shape[0]
    return loss, accuracy


def train_test():   # 调参：迭代次数遍历
    x = []
    y1 = []
    y2 = []
    # 数据预处理
    dataset = readfile("train.csv")  # 读文件（先不去掉属性）
    dataset = one_hot(dataset)  # one-hot编码
    dataset = normalization(dataset)  # 归一化
    # 划分数据集
    train_set, valid_set = split_data(dataset)
    # bpnn
    func_type = "sigmoid"
    learning_rate = 0.05   # 调小学习率就不会溢出
    hnode_num = 50
    iteration = 500
    threshold = 1
    i = 0
    # 初始化4个参数
    w_hidden, b_hidden, w_output, b_output = init_para(dataset, hnode_num)
    for i in range(iteration):
        # 前向传播：得到输出层和隐藏层的输出
        output, hidden_output = forward_pass(dataset, w_hidden, b_hidden, w_output, b_output, func_type)
        # 后向传播：更新两组w,b
        w_hidden, b_hidden, w_output, b_output = backward_pass(dataset, output, hidden_output, learning_rate,
                                                               w_hidden, b_hidden, w_output, b_output, func_type)
        if (i + 1) % 10 == 0:
            loss, accuracy = validation(valid_set, func_type, w_hidden, b_hidden, w_output, b_output, threshold)
            print("BPNN在激活函数为%s，迭代次数为%s时，训练集的loss为%s，准确率为%s（threshold=%s）" % (func_type, i + 1, loss,
                                                                              accuracy, threshold))
            x.append(i + 1)
            y1.append(loss)
            y2.append(accuracy)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, y1)
    plt.grid(True)
    plt.xlabel('Iteration Number')
    plt.ylabel('Loss')
    plt.title('BPNN(when hnode_num=%s, learning_rate=%s, %s)' % (hnode_num, learning_rate, func_type))

    plt.subplot(212)
    plt.plot(x, y2)
    plt.grid(True)
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy(threshold = %s)' % threshold)
    # plt.title('BPNN(when hnode_num=50, learning_rate=0.05, sigmoid)')
    plt.show()  # 显示图像


def test():     # 测试，即训练集迭代更新得到权重和偏置，用验证集验证结果，作图，调参
    x = []
    y = []
    # 数据预处理
    dataset = readfile("train.csv")  # 读文件（先不去掉属性）
    dataset = one_hot(dataset)  # one-hot编码
    dataset = normalization(dataset)  # 归一化
    # 划分数据集
    train_set, valid_set = split_data(dataset)
    # bpnn
    func_type = 1
    learning_rate = 0.05
    hnode_num = 50
    iteration = 500
    w_hidden, b_hidden, w_output, b_output = bpnn(hnode_num, train_set, iteration, learning_rate, func_type)
    # 预测验证集
    loss, accuracy = validation(valid_set, func_type, w_hidden, b_hidden, w_output, b_output)
    print("loss of valid_set is %s, accuracy is %s (when threshold is 0.3)" % (loss, accuracy))


def predict():  # 训练集训练得到权重和偏置，预测测试集的次数
    '''
    # 读文件
    train_set = readfile("train2.csv")
    test_set = readfile("test2.csv")
    # one-hot操作
    train_set = one_hot(train_set)
    test_set = one_hot(test_set)
    pd.set_option('display.max_columns', 60)
    print(train_set.head)
    print(test_set.head)
    # 归一化
    train_set = normalization(train_set)
    test_set = normalization(test_set)
    '''

    # 数据预处理
    dataset = readfile("train3.csv")  # 读文件（先不去掉属性）
    dataset = one_hot(dataset)  # one-hot编码
    dataset = normalization(dataset)  # 归一化
    # 划分数据集
    train_set, test_set = split_data(dataset)

    # bpnn
    func_type = 1
    learning_rate = 0.05
    hnode_num = 50
    iteration = 500
    w_hidden, b_hidden, w_output, b_output = bpnn(hnode_num, train_set, iteration, learning_rate, func_type)
    # predict
    # print(w_hidden.shape)
    loss, accuracy = validation(train_set, learning_rate, func_type, w_hidden, b_hidden, w_output, b_output)
    print("loss of train_set is %s, accuracy is %s" % (loss, accuracy))
    test_output, test_hidden_output = forward_pass(test_set, w_hidden, b_hidden, w_output, b_output, func_type)
    print(test_output)


def main():
    # predict()
    # test()
    train_test()


if __name__ == "__main__":
    main()
