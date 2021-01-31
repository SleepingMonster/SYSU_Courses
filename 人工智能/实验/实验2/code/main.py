import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class DecisionNode(object):
    # 类似于C里面的构造函数，初始化类实例的属性
    def __init__(self, label=None, attribute=None, branch=None):
        self.attribute = attribute  # 节点的属性标签
        self.label = label  # 保存当前分支的label，当是叶结点的时候值才有真正意义
        self.branch = branch


def readfile(filename):
    """
    将数据和label读出来，存入labelSet和dataSet中
    :return: labelSet & dataSet
    """
    dataset = pd.read_csv(filename)
    labelset = list(dataset.columns.values)     # 这样可以读出属性名
    dataset = list(dataset.values)     # 读出数据，存到列表中，这里去除了第一行的表头！！
    return dataset, labelset


def k_fold(dataset, k, i):
    """
    "交叉验证法“ 划分数据集为训练集+验证集
    :param dataset: 大数据集
    :param k:划分的总份数
    :param i:取第i份作为验证集
    :return:返回训练集和验证集
    """
    total = len(dataset)
    step = total // k   # 这样可以返回下取整: 步长
    start1 = i * step
    end1 = start1 + step
    train_set = dataset[:start1] + dataset[end1:]
    valid_set = dataset[start1:end1]
    return train_set, valid_set


def get_attribute_dic(dataset, labelset):
    """
    计算各属性的可能取值
    :param dataset: 数据集
    :param labelset: 属性名称集合
    :return: 返回属性字典：键值key为labelset中的下标，值value为对应属性可能的取值
    """
    attribute_num = len(labelset) - 1   # 属性个数：减去最后的标签
    attribute_dic = {}
    for i in range(attribute_num):
        temp_set = set()
        for line in dataset:
            temp_set.add(line[i])     # 获取当前特征所有可能的取值
        attribute_dic[i] = temp_set
    return attribute_dic



def cal_entropy(subdataset, index):
    """
    可以用来计算经验熵（index=-1)或者某个属性的熵，或者条件熵（dataset经过处理），用于ID3或C4.5
    :param subdataset: 数据集
    :param index: 目标的列号
    :return: 返回熵值
    """
    size = len(subdataset)
    count = {}  # 存储label或attribute的取值及其数量
    for line in subdataset:
        label = line[index]
        count[label] = count.get(label, 0) + 1
    result = 0.0
    for i in count.values():
        p = float(i) / size
        if p != 0:
            result -= p * math.log2(p)
            # result -= p * np.log2(p)
    return result


def cal_gini(dataset, attribute):
    """
    计算CART方法的指标gini。只计算对应属性的gini值
    :param dataset: 数据集
    :param attribute: 属性下标
    :return: 返回gini值
    """
    sub_attribute_count = {}    # 储存每个子属性的个数
    sub_attribute_label = {}    # 储存每个子属性所含的label及数量
    total = len(dataset)
    for line in dataset:
        sub_attribute = line[attribute]
        sub_attribute_count[sub_attribute] = sub_attribute_count.get(sub_attribute, 0)
        sub_attribute_count[sub_attribute] += 1     # 次数+1
        sub_attribute_label[sub_attribute] = sub_attribute_label.get(sub_attribute, {})     # get默认返回空的dictionary，然后赋值创建
        if line[-1] not in sub_attribute_label[sub_attribute]:
            sub_attribute_label[sub_attribute][line[-1]] = 0    # 将对应的label的次数赋值为0
        sub_attribute_label[sub_attribute][line[-1]] += 1

    gini = 0
    for i1 in sub_attribute_count.keys():
        size = sub_attribute_count[i1]   # sub_attribute的个数
        gini_temp = 1
        for value in sub_attribute_label[i1].values():
            gini_temp -= np.square(value / size)
        gini += size / total * gini_temp
    return gini


def get_sub_dataset(dataset, index, attribute):
    """
    从数据集中提取出某个属性取值相同的部分
    :param dataset: 数据集
    :param index: which 属性
    :param attribute: 属性取值
    :return: 返回子数据集
    """
    sub_dataset = []
    for record in dataset:
        if record[index] == attribute:
            sub_dataset.append(record)
    return sub_dataset


def choose_best_attribute(dataset, attribute_dic, available_attribute, strategy):
    """
    根据某一个指标（ID3,C4.5,CART）来选择最优的属性作为当前子树的划分标准
    :param dataset: 数据集
    :param attribute_dic: 属性字典
    :param available_attribute: 当前可以选择的属性集合，存的是属性在attribute_dic中对应的下标
    :param method: 指标
    :return:
    """
    if strategy == "ID3":
        data_size = len(dataset)    # 数据集的总长度
        empirical_entropy = cal_entropy(dataset, -1)  # 经验熵 empirical entropy
        info_gain_list = []     # 信息增益的数组
        for attribute in available_attribute:   # 这里存储的是下标，表示属性
            conditional_entropy = 0.0   # 条件熵
            for sub_attribute in attribute_dic[attribute]:  # 遍历这个属性对应的取值sub_attribute
                sub_dataset = get_sub_dataset(dataset, attribute, sub_attribute)    # 获得对应属性的子数据集
                p = len(sub_dataset) / data_size    # p(a)
                conditional_entropy += p * cal_entropy(sub_dataset, -1)     # 计算条件熵
            # 这里不用insert（insert要指明下标），而是用append加到list最后，因为attribute不是连续的
            info_gain_list.append(empirical_entropy - conditional_entropy)
        max_index = np.argmax(info_gain_list)  # 返回的是信息增益最大的下标
        # print(available_attribute[max_index])
        return available_attribute[max_index]   # 返回的是信息增益最大的属性的下标

    elif strategy == "C4.5":
        data_size = len(dataset)  # 数据集的总长度
        empirical_entropy = cal_entropy(dataset, -1)  # 经验熵 empirical entropy
        info_gain_ratio_list = []  # 信息增益率的数组
        for attribute in available_attribute:  # 这里存储的是下标，表示属性
            conditional_entropy = 0.0  # 条件熵
            for sub_attribute in attribute_dic[attribute]:  # 遍历这个属性对应的取值sub_attribute
                sub_dataset = get_sub_dataset(dataset, attribute, sub_attribute)  # 获得对应属性的子数据集
                p = len(sub_dataset) / data_size  # p(a)
                conditional_entropy += p * cal_entropy(sub_dataset, -1)  #
            split_info = cal_entropy(dataset, attribute)    # 计算特征attribute的信息熵
            if split_info == 0:     # 证明这个attribute对决策没贡献（取得都是相同的sub_attribute）
                continue
            # 这里不用insert（insert要指明下标），而是用append加到list最后，因为attribute不是连续的
            info_gain_ratio_list.append((empirical_entropy - conditional_entropy)/split_info)
        max_index = np.argmax(info_gain_ratio_list)  # 返回的是信息增益率最大的下标
        return available_attribute[max_index]

    elif strategy == "CART":
        gini_list = []
        for attribute in available_attribute:
            gini_temp = cal_gini(dataset, attribute)
            gini_list.append(gini_temp)
        min_index = np.argmin(gini_list)
        return available_attribute[min_index]


def create_tree(dataset, available_attribute, attribute_dic, parent_label, strategy):
    """
    构建决策树
    :param dataset: 数据集
    :param available_attribute: 可选属性
    :param attribute_dic: 属性字典
    :param parent_label: 父节点的label
    :param strategy: 选择属性时候的方法&指标
    :return: 根节点
    """
    label_list = [record[-1] for record in dataset]
    # 3个边界条件：
    # 条件3：dataset为空集,则取父节点的属性：这个要放在最前面！！cause有可能越界！！
    if len(dataset) == 0:
        return DecisionNode(label=parent_label)
    # 条件1：当dataset里面的样本都取同一label
    if label_list.count(label_list[0]) == len(label_list):
        return DecisionNode(label=label_list[0])
    # 条件2：当没有属性可选时，即available_atttribute为空时
    if len(available_attribute) == 0:
        label = max(label_list, key=label_list.count)  # 找出众数标签
        return DecisionNode(label=label)
    # 选择出最好的特征,返回的是attribute中对应的下标
    best_attribute = choose_best_attribute(dataset, attribute_dic, available_attribute, strategy)
    available_attribute.remove(best_attribute)
    branch = {}
    parent_label = max(label_list, key=label_list.count)     # 传给下一轮递归，实际不一定有意义
    for sub_attribute in attribute_dic[best_attribute]:     # 利用最好的属性对dataset进行划分，并构建子树
        sub_dataset = get_sub_dataset(dataset, best_attribute, sub_attribute)
        # available_attribute[:]表示传值！不加的话是传引用！
        # 如果函数收到的是一个可变对象(比如字典或者列表)的引用,就能修改对象的原始值--相当于通过“传引用”来传递对象。
        branch[sub_attribute] = create_tree(sub_dataset, available_attribute[:], attribute_dic, parent_label, strategy)
    return DecisionNode(label=parent_label, attribute=best_attribute, branch=branch)


def validation(valid_set, root):
    """
    返回预测准确率
    :param valid_set: 验证集
    :param root: 根节点
    :return: 准确率
    """
    cnt = 0     # 记录预测正确的个数
    for line in valid_set:
        cur = root
        while cur.branch is not None:
            cur = cur.branch[line[cur.attribute]]
        if cur.label == line[-1]:
            cnt += 1
    return cnt/len(valid_set)


def valid_predict():
    dataset, label_set = readfile('car_train.csv')
    x = []
    y = []
    strategy = "C4.5"
    for k in range(3, 20):  # 交叉验证
        temp = 0
        for i in range(k):  # 对不同的验证集取均值作为一个k的结果
            train_set, valid_set = k_fold(dataset, k, i)
            attribute_dic = get_attribute_dic(dataset, label_set)
            available_attribute = list(range(0, len(label_set) - 1))
            root = create_tree(train_set, available_attribute, attribute_dic, -1, strategy)
            temp += validation(valid_set, root)
        print("利用%s方法，对数据集进行%s折划分后的准确率为%s" % (strategy, k, temp / k))
        x.append(k)
        y.append(temp / k)
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('%s' % strategy)
    plt.show()


def test_split_dataset(dataset):
    train_set = dataset[:15]    # 左闭右开，即不包括15。且是从0开始的！0~14是train_set，15~17是test_set
    test_set = dataset[15:]
    # print(test_set)
    return train_set, test_set


def predict(test_set, root):
    """
    验收数据的预测
    :param test_set: 验收数据待预测的样本
    :param root: 根节点
    :return: 输出预测结果，无返回值
    """
    for line in test_set:
        cur = root
        while cur.branch is not None:
            cur = cur.branch[line[cur.attribute]]
        print(cur.label)


if __name__ == "__main__":
    # valid_predict()     # 这是验证集预测，得准确率
    # 下面的是跑验收数据
    dataset, label_set = readfile('DecisionTree验收数据.csv')
    strategy = "C4.5"
    train_set, test_set = test_split_dataset(dataset)
    attribute_dic = get_attribute_dic(dataset, label_set)
    available_attribute = list(range(0, len(label_set) - 1))
    root = create_tree(train_set, available_attribute, attribute_dic, -1, strategy)
    predict(test_set, root)


