import numpy as np
from matplotlib import pyplot as plt
# import tqdm
import random
import time

def readData(filename):
    file = np.load(filename)
    return file


def shuffle_data(train_images, train_labels):
    # 随机打乱训练集
    data_class_list = list(zip(train_images, train_labels))
    random.shuffle(data_class_list)
    train_images, train_labels = zip(*data_class_list)  # 解压

    return np.array(list(train_images)), np.array(list(train_labels))


def preprocess_data(class_num, train_images, train_labels, test_images, test_labels):
    # 给images加一列：用于b
    train_images = train_images.T
    train_images = np.insert(train_images, 0, np.ones(train_images.shape[1]), axis=0)
    train_images = train_images.T
    test_images = test_images.T
    test_images = np.insert(test_images, 0, np.ones(test_images.shape[1]), axis=0)
    test_images = test_images.T
    # 拓展labels：转成one-hot形式
    onehot_train_label = np.eye(class_num)[train_labels]  # [样例数，类别数]
    onehot_test_label = np.eye(class_num)[test_labels]
    return train_images, onehot_train_label, test_images, onehot_test_label


def init_w(choice, row, col):
    '''
    :param choice: 初始化方式：0-全零初始化；1-随机初始化；2-xavier Glorot normal
    :param row: 行数
    :param col: 列数
    :return: 初始化后的权重向量w
    '''
    w = []
    if choice == 0:            # 全零初始化
        w = np.zeros([row, col])
    elif choice==1:            # 随机初始化
        w = np.random.uniform(low=-1.0, high=1.0, size=[row, col])
    elif choice==2:            # xavier Glorot normal
        w = np.random.normal(loc=0., scale=1, size=[row, col]) / np.sqrt(col)
    return w


def sigmoid(x):
    result = 0.5*(1+np.tanh(0.5*x))
    return result


def activation_func(func_type, x):
    if func_type == "sigmoid":
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
        temp = hidden_output.copy()     # 深拷贝numpy
        temp[temp < 0] = 0.01
        temp[temp >= 0] = 1
        return temp


def softmax(x):     #2维输入，对每一行做softmax
    max_x_row=np.max(x,axis=1)
    max_x_row=np.repeat(np.reshape(max_x_row, (-1,1)), x.shape[1], axis=1)
    #print(max_x_row.shape)
    exp_result = np.exp(x - max_x_row)

    sum_temp = np.sum(exp_result,axis=1)
    result = exp_result/sum_temp[:, None]
    #print(sum_temp.shape)
    return result


def forward_pass(w_hidden, w_output, b_output, dataset, activate_choice):
    '''
    前向传播
    :param w_hidden: 输入层-隐藏层参数w
    :param w_output: 隐藏层-输出层参数w
    :param b_output: 隐藏层-输出层偏置b
    :param dataset: 数据集
    :param activate_choice: 激活函数选择: "sigmoid", "tanh", "relu"
    :return: 隐藏层输出和输出层输出
    '''
    # 隐藏层计算
    hidden_input = np.dot(dataset, w_hidden)
    hidden_output = activation_func(activate_choice, hidden_input)
    # 输出层计算
    data_num = dataset.shape[0]
    b_output1 = np.repeat(b_output, data_num,axis=0)    # 沿着行扩展成[N,10]
    output_input = np.dot(hidden_output, w_output) + b_output1
    output_output = softmax(output_input)
    return hidden_output, output_output


def cal_accuracy(output, real_result):
    '''
    计算准确率
    :param output: 输出层输出，[N,10]
    :param real_result: 输出层真实值, [N,]
    :return:
    '''
    predict_result = np.argmax(output, axis=1)      # 选出预测值（max one）,[60000, ]
    temp = predict_result - real_result
    true_count = predict_result.shape[0]-np.count_nonzero(temp)     # 减去非零值！妙！
    return true_count/predict_result.shape[0]


def cross_entropy_loss(output_output, onehot_labels):
    data_num = output_output.shape[0]
    log_result = np.log(output_output+1e-8)
    # log_reuslt,onehot相乘，是点乘，（两层sum所以不用加axis）
    loss = -1 / data_num * np.sum(onehot_labels * log_result)
    return loss


def derivative_CE(output_output, onehot_labels, hidden_output):
    '''
    :param output_output: 输出层输出
    :param onehot_labels: 数据集labels
    :param hidden_output: 隐藏层output
    :return: 输出层error和cross-entropy导数
    '''
    data_num = output_output.shape[0]
    softmax_result = output_output - onehot_labels
    result = np.dot(hidden_output.T, softmax_result) / data_num
    return softmax_result, result


def backward_pass(hidden_output, output_output, w_hidden, w_output, b_output, activation_choice, dataset, onehot_labels, lr):
    # 计算输出层误差，输出层梯度
    output_error, output_grad = derivative_CE(output_output, onehot_labels, hidden_output)    # [N,10] [hidden_num,10]
    # print(output_grad.shape)
    # 计算隐藏层误差
    hidden_error = derivation(activation_choice, hidden_output) * np.dot(output_error, w_output.T)
    # 更新隐藏层-输出层参数：w_output
    w_output = w_output - lr * output_grad
    output_error_mean = np.mean(output_error, axis=0)   # 算出output_error均值：从[N,10]变成[1,10]
    b_output = b_output - lr * output_error_mean
    # 更新输入层-隐藏层参数：w_hidden
    w_hidden = w_hidden - lr * np.dot(dataset.T, hidden_error) / dataset.shape[0]
    return w_hidden, w_output, b_output


def train(train_images, train_labels, test_images, test_labels, train_onehot_labels, test_onehot_labels, w_hidden, w_output, b_output, activation_choice, lr, iteration_number, batch_size):
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    data_num = train_images.shape[0]
    # for i in tqdm.tqdm(range(iteration_number)):

    for i in range(iteration_number):
        train_acc=0
        train_loss=0
        for j in range(data_num//batch_size):
            train_images_batch = train_images[j*batch_size: (j+1)*batch_size]
            train_labels_batch = train_labels[j*batch_size: (j+1)*batch_size]
            train_onehot_labels_batch = train_onehot_labels[j*batch_size: (j+1)*batch_size]
            # 前向传播
            train_hidden_output, train_output_output = forward_pass(w_hidden, w_output, b_output, train_images_batch, activation_choice)

            # 计算train和test的正确率
            train_acc += cal_accuracy(train_output_output, train_labels_batch)
            # 计算误差
            train_loss += cross_entropy_loss(train_output_output, train_onehot_labels_batch)

             # 后向传播
            w_hidden, w_output, b_output = backward_pass(train_hidden_output, train_output_output, w_hidden, w_output,b_output,
                                           activation_choice, train_images_batch, train_onehot_labels_batch, lr)
        # 运行测试集结果
        _, test_output_output = forward_pass(w_hidden, w_output, b_output, test_images, activation_choice)
        test_acc = cal_accuracy(test_output_output, test_labels)
        test_loss = cross_entropy_loss(test_output_output, test_onehot_labels)
        #print(w_hidden)
        train_acc_list.append(train_acc/(data_num//batch_size))
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss/(data_num//batch_size))
        test_loss_list.append(test_loss)
        print("第%s次迭代：train_acc=%s, test_acc=%s" % (i, train_acc/(data_num//batch_size), test_acc))
        print("第%s次迭代：train_loss=%s, test_loss=%s" % (i, train_loss/(data_num//batch_size), test_loss))
        # with open("temp.txt",'a') as file:
        #     file.write('%d %.4f %.4f\n' %(i, train_acc, train_loss))
        #     file.write('%d %.4f %.4f\n' % (i, test_acc, test_loss))
    return train_acc_list, test_acc_list, train_loss_list, test_loss_list


def draw_loss(loss_train, loss_test):
    count = [i for i in range(len(loss_train))]
    plt.plot(count, loss_train, label='train')
    plt.plot(count, loss_test, label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss of Train_set and Test_set')
    plt.legend()  # 显示图例
    plt.show()
    return
    # plt.savefig('loss_mse4.jpg')

def draw_acc(acc_train, acc_test):
    count = [i for i in range(len(acc_train))]
    plt.plot(count, acc_train, label='train')
    plt.plot(count, acc_test, label='test')
    plt.title('Accuracy of Train_set and Test_set')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()  # 显示图例
    plt.show()
    return
    # plt.savefig('acc_mse4.jpg')


if __name__ == "__main__":
    train_images = readData("Dataset/train-images.npy")     # [N, 784]
    train_labels = readData("Dataset/train-labels.npy")     # [N, ]
    test_images = readData("Dataset/test-images.npy")
    test_labels = readData("Dataset/test-labels.npy")

    # 数据预处理
    class_num = 10  # 多分类种类数：0-9
    train_images, train_labels = shuffle_data(train_images, train_labels)
    train_images, train_onehot_labels, test_images, test_onehot_labels = preprocess_data(class_num, train_images, train_labels, test_images, test_labels)
    # 初始化参数
    t1 = time.time()
    feature_num = train_images.shape[1]     # 数据的特征数：785
    hidden_num = 150     # 隐藏层节点数
    # 参数1
    w_hidden = init_w(choice=2, row=feature_num, col=hidden_num)
    w_output = init_w(choice=2, row=hidden_num, col=class_num)
    b_output = init_w(choice=2, row=1, col=class_num)
    # 训练：前向传播+后向传播
    # relu:0.0005; sigmoid: 0.5; tanh:0.5
    activation_choice = "sigmoid"
    lr = 0.8
    iteration_number = 200
    batch_size = 60000
    train_acc_list, test_acc_list, train_loss_list, test_loss_list = train(train_images, train_labels, test_images,
        test_labels, train_onehot_labels, test_onehot_labels, w_hidden, w_output, b_output, activation_choice, lr, iteration_number, batch_size)
    t2 = time.time()
    print(t2-t1)

    # 画图
    draw_loss(train_loss_list, test_loss_list)
    draw_acc(train_acc_list, test_acc_list)
    f = open("result/test_result14.txt", "w")
    for i in range(len(train_acc_list)):
        f.write("%.6f " % train_acc_list[i])
        f.write("%.6f \n" % train_loss_list[i])
        f.write("%.6f " % test_acc_list[i])
        f.write("%.6f \n" % test_loss_list[i])
    f.close()




