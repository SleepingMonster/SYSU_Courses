# README

### 利用联合方法求解VRP问题

#### IDE

JupyterLab（CUDA环境）

#### 实验环境

`matplotlib`，`numpy`，`pytorch`

#### 文件组织

- `VRP.py`：主文件，包括了所有的函数。

#### 运行说明

**输入**：直接运行`VRP.py`文件

**输出**：训练集和测试集中loss和最短平均路径随迭代次数的变化图，并且会输出到文件`result.txt`，输出训练集和测试集上最后一个batch的路径（每5次循环输出一次），分别输出到文件`pi_train.txt`和`pi_test.txt`中

