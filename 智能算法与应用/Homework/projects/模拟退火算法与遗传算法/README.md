# README

### 局部搜索算法和模拟退火算法

#### IDE

pycharm

#### 实验环境

`matplotlib`，`numpy`

#### 文件组织

- `Simulated Annealing Algorithm.py`：主文件，运行此文件可得到效果图和输出提示。
- `util.py`：包含了读文件、画图、局部搜索算法等公用函数。

#### 运行说明

**输入**：运行`Simulated Annealing Algorithm.py`文件，在main函数中可以：

- 修改`method`为“SA”或“Local Search only"，表示使用的策略；
- 修改`choice`为1-5，表示局部搜索策略的选择；
- 修改`TEST`为1或2，表示运行的测试样例。

**输出**：得到遍历完后的最优路径图像、官网给出的最优路径图像及两者的对比。并得到运行时间，当前最优路径的大小及结果。



### 种群算法

#### IDE

pycharm

#### 实验环境

`matplotlib`，`numpy`

#### 文件组织

- `simple genetic algorithm.py`：主文件，运行此文件可得到效果图和输出提示。
- `util.py`：包含了读文件、画图、局部搜索算法等公用函数。
- `SGA_setting.py`：参数设置文件

#### 运行说明

**输入**：

在`simple genetic algorithm.py`文件，在主函数中，可以按照提示输入两个测试样例，详情请看主函数中的注释

在`SGA_setting.py`中，可以根据提示，修改相关参数，比如种群大小，迭代次数等等

**输出**：得到遍历完后的最优路径图像、官网给出的最优路径图像及两者的对比，最优路径随时间变化的曲线图。并得到运行时间，当前最优路径的大小。

（这个代码运行时间较长，请耐心等待）