# 多媒体技术 作业二

> 姓名：TRY
>
> 学号：
>
> 日期：2021/6/7



## 第一题

![image-20210525002833974](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210525002833974.png)

解：

#### (a) 

- **哈夫曼编码**需要有关信息源的先验统计知识，而这样的信息通常很难获得。这在多媒体应用中表现尤为突出。在多媒体应用中，数据在到达之前是未知的，例如直播（或流式）的音频和视频。即使能够获得这些统计数字，**符号表的传输**仍然是相当大的一笔**开销**。
- **自适应哈夫曼编码**的统计数字是随着数据流的到达而动态地收集和更新的。概率**不再是基于先验知识**，而是基于到目前为止实际收到的数据。随着接收到的符号的概率分布的改变，符号将会被赋予新的（更长或更短的）码字。而且，自适应哈夫曼编码是**动态**的，因此可以提供**更好的压缩效果**，并**节省符号表的传输所需的开销**。



#### (b)(i)

接收到的字符是：`b(01)a(01)c(00 10)c(101)`

推导过程如下：

编码串为：`01010010101`

初始，自适应哈夫曼树和各字符的编码&计数如下：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526003822617.png" alt="image-20210526003822617" style="zoom:67%;" />

| 符号 | a    | b    | c    | d    | NEW  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 计数 | 2    | 2    | 0    | 0    | 0    |
| 编码 | 1    | 01   | 10   | 11   | 00   |

- 首先，按照上面的表格，匹配接收到字符`b(01)`。此时，自适应哈夫曼树如下：

  | b计数变为3                                                   | b和a发生交换                                                 |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526004354140.png" alt="image-20210526004354140" style="zoom:67%;" /> | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526004433610.png" alt="image-20210526004433610" style="zoom:67%;" /> |

  各字符的编码&计数如下：

  | 符号 | a    | b    | c    | d    | NEW  |
  | ---- | ---- | ---- | ---- | ---- | ---- |
  | 计数 | 2    | 3    | 0    | 0    | 0    |
  | 编码 | 01   | 1    | 10   | 11   | 00   |

- 按照上面的表格，匹配接收到字符`a(01)`。此时，自适应哈夫曼树如下：

  <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526004730402.png" alt="image-20210526004730402" style="zoom:67%;" />

  各字符的编码&计数如下：

  | 符号 | a    | b    | c    | d    | NEW  |
  | ---- | ---- | ---- | ---- | ---- | ---- |
  | 计数 | 3    | 3    | 0    | 0    | 0    |
  | 编码 | 01   | 1    | 10   | 11   | 00   |

- 按照上面的表格，匹配接收到字符`NEW(00)`，代表接下来会接收一个新字符。

  按照上面的表格，匹配接收到字符`c(10)`。此时，自适应哈夫曼树如下：

  | 添加新字符c                                                  | 交换左右子树                                                 |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526005207661.png" alt="image-20210526005207661" style="zoom:67%;" /> | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526005218662.png" alt="image-20210526005218662" style="zoom:67%;" /> |

  各字符的编码&计数如下：

  | 符号 | a    | b    | c    | d    | NEW  |
  | ---- | ---- | ---- | ---- | ---- | ---- |
  | 计数 | 3    | 3    | 1    | 0    | 0    |
  | 编码 | 11   | 0    | 101  | 11   | 100  |

- 按照上面的表格，匹配接收到字符`c(101)`。此时，自适应哈夫曼树如下：

  <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526005500645.png" alt="image-20210526005500645" style="zoom:67%;" />

  各字符的编码&计数如下：

  | 符号 | a    | b    | c    | d    | NEW  |
  | ---- | ---- | ---- | ---- | ---- | ---- |
  | 计数 | 3    | 3    | 2    | 0    | 0    |
  | 编码 | 11   | 0    | 101  | 11   | 100  |

因此，接收到的字符串为**`bacc`**。



#### (b)(ii)

接收每一个字符后的自适应哈夫曼树如下：

> 推导过程与`(b)(i)`相同，此处省略。

| 接收b(01)                                                    | 接收a(01)                                                    | 接收c(00 10)                                                 | 接收c(101)                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="file://C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526004433610.png?lastModify=1621961933" alt="image-20210526004433610"  /> | ![image-20210526004730402](file://C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526004730402.png?lastModify=1621961933) | <img src="file://C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526005218662.png?lastModify=1621961933" alt="image-20210526005218662" style="zoom: 80%;" /> | <img src="file://C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526005500645.png?lastModify=1621961933" alt="image-20210526005500645" style="zoom:80%;" /> |



## 第二题

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210526010135748.png" alt="image-20210526010135748" style="zoom: 80%;" />



### 2.1 理论原因分析部分

- **GIF压缩**：采用无损压缩技术，如果图像不多于 256 色，则既可以减少文件的大小又可以保持图片质量，普遍用于图标按钮等只需要少量颜色的图像，如黑白图。`GIF`压缩是将图片转换成 256 色进行传输，传输方发送颜色对应的索引值（如果双方没有统一索引表，则连同索引表一起传输）。
- **JPEG压缩**：采用有损压缩技术，通过对图像进行色彩空间转换（`RGB`转换到`YCbCr`），二次采样，`DCT`，量化，熵编码等过程来对图像进行压缩，不适用于线条绘图和其他文字和图标的图形。具体来说，`JPEG` 应用 `DCT` 减少高频内容，更有效地将结果保存为位串。`JPEG`压缩使用了人类对灰度视觉敏感度的原理，对亮度进行细量化，色度进行粗量化，显示效果较好。而且，`JEPG`将每个 8*8 的块进行压缩并编码成大小不一的数据流，并传输，接收方接收数据流并根据 `huffman`表和量化表进行还原（如果双方表没有统一，也需要连同图片一起传输）。 
- 综上所述：**GIF** 适用于颜色数量少、图像细节不明显的图像，原因在于其采用的是 `LZW` 压缩算法，因为颜色数较少，`LZW` 的压缩率较高，**速度快**，因此更适合本题中卡通图片的压缩（卡通图片颜色比较单一、色彩较少；而`JPEG`保留了24位的颜色信息，没有必要且占用了内存）；而 **JPEG** 压缩在颜色、细节多的图像中的压缩效果会比 `GIF` 好，压缩比也相应较高，因此更适用于本题中的动物图片。 



### 2.2 程序实现部分

#### 2.2.1 实验环境

- 编程语言：Python 3.8

- 本地IDE： pycharm / jupyter notebook

- 文件组织：`jepg_compression.py` 和 `util.py`。前者包含了jepg压缩的主要模块，直接运行就可得到自己实现的jpeg压缩的图片。后者包含了实验中用到的量化矩阵和huffman编码表。



#### 2.2.2 JPEG压缩算法描述&实现

​		`JPEG`压缩编码算法一共分为 9 个步骤：颜色空间转换（`RGB`转化为`YCbCr`）、padding操作、二次采样、分块、离散余弦变换`(DCT)`、量化、`Zigzag` 扫描排序、DC和AC编码（包括DC 系数的DPCM编码和AC 系数的游程长度编码）、DC系数和AC系数的熵编码。

​		而由于要看压缩的正确与否，需要实现相应的逆操作，如逆熵编码、逆DC和AC编码，反量化、IDCT、拼接、逆采样。

​		以下分模块进行实现的说明（包括其逆操作）。

------

##### 1. 读图片

使用`numpy`函数和PIL库函数来实现图片的读取和解压（本来的jpg格式就是压缩过的，打开图片后电脑自动解压并显示）。

```python
def readImage(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image
```



##### 2. 颜色空间转化

JPEG采用的是`YCbCr`颜色空间，因此需要将原来的RGB空间转化到`YCbCr`空间。其中，`Y`表示亮度，`Cr`、`Cb`表示色度。

而在书本P71页，可知两者的转换公式为：
$$
\begin{bmatrix}
Y\\C_b\\C_r
\end{bmatrix}=
\begin{bmatrix}
0.299 & 0.587 & 0.114\\
-0.168736 & -0.331264 & 0.5\\
0.5 & -0.418688 & -0.081312
\end{bmatrix}
\begin{bmatrix}
R\\G\\B
\end{bmatrix}+
\begin{bmatrix}
16\\128\\128
\end{bmatrix}
$$
对矩阵求逆，得
$$
\begin{bmatrix}
R\\G\\B
\end{bmatrix}=
\begin{bmatrix}
1 & 0 & 1.402\\
1 & -0.331264 & -0.714136\\
1 & 1.772 & 0
\end{bmatrix}
\begin{bmatrix}
Y-16\\C_b-128\\C_r-128
\end{bmatrix}
$$
而且，由于需要返回[0,255]的数，故需要对不在范围的数进行手动转换。如将小于0的数赋值为0，大于255的数赋值为255。

因此，RGB转化为YCbCr的函数为：

```python
# RGB转化为YCbCr，返回[0,255]范围的数（ycbcr的要求）
def rgb2ycbcr(image):
    args = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    ycbcr = image.dot(args.T)
    ycbcr[:, :, [0]] += 16
    ycbcr[:, :, [1,2]] += 128
    # 手动将这个设置为[0,255]范围，uint8是不可以的（原理不同）
    ycbcr[:, :, [0]][ycbcr[:, :, [0]]<0] = 0
    ycbcr[:, :, [0]][ycbcr[:, :, [0]]>255] = 255
    ycbcr[:, :, [1,2]][ycbcr[:, :, [1,2]]<0] = 0
    ycbcr[:, :, [1,2]][ycbcr[:, :, [1,2]]>255] = 255
    return ycbcr
```

其逆操作为：

```python
def ycbcr2rgb(ycbcr):
    args = np.array([[1, 0, 1.402], [1, -0.344135, -0.714136], [1, 1.772, 0]])
    ycbcr[:, :, [0]] -= 16
    ycbcr[:, :, [1,2]] -= 128
    image = ycbcr.dot(args.T)
    # 这里也要手动设置[0,255]的范围！！因为uint8这个函数就不对！
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j,0]=min(255, max(0, round(image[i,j,0])))
            image[i,j,1]=min(255, max(0, round(image[i,j,1])))
            image[i,j,2]=min(255, max(0, round(image[i,j,2])))
            
    # 但这里要调用uint8函数，不然无法还原图片！（Image库函数要求调用uint8！）
    return np.uint8(image)
```



##### 3. padding操作

​		由于图片的长宽都需要满足是 16 的倍数（二次采样长宽会缩小 1/2 ，量化时长宽会缩小 1/8），所以需要将图片填充至 16 的倍数。故需要进行padding，在图片的右边和下边填充0即可。

- 不在左边和上边填充的原因：二次采样时采样点在左上角，需保证采样的信息是原图片的信息。

```python
def padding(image):
    row = image.shape[0]
    col = image.shape[1]
    row1 = row
    col1 = col
    fill_size0 = 16 - row % 16  # row要扩展的像素数
    fill_size1 = 16 - col % 16  # col要扩展的像素数
    if fill_size0 != 16:
        row1 = row + fill_size0
    if fill_size1 != 16:
        col1 = col + fill_size1
    new_image = np.zeros((row1, col1, 3), dtype=float)
    new_image[:row, :col] = image
    return new_image

# 去掉padding
def de_padding(image, new_image):
    row = image.shape[0]
    col = image.shape[1]
    result = np.array(new_image[:row, :col])
    print(result.shape)
    return result
```



##### 4. 色度二次采样

​		人类的眼睛对于亮度差异的敏感度高于色彩变化，因此可以认为 `Y` 分量要比 `Cb,Cr` 分量重要的多。本代码中，对于每个 2*2 的块进行采样，比例是`Y:Cb:Cr=4:2:0`, 经过采样处理后，每个单元中的值分别有 4 个 `Y`、1 个 `Cb`、1 个 `Cr`。其中，`Cr`取左上角，`Cb`取左下角的值。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606200132314.png" alt="image-20210606200132314" style="zoom: 67%;" />

```python
# 色度二次采样
def sampling(image, row, col):
    samp_y, samp_cb, samp_cr = [], [], []  # 先写成一维数组，再reshape成二维的
    for i in range(row):
        for j in range(col):
            samp_y.append(image[i,j,0])
            if i%2==0 and j%2==0:
                samp_cr.append(image[i,j,2])
            elif j%2==0:
                samp_cb.append(image[i,j,1])
    samp_y = np.array(samp_y).reshape((row,col))  
    samp_cb = np.array(samp_cb).reshape((row//2, col//2))   
    samp_cr = np.array(samp_cr).reshape((row//2, col//2))
    return samp_y, samp_cb, samp_cr

# 逆色度二次采样：重构(row, col)大小的image
def inverse_sampling(samp_cb, samp_cr, row, col):
    src_cb, src_cr = np.zeros((row, col), dtype=float), np.zeros((row, col), dtype=float)
    for i in range(samp_cb.shape[0]):
        for j in range(samp_cb.shape[1]):
            cb, cr = samp_cb[i][j], samp_cr[i][j]
            src_cb[i*2:i*2+2, j*2:j*2+2] = np.array([cb, cb, cb, cb]).reshape((2,2))
            src_cr[i*2:i*2+2, j*2:j*2+2] = np.array([cr, cr, cr, cr]).reshape((2,2))
    return src_cb, src_cr
```



##### 5. 分块

​	在量化前，需要对图像进行8*8的分块操作。之后，再将图像进行组合。

```python
# 切分图像成8*8的块
def split_blocks(image, block_size=8):
    row_blocks = np.split(image, image.shape[0]/block_size)  # 先按行切分
    result = []  # 得到的是二维的[[首8行中的所有block],[]]
    for block in row_blocks:
        result.append(np.split(block, image.shape[1]/block_size, axis=1))
    result = np.array(result)
    return result

def combine_blocks(blocks, row, col):
    image = np.zeros((row, col))
    indices = [(i,j) for i in range(0, row, 8) for j in range(0, col, 8)]
    for block, index in zip(blocks, indices):
        i, j = index
        image[i:i+8, j:j+8] = block
    return image
```



##### 6. DCT操作

​		针对 N\*N 的像素块逐一进行 DCT 操作。JPEG 的编码过程需要进行`DCT`（离散余弦变换），而解码过程则需要`iDCT`（逆离散余弦变换）。8\*8 的二维像素块经过 DCT 操作之后，就得到了 8*8 的变换系数矩阵。

​		根据书本P157页可知，`DCT`变换为：
$$
F(u,v) =\frac{C(u)C(v)}{4}\sum_{i=0}^7\sum_{j=0}^7 cos\frac{(2i+1)u\pi}{16}cos\frac{(2j+1)v\pi}{16}f(i,j)\\
其中,i,j,u,v=0,1,...,7，常数C(u)，C(v)由下式给出：\\
C(\xi)=\begin{cases} \frac{\sqrt{2}}{2}，\xi=0\\1，其他\end{cases}
$$
​		`iDCT`变换为：
$$
\widetilde f(i,j)=\sum_{u=0}^7\sum_{v=0}^7\frac{C(u)C(v)}{4} cos\frac{(2i+1)u\pi}{16}cos\frac{(2j+1)v\pi}{16}F(u,v)
$$
​		而由矩阵的特点可知，可将上述两级求和改成矩阵相乘。因此，需先求出 DCT系数矩阵，然后进行 DCT变换。

```python
# 得到block_size=8时的dct系数矩阵（8*8）(u,i)或(v,j)
def get_dct_matrix():
    dct_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i==0:
                dct_matrix[i][j] = 1/np.sqrt(8)
            else:
                dct_matrix[i][j] = 1/2 * math.cos((2*j+1)*i*math.pi/16)
    return dct_matrix
    
# DCT操作
def dct(block_matrix, dct_matrix):
    dct_matrix = np.dot(np.dot(dct_matrix, block_matrix), dct_matrix.T)
    return dct_matrix

# 逆dct操作
def inverse_dct(block_matrix, dct_matrix):
    idct_matrix = np.dot(np.dot(dct_matrix.T, block_matrix), dct_matrix)
    return idct_matrix
```



##### 7. 量化

​		图像数据转换为 `DCT` 系数 $F(u,v)$ 之后，进入量化。量化阶段需要两个 8*8 量化矩阵数据，一个是“亮度量化矩阵”，另一个是“色度量化矩阵”，将`DCT`系数除以量化矩阵的值之后取整（四舍五入），即完成了量化过程。
$$
\widehat{F}(u,v)=round(\frac{F(u,v)}{Q(u,v)})\\
其中，F(u,v)表示DCT系数，Q(u,v)是量化矩阵，\widehat{F}(u,v)表示量化后的DCT系数
$$
​		不难发现，这一部分会丢失数据内容。因此，也是`JPEG`压缩中产生信息丢失的主要原因。对于 `Y` 通道使用亮度量化表，为前者细量化，对于 `Cb,Cr` 采用色度量化表，为粗量化。
​		而且，可以给量化矩阵 $Q(u,v)$ 乘以比例值来**改变压缩率**。设质量因子$qf\in[1,100]$，当$qf=100$ 时得到最高质量的压缩图像，当$qf=1$ 时得到最低质量的压缩图像。而scaling factor为量化矩阵乘取的比例值。

```python
# 量化
def quantization(dct_coefficient, quantization_matrix, qf):
    if qf>=50:
        scaling_factor = (100-qf)/50
    else:
        scaling_factor = 50/qf
    if scaling_factor!=0:
        qx = np.round(quantization_matrix * scaling_factor)
    else:
        qx=np.ones((8,8))
    qx = np.uint8(qx)
    # print(qx)
    result = np.round(dct_coefficient/qx)
    return result

# 反量化
def de_quantization(dct_coeff_quan, quantization_matrix, qf):
    if qf>=50:
        scaling_factor = (100-qf)/50
    else:
        scaling_factor = 50/qf
    # print(scaling_factor)
    if scaling_factor!=0:
        qx = np.round(quantization_matrix * scaling_factor)
    else:
        qx = np.ones((8,8))
    qx = np.uint8(qx)
    dct_coefficient = dct_coeff_quan * qx
    return dct_coefficient
```

 

##### 8. Z字形扫描

​		Z字形扫描是AC系数进行游程编码时需要的操作，是有规律可循的：当 $i+j$ 为偶数时，从左下往右上走；当 $i+j$ 为奇数时，从右上往左下走。且遇到边缘时进行平移。因此，代码如下：

```python
# Z字形扫描，用于游程编码
def zigzag_scanning(dct_coeff_quan):
    result = []
    n = 8
    i = 0  # row
    j = 1  # col
    while i<n and j<n:
        result.append(dct_coeff_quan[i,j])
        if (i+j) %2 == 0:
            if i>0 and j!=n-1:
                i-=1
                j+=1
            elif j==n-1:
                i+=1
            else:
                j+=1
        elif (i+j)%2 == 1:
            if j>0 and i!=n-1:
                i+=1
                j-=1
            elif i==n-1:
                j+=1
            else:
                i+=1
    return np.array(result)
```

其逆操作类似，也是按照同样的规律填充8*8矩阵。



##### 9. 对DC和AC系数进行编码

​		DC系数是8*8图像块的第一个值（左上角元素值），而剩下的63个分量都是AC系数。

​		对于AC系数进行游程编码（RLC）。将 $\widehat{F}$ 的AC系数中的每一个0串用$(RUNLENGTH,VALUE)$的数字对方式表示值，其中前者表示串里0的数目，后者表示下一个非0系数。

​		对于DC系数进行DPCM编码。DPCM编码只需要对整个图像执行一次。

```python
# 对DC和AC系数进行DPCM编码和游程编码
def DC_AC_coding(blocks):
    coding_results = []
    for i in range(len(blocks)):
        mid_result = []
        # 在第一个位置放入DC系数的DPCM编码
        if i==0:
            mid_result.append(blocks[i][0][0])
        else:
            mid_result.append(blocks[i][0][0] - blocks[i-1][0][0])
        # Z字形扫描
        ac_vector = zigzag_scanning(blocks[i])
        # AC系数的游程编码
        mid_result.extend(RLC(ac_vector))
        # 此时，mid_result = [dc, (ac_tuple),(ac_tuple),...]
        coding_results.append(mid_result)
    return coding_results

# 逆DC, AC编码
def de_DC_AC_coding(coding_results):
    result_blocks = []
    for i in range(len(coding_results)):
        dc_ac_coding = coding_results[i]  # [dc, (ac_tuple),(ac_tuple),...]
        mid_result = []
        # 恢复dc
        if i==0:
            mid_result.append(dc_ac_coding[0])
        else:
            mid_result.append(dc_ac_coding[0] + result_blocks[i-1][0][0])  # 累加！！
        # 恢复ac
        for j in range(1, len(dc_ac_coding)):
            runlength, value = dc_ac_coding[j]
            # 若果是最后的(0,0)
            if runlength==0 and value==0:
                temp = len(mid_result)
                for k in range(temp, 64):
                    mid_result.append(0)
                break
            # 前面的ac编码
            for m in range(runlength):
                mid_result.append(0)
            if runlength!=15 or value!=0:   # (15,0)的话就不需要再加0了
                mid_result.append(value)
        if len(mid_result)!=64:
            print ("the %d block number error %d" %(i, len(mid_result)))
        
        # 逆Z扫描
        mid_result = de_zigzag(mid_result)
        result_blocks.append(mid_result)
    return result_blocks
```



##### 10. 熵编码

​		最后对DC系数和AC系数进行熵编码。

​		对DC系数来说，将其用$(SIZE, AMPLITUDE)$表示。其中前者表示需要用多少位来表示DC系数，后者表示实际使用的位数。且对前者$SIZE$进行huffman编码。

​		对于AC系数来说，将数字对$(RUNLENGTH,VALUE)$中的$VALUE$用$(SIZE,AMPLITUDE)$表示，并将其拼接成如下两个symbol：
$$
Symbol\ 1:(RUNLENGTH, SIZE)\\
Symbol\ 2:AMPLITUDE\\
其中，Symbol\ 1使用huffman编码
$$
​		而huffman编码则使用了默认的huffman表（4个表，对应色度/亮度的DC/AC系数）进行编码。	

>  **值得注意的是**：
>
> - 如果DPCM为负数，则用其反码表示。但由于`-1`和`0`的表示都是“0”（-1的反码为0），无法区分。故在代码中，我将`0`表示是为(0,0)，`-1`表示为(1,0)。
> - 同时，对于特殊扩展编码(15,0)，不转换成(15,1)和0，直接进行huffman编码，可以节省空间（因为(15,0)的huffman编码明显短于(15,1)）。

​		因此，熵编码的代码如下：

```python
# 熵编码
def entropy_coding(coding_results, is_Y):
    result = []
    for i in range(len(coding_results)):
        coding = coding_results[i]
        # DC系数的huffman编码
        dc = coding[0]
        dc_amplitude, is_zero = dec2bin(dc)
        dc_size = len(dc_amplitude)
        if is_zero is True:   # 由于0和-1的二进制表示都是0，所以需要设置size=0来辅助判断这是0还是-1
            dc_size = 0
        dc_huffman_code = ""
        if is_Y is True:
            dc_huffman_code = DC_Luminance_Huffman[dc_size]
        else:
            dc_huffman_code = DC_Chroma_Huffman[dc_size]
        entropy_code = dc_huffman_code + dc_amplitude
        for j in range(1, len(coding)):
            ac_runlength, ac_value = coding[j]
            ac_huffman_code = ""
            # 判断是不是(15,0)。如果是则不用扩展value（实验中有(15,0)）
            if ac_runlength==15 and ac_value==0:
                if is_Y is True:
                    ac_huffman_code = AC_Luminance_Huffman[(15,0)]
                else:
                    ac_huffman_code = AC_Chroma_Huffman[(15,0)]
                entropy_code += ac_huffman_code
            else:
                ac_amplitude, _ = dec2bin(ac_value)
                ac_size = len(ac_amplitude)
                if is_Y is True:
                    ac_huffman_code = AC_Luminance_Huffman[(ac_runlength, ac_size)]
                else:
                    ac_huffman_code = AC_Chroma_Huffman[(ac_runlength, ac_size)]
                entropy_code += ac_huffman_code + ac_amplitude
        
        result.append(entropy_code)
    return result
```

​		在解码时，需要进行逆熵编码操作。

> 值得注意的是：
>
> - 由于是huffman编码（没有一个编码是另一个编码的前缀），可以通过不断读入字符来看是否匹配来解出huffman编码前的symbol 1。
> - 仍然需要对0和-1进行特殊的判断。
> - 【坑】遍历一个图像块的编码的变量`j`不可以使用for循环，只能使用while循环。（for循环不允许在循环内部改变`j`的值！

```python
# 逆熵编码
def de_entropy_coding(entropy_result, is_Y):
    items = {}  # huffman编码map
    key = 0
    coding_results = []
    for i in range(len(entropy_result)):
        block = entropy_result[i]
        is_dc = True    # 表示现在是否在解码第一个dc值
        temp_str = ""
        mid_coding = []
        j = 0
        while j < len(block):
            temp_str += block[j]
            j+=1
            # 判断当前的huffman编码表
            if is_Y is True and is_dc is True:
                items = DC_Luminance_Huffman.items()
            elif is_Y is True and is_dc is False:
                items = AC_Luminance_Huffman.items()
            elif is_Y is False and is_dc is True:
                items = DC_Chroma_Huffman.items()
            else:
                items = AC_Chroma_Huffman.items()
            for (k,v) in items:
                if temp_str == v:
                    key = k
                    temp_str = ""
                    # 如果现在解码的是dc，则key为int，表示dc_size
                    if is_dc is True:
                        dc_amplitude = ""
                        for m in range(0, key):
                            dc_amplitude += block[j]
                            j+=1
                        if key == 0:   # 加判断：如果dc_size=0，则dc=0.！！！
                            dc = 0
                            j+=1
                        else:
                            dc = bin2dec(dc_amplitude, False)
                        mid_coding.append(dc)
                        is_dc = False
                    # 如果现在解码的是ac，则key为(int,int)，表示(runlength, size)
                    else:
                        ac_runlength, ac_size = key
                        # 不需要判断是不是特殊编码(15,0)，因为(15,0)也是一样恢复回去
                        ac_amplitude = ""
                        for m in range(ac_size):
                            ac_amplitude += block[j]
                            j+=1
                        if ac_amplitude=="":   # 只有可能是(15,0)
                            mid_coding.append((ac_runlength, 0))
                        else:
                            is_end = False
                            if j == len(block):
                                is_end = True
                            ac_value = bin2dec(ac_amplitude, is_end)
                            mid_coding.append((ac_runlength, ac_value))
                    break
        coding_results.append(mid_coding)
    return coding_results         
```

------

最终，由于从DCT开始，都是对其中一个分量进行操作，故对函数进行如下打包：

```python
# 对已分块的每个通道（Y,Cb,Cr）进行dct,量化，DC&AC编码，熵编码
# image是采样后的其中一个分量，is_Y表示是否是Y通道
def process_channel(image_channel, is_Y):
    # DCT+量化
    result = DCT_quan(image_channel, is_Y)
    # DC和AC编码
    result = DC_AC_coding(result)
    # 熵编码
    result = entropy_coding(result, is_Y)
    return result

# 逆熵编码，逆DC和AC编码，逆DCT+量化
def de_process_channel(coding, is_Y, row, col):
    # 逆熵编码
    result = de_entropy_coding(coding, is_Y)
    # 逆DC和AC编码
    result = de_DC_AC_coding(result)
    # 逆DCT+量化
    result = de_DCT_quan(result, is_Y, row, col)
    return result
```

最终的`main`函数是：

```python
# 读图片
image = readImage("./动物卡通图片.jpg") 
row, col = image.shape[0], image.shape[1]
print("original row and col is",row, col)
# RGB转ycbcr
ycbcr = rgb2ycbcr(image)
# padding
image_pad = padding(ycbcr)
row_new, col_new = image_pad.shape[0], image_pad.shape[1]
print("new row and col is",row_new, col_new)
# 二次采样
samp_y, samp_cb, samp_cr = sampling(image_pad, row_new, col_new)
# 对已分块的每个通道（Y,Cb,Cr）进行dct,量化，DC&AC编码，熵编码
res_y = process_channel(samp_y, is_Y=True)
res_cb = process_channel(samp_cb, is_Y=False)
res_cr = process_channel(samp_cr, is_Y=False) 
# 计数
count_bit(res_y, res_cb, res_cr)
# 逆熵编码，逆DC和AC编码，逆DCT+量化
res_y = de_process_channel(res_y, True, samp_y.shape[0], samp_y.shape[1])
res_cb = de_process_channel(res_cb, False, samp_cb.shape[0], samp_cb.shape[1])
res_cr = de_process_channel(res_cr, False, samp_cr.shape[0], samp_cr.shape[1])
# 逆二次采样：还原
src_cb, src_cr = inverse_sampling(res_cb, res_cr, row_new, col_new)
# 将还原的三个2-D narray拼成3-D的 narray
new_image = np.zeros((row_new, col_new, 3), dtype=float)  
new_image[:, :, 0]=res_y
new_image[:, :, 1]=src_cb
new_image[:, :, 2]=src_cr
# de-padding：去掉padding
image1 = de_padding(image, new_image)
# ycbcr转回RGB
image1_decompress = ycbcr2rgb(image1)
# 还原成图片
Image.fromarray(image1_decompress)
```



#### 2.2.3 结果对比

##### 视觉效果

在本实验中，可以通过解压压缩后的jpg图片的视觉效果来反映压缩的效果（即实现逆操作）。

动物卡通图片：

| 原图                                                         | JPEG                                                         | GIF                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210606213726765](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213726765.png) | ![image-20210606213802647](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213802647.png) | ![image-20210606213854291](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213854291.png) |
| ![image-20210606213920490](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213920490.png) | ![image-20210606213942905](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213942905.png) | ![image-20210606213957947](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210606213957947.png) |

动物照片：

| 原图                                                         | JPEG                                                         | GIF                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210607000408516](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000408516.png) | ![image-20210607000428297](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000428297.png) | ![image-20210607000441253](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000441253.png) |
| <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000520795.png" alt="image-20210607000520795" style="zoom:130%;" /> | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000539835.png" alt="image-20210607000539835" style="zoom:150%;" /> | <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210607000606993.png" alt="image-20210607000606993" style="zoom: 200%;" /> |

​		通过三者原图和局部图的视觉效果对比，可以发现`GIF`压缩之后的失真度高，细节丢失严重，如鹦鹉的翅膀上的纹路基本丢失了，而身体上的阴影也被放大。相比之下，`jpeg`的颜色和纹路更加清晰，但也因为高频信息的丢失，导致在一些颜色的边缘出现了模糊的迹象（如翅膀边缘附近有阴影）。

> 注：结果图片保存在result文件夹中。



##### 压缩率

压缩率的定义如下：
$$
压缩率=\frac{B_1}{B_0}\\
其中，B_0为原图像总字节数，B_1为压缩后图像的总字节数
$$
对于动物卡通图片来说：（JPEG_standard表示使用电脑标准的JPEG压缩）

|          | 原图bmp | JPEG_standard | JPEG_mine | GIF    |
| -------- | ------- | ------------- | --------- | ------ |
| 图片大小 | 2095kB  | 117kB         | 97kB      | 481kB  |
| 压缩率   |         | 5.58%         | 4.63%     | 22.95% |

对于动物照片来说：

|          | 原图bmp | JPEG_standard | JPEG_mine | GIF    |
| -------- | ------- | ------------- | --------- | ------ |
| 图片大小 | 2047kB  | 226kB         | 170kB     | 733kB  |
| 压缩率   |         | 11.04%        | 8.30%     | 35.80% |

**分析**：从上述表格可看出，在两种图片中，`GIF`的压缩率都大于`JPEG`的压缩率，即`GIF`的压缩程度不如`JPEG`的压缩程度。且动物卡通图片的压缩率普遍小于动物图片的压缩率。但与上面理论分析不同的是，`GIF`在动物卡通图片中的表现仍然不如`JPEG`，笔者猜测应该是由于此卡通图片的颜色较多导致的。如果颜色较少（如黑白图的话），`GIF`的压缩程度应会高于`JPEG`。



##### 失真率

​		在这里使用python的PIL库函数，实现直方图匹配（mode=1）和图像感知的哈希算法（mode=2），以计算两个图片的相似度。相似度越高，失真率越低。

```python
def difference(hist1, hist2):
    sum1 = 0
    for i in range(len(hist1)):
        if hist1[i] == hist2[i]:
            sum1 += 1
        else:
            sum1 += 1 - float(abs(hist1[i] - hist2[i]))/ max(hist1[i], hist2[i])
    return sum1/len(hist1)

# mode=1：直方图匹配；mode=2：图像感知的哈希算法
def similary_calculate(path1 , path2 , mode):
    if(mode == 2):
        img1 = Image.open(path1).resize((8,8)).convert('1')  
        img2 = Image.open(path2).resize((8,8)).convert('1')
        hist1 = list(img1.getdata())
        hist2 = list(img2.getdata())
        return difference(hist1, hist2)
 
    # 预处理
    img1 = Image.open(path1).resize((256,256)).convert('RGB')  
    img2 = Image.open(path2).resize((256,256)).convert('RGB')
    if(mode == 1):
        return difference(img1.histogram(), img2.histogram())
```

​		根据上述代码，计算得到结果：

|                    | 动物照片.bmp & .jpg | 动物图片.bmp & .gif | 动物卡通图片.bmp & .jpg | 动物卡通图片.bmp & .gif |
| ------------------ | ------------------- | ------------------- | ----------------------- | ----------------------- |
| 直方图匹配         | 0.8869              | 0.2843              | 0.8281                  | 0.2964                  |
| 图像感知的哈希算法 | 0.9531              | 0.6875              | 0.9375                  | 0.7031                  |

**分析**：无论使用哪种方法计算相似度，`jpg`的相似度都要远高于`gif`，因此说明`jpg`压缩方式的失真率小于`gif`，与上面观察结果一致。