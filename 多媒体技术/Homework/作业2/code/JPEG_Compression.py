import numpy as np
from PIL import Image
from util import *
import math
import string

# 读取图片
def readImage(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image


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
    # return np.uint8(ycbcr)  
    return ycbcr


# YCbCr转化为RGB
def ycbcr2rgb(ycbcr):
    # args = np.array([[1, -0.000001, 1.402], [1, -0.344135, -0.714136], [1, 1.772, 0]])
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


# 对图片的右边缘和下边缘进行padding（16为倍数：采样1/2，DCT 1/8）
# 由于要取左上角的值，故左上不能padding，否则取得就是padding的值。
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
    # new_image = np.array([[[0,0, 0] for i in range(col1)] for j in range(row1)])
    new_image = np.zeros((row1, col1, 3), dtype=float)
    # new_image[:row, :col, :] = image[:, :, :]
    new_image[:row, :col] = image
    return new_image


# 去掉padding
def de_padding(image, new_image):
    row = image.shape[0]
    col = image.shape[1]
    result = np.array(new_image[:row, :col])
    # print(result.shape)
    return result


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


# 切分图像成8*8的块
def split_blocks(image, block_size=8):
    row_blocks = np.split(image, image.shape[0]/block_size)  # 先按行切分
    result = []  # 得到的是二维的[[首8行中的所有block],[]]
    for block in row_blocks:
        result.append(np.split(block, image.shape[1]/block_size, axis=1))
    result = np.array(result)
    # print(result.shape)
    return result


def combine_blocks(blocks, row, col):
    image = np.zeros((row, col))
    indices = [(i,j) for i in range(0, row, 8) for j in range(0, col, 8)]
    for block, index in zip(blocks, indices):
        i, j = index
        image[i:i+8, j:j+8] = block
    return image


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


# 还原Z字形扫描
def de_zigzag(zigzag_array):
    result = np.zeros((8,8))
    n = 8
    i = 0  # row
    j = 0  # col
    count = 0
    while i<n and j<n:
        if i>=n or j>=n:
            print(i,j)
        result[i][j]=zigzag_array[count]
        count += 1
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
    return result


# AC系数上的游程编码(RLC)，且已将runlength进行(15,0)处理
def RLC(ac_vector):
    # print(ac_vector.shape)
    count=0
    result = []
    for i in range(len(ac_vector)):
        if ac_vector[i]!=0:
            while(count>15):
                result.append((15,0))
                count-=15
            result.append((count,ac_vector[i]))
            count=0
        else:
            count += 1
    # 末尾0
    if count!=0:
        result.append((0,0))   
    return result


# 分块,dct,量化（image是其中一个通道，is_Y表示是否是Y通道）
def DCT_quan(image, is_Y):
    # 量化矩阵
    quantization_matrix = 0
    if is_Y is True:
        quantization_matrix = brightness_quantization_matrix
    else:
        quantization_matrix = chroma_quantization_matrix
    # 分块8*8
    blocks = split_blocks(image, 8)
    # 得到dct矩阵
    dct_matrix = get_dct_matrix()
    result_blocks = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            # DC 和 AC系数熵编码前的准备：
            mid_result = []
            # DCT
            dct_coefficient = dct(blocks[i,j], dct_matrix)
            # 量化
            dct_coeff_quan = quantization(dct_coefficient, quantization_matrix, 50)
            # collect
            result_blocks.append(dct_coeff_quan)
    return result_blocks


# 反量化，iDCT，组合blocks
def de_DCT_quan(result, is_Y, row, col):
    # 量化矩阵
    quantization_matrix = 0
    if is_Y is True:
        quantization_matrix = brightness_quantization_matrix
    else:
        quantization_matrix = chroma_quantization_matrix
    
    result_blocks = []
    # 得到dct矩阵
    dct_matrix = get_dct_matrix()
    for i in range(len(result)):
        # 取出一个8*8的块
        dct_coeff_quan = result[i]
        # 反量化
        dct_coeff = de_quantization(dct_coeff_quan, quantization_matrix, 50)
        # iDCT
        idct_matrix = inverse_dct(dct_coeff, dct_matrix)
        result_blocks.append(idct_matrix)
    result1 = combine_blocks(result_blocks, row, col)
    return result1


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
            print(dc_ac_coding)
        
        # 逆Z扫描
        mid_result = de_zigzag(mid_result)
        result_blocks.append(mid_result)
    return result_blocks



# 十进制和二进制的相互转换
def dec2bin(num):
    result = ""
    num = int(num)
    is_zero = False
    if num ==0:
        is_zero=True
    if num<0:
        num_binary = (bin(-num))[2:]  # 去掉"ob"
        for i in range(len(num_binary)):
            if num_binary[i]=='0':
                result = result + '1'
            elif num_binary[i] =='1':
                result = result + '0'
    else:
        result = (bin(num))[2:]  # 去掉"ob"
    return result, is_zero


def bin2dec(num, is_end):
    mid_str = ""
    if num[0]=='0' and is_end is False:
        for i in range(len(num)):
            if num[i]=='1':
                mid_str += '0'
            else:
                mid_str += '1'
        return -int(mid_str, 2)
    else:
        mid_str = num
        return int(mid_str, 2)


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
                        if key == 0:   # 加判断：如果dc_size=0，则dc=0.！！！一定要加，否则无法判断是0还是-1
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


def count_bit(res_y, res_cb, res_cr):
    result_bit = 0
    for i in res_y:
        result_bit += len(i)
    for i in res_cb:
        result_bit += len(i)
    for i in res_cr:
        result_bit += len(i)
    print("Total size of image after this jpeg compression is "+str(result_bit/1024/8))


def difference(hist1, hist2):
    sum1 = 0
    for i in range(len(hist1)):
        if hist1[i] == hist2[i]:
            sum1 += 1
        else:
            sum1 += 1 - float(abs(hist1[i] - hist2[i])) / max(hist1[i], hist2[i])
    return sum1 / len(hist1)


def similary_calculate(path1, path2, mode):
    if (mode == 2):
        img1 = Image.open(path1).resize((8, 8)).convert('1')
        img2 = Image.open(path2).resize((8, 8)).convert('1')
        hist1 = list(img1.getdata())
        hist2 = list(img2.getdata())
        return difference(hist1, hist2)
    # 预处理
    img1 = Image.open(path1).resize((256, 256)).convert('RGB')
    img2 = Image.open(path2).resize((256, 256)).convert('RGB')
    if (mode == 1):
        return difference(img1.histogram(), img2.histogram())
    return 0


# 读图片
# picture = "动物卡通图片"
picture = "动物照片"
image = readImage("../data/"+ picture + ".jpg")
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
# picture = "动物照片"
image_test = Image.open("../data/"+ picture + ".jpg")
image_test.save("../result/"+ picture + "1.jpg")
image_test.save("../result/"+ picture + "2.bmp")
image_test.save("../result/"+ picture + "3.gif")
# 还原图片
Image.fromarray(image1_decompress)
mine = Image.fromarray(image1_decompress)
mine.save("../result/"+ picture + "_mine.jpg")


print("\n运用直方图匹配方法计算失真率：")
simi1 = similary_calculate("../result/动物照片2.bmp", "../result/动物照片_mine.jpg", 1)
simi2 = similary_calculate("../result/动物照片2.bmp", "../result/动物照片3.gif", 1)
simi3 = similary_calculate("../result/动物卡通图片2.bmp", "../result/动物卡通图片_mine.jpg", 1)
simi4 = similary_calculate("../result/动物卡通图片2.bmp", "../result/动物卡通图片3.gif", 1)
print("动物照片bmp&jpg的相似度为: %f"% simi1)
print("动物照片bmp&gif的相似度为: %f"% simi2)
print("动物卡通图片bmp&jpg的相似度为: %f"% simi3)
print("动物卡通图片bmp&gif的相似度为%f: "% simi4)

simi1 = similary_calculate("../result/动物照片2.bmp", "../result/动物照片_mine.jpg", 2)
simi2 = similary_calculate("../result/动物照片2.bmp", "../result/动物照片3.gif", 2)
simi3 = similary_calculate("../result/动物卡通图片2.bmp", "../result/动物卡通图片_mine.jpg", 2)
simi4 = similary_calculate("../result/动物卡通图片2.bmp", "../result/动物卡通图片3.gif", 2)
print("\n运用图像感知的哈希算法计算失真率：")
print("动物照片bmp&jpg的相似度为: %f"% simi1)
print("动物照片bmp&gif的相似度为: %f"% simi2)
print("动物卡通图片bmp&jpg的相似度为: %f"% simi3)
print("动物卡通图片bmp&gif的相似度为: %f"% simi4)


