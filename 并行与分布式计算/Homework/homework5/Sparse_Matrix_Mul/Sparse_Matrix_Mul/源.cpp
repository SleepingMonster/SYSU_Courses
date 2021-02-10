#include<iostream>
#include<fstream>
#include<vector>
#include<time.h>
#include<omp.h>
using namespace std;

int row_size, col_size, num;
typedef vector<double> Vector;
//用行压缩方式储存矩阵：
vector<double> val;//大小为num，存值
vector<int> col_idx;//大小为num，存对应值的列值
vector<int> row_ptr;//大小为行数，存每一行第一个非零元素在val中的索引
int thread_num;

Vector mul(Vector& vec)
{
	Vector result(row_ptr.size(),0);
#pragma omp parallel for num_threads(thread_num)//并行！设置了并行线程数
	for (int i = 0; i < row_ptr.size(); i++)
	{
		int end;
		/*if (i != row_ptr.size() - 1)
		{
			int j = i + 1;//下一个有非零元素的行的下标
			while (j < row_ptr.size())
			{
				if (row_ptr[j] != 0)
					break;
				j++;
			}
			if (j != row_ptr.size())
				end = row_ptr[j] - 1;
			else
				end = num - 1;
		}
		else
			end = num - 1;*/
		end = i == row_ptr.size() - 1 ? num - 1 : row_ptr[i + 1] - 1;
		for (int m = row_ptr[i]; m <= end; m++)
		{
			result[i] += val[m] * vec[col_idx[m]];//该行的非零值和vec的对应单元值相乘，累加
		}
	}
	return result;
}

int main()
{
	ifstream fin("1138_bus.mtx");
	if (!fin)
	{
		cout << "打开文件失败！" << endl;
		exit(1);
	}
	while (fin.peek() == '%')
		while (fin.get() != '\n') ;//这样可以跳过前面的注释
	//读取行数、列数、非零值的大小
	fin >> row_size >> col_size >> num;
	val.resize(num);
	col_idx.resize(num);
	row_ptr.resize(row_size,0);
	
	Vector vec(col_size);
	int x, y;//x=行,y=列
	double t;//元素值
	int former = -1;
	for (int i = 0; i < num; i++)
	{
		fin >> y >> x >> t;//读取每一个单元，且第一个数看成列，第二个数看成行
		if ((x - 1) != former)
		{
			row_ptr[x - 1] = i;//第x行的非零元素是i开始的
			former = x - 1;
		}	
		val[i] = t;
		col_idx[i] = y-1;
	}
	for (int i = 0; i < col_size; i++)
	{
		vec[i] = rand()%100+1;
	}
	cout << "请输入并行的线程数：";
	cin >> thread_num;
	clock_t start = clock();
	Vector result;
	for(int i=0;i<1e4;i++)
		result = mul(vec);
	clock_t end = clock();
	cout << thread_num<<"级线程并行时间(s)：" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	for (auto i:result)
		cout << i << endl;
	system("pause");
	return 0;
}