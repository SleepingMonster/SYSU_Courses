#include <iostream>
#include <omp.h>
#include<time.h>
using namespace std;

#define size 1000
#define ThreadNumber 8

int firstMatrix[size][size];//矩阵
int secondMatrix[size][1];//向量
int resultMatrix[size][1];//结果

void calculate(int row, int col)//计算单元的乘积
{
	int result=0;
	for (int i = 0; i < size; i++)
	{
		result += firstMatrix[row][i] * secondMatrix[i][col];
	}
	resultMatrix[row][col] = result;
}

void matrixInit()//矩阵、向量初始化
{
	#pragma omp parallel for num_threads(ThreadNumber)//并行
	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++)
			firstMatrix[row][col] = rand() % 10;
		secondMatrix[row][0] = rand() % 10;
	}
}

void matrixMulti()//矩阵乘法
{
	#pragma omp parallel for num_threads(ThreadNumber)//并行
	for (int row = 0; row < size; row++)
	{
		for (int col = 0; col < size; col++)
			calculate(row, col);
	}
}

int main()
{
	matrixInit();
	clock_t t1 = clock();
	matrixMulti();
	clock_t t2 = clock();
	cout << "time:" << t2 - t1;
	/*for (int i = 0; i < size; i++)
	{
		cout << firstMatrix[0][i] << " ";
	}
	cout << endl << endl;
	for (int j = 0; j < size; j++)
		cout << secondMatrix[j][0]<<" ";
	cout << endl<<endl;
	cout << resultMatrix[0][0];*/
	system("pause");
	return 0;
}
