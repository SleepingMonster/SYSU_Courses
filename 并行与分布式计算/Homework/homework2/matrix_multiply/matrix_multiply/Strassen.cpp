#include <iostream>
#include <ctime>
//#include <Windows.h>
using namespace std;
#include "Strassen.h"
#define N 256
/*int main()
{
	Strassen_class<int> stra;
	int MatrixSize = N;

	int** MatrixA;    //矩阵A
	int** MatrixB;    //矩阵B
	int** MatrixC;    //结果矩阵

	clock_t start1;
	clock_t end1;

	clock_t start2;
	clock_t end2;

	//申请内存
	MatrixA = new int *[MatrixSize];
	MatrixB = new int *[MatrixSize];
	MatrixC = new int *[MatrixSize];

	for (int i = 0; i < MatrixSize; i++)
	{
		MatrixA[i] = new int[MatrixSize];
		MatrixB[i] = new int[MatrixSize];
		MatrixC[i] = new int[MatrixSize];
	}

	stra.FillMatrix(MatrixA, MatrixB, MatrixSize);  //矩阵赋值

  //*******************conventional multiplication test
	start1 = clock();

	stra.MUL(MatrixA, MatrixB, MatrixC, MatrixSize);//普通矩阵相乘算法 T(n) = O(n^3)

	end1 = clock();

	cout << "\n矩阵运算结果... \n";
	//stra.PrintMatrix(MatrixC, MatrixSize);

	//*******************Strassen multiplication test
	start2 = clock();

	stra.Strassen(N, MatrixA, MatrixB, MatrixC); //strassen矩阵相乘算法

	end2 = clock();


	cout << "\n矩阵运算结果... \n";
	//stra.PrintMatrix(MatrixC, MatrixSize);

	cout << "\n普通矩阵乘法: " << (double)(end1 - start1) / CLOCKS_PER_SEC << " s";
	cout << "\nStrassen算法:" << (double)(end2 - start2) / CLOCKS_PER_SEC << " s";
	system("Pause");
	return 0;

}*/
