#pragma once
#ifndef STRASSEN_HH
#define STRASSEN_HH
template<typename T>
class Strassen_class {
public:
	void ADD(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize);//矩阵相加
	void SUB(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize);//矩阵相减
	void MUL(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize);//一般算法实现
	void FillMatrix(T** MatrixA, T** MatrixB, int length);//A,B矩阵赋值
	void PrintMatrix(T **MatrixA, int MatrixSize);//打印矩阵
	void Strassen(int N, T **MatrixA, T **MatrixB, T **MatrixC);//Strassen算法实现
};
template<typename T>//矩阵相加
void Strassen_class<T>::ADD(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
		}
	}
}
template<typename T>//矩阵相减
void Strassen_class<T>::SUB(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
		}
	}
}
template<typename T>//矩阵相乘：一般算法
void Strassen_class<T>::MUL(T** MatrixA, T** MatrixB, T** MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = 0;
			for (int k = 0; k < MatrixSize; k++)
			{
				MatrixResult[i][j] = MatrixResult[i][j] + MatrixA[i][k] * MatrixB[k][j];
			}
		}
	}
}


template<typename T>//Strassen
void Strassen_class<T>::Strassen(int N, T **MatrixA, T **MatrixB, T **MatrixC)
{

	int HalfSize = N / 2;
	int newSize = N / 2;

	if (N <= 64)    //分治门槛，小于这个值时不再递归，而是采用普通矩阵乘法
	{
		MUL(MatrixA, MatrixB, MatrixC, N);
	}
	else
	{
		T** A11;
		T** A12;
		T** A21;
		T** A22;

		T** B11;
		T** B12;
		T** B21;
		T** B22;

		T** C11;
		T** C12;
		T** C21;
		T** C22;

		T** M1;
		T** M2;
		T** M3;
		T** M4;
		T** M5;
		T** M6;
		T** M7;
		T** AResult;
		T** BResult;

		//申请空间1
		A11 = new T *[newSize];
		A12 = new T *[newSize];
		A21 = new T *[newSize];
		A22 = new T *[newSize];

		B11 = new T *[newSize];
		B12 = new T *[newSize];
		B21 = new T *[newSize];
		B22 = new T *[newSize];

		C11 = new T *[newSize];
		C12 = new T *[newSize];
		C21 = new T *[newSize];
		C22 = new T *[newSize];

		M1 = new T *[newSize];
		M2 = new T *[newSize];
		M3 = new T *[newSize];
		M4 = new T *[newSize];
		M5 = new T *[newSize];
		M6 = new T *[newSize];
		M7 = new T *[newSize];

		AResult = new T *[newSize];
		BResult = new T *[newSize];

		int newLength = newSize;

		//申请空间2
		for (int i = 0; i < newSize; i++)
		{
			A11[i] = new T[newLength];
			A12[i] = new T[newLength];
			A21[i] = new T[newLength];
			A22[i] = new T[newLength];

			B11[i] = new T[newLength];
			B12[i] = new T[newLength];
			B21[i] = new T[newLength];
			B22[i] = new T[newLength];

			C11[i] = new T[newLength];
			C12[i] = new T[newLength];
			C21[i] = new T[newLength];
			C22[i] = new T[newLength];

			M1[i] = new T[newLength];
			M2[i] = new T[newLength];
			M3[i] = new T[newLength];
			M4[i] = new T[newLength];
			M5[i] = new T[newLength];
			M6[i] = new T[newLength];
			M7[i] = new T[newLength];

			AResult[i] = new T[newLength];
			BResult[i] = new T[newLength];
		}
		//划分成4个子矩阵
		for (int i = 0; i < N / 2; i++)
		{
			for (int j = 0; j < N / 2; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + N / 2];
				A21[i][j] = MatrixA[i + N / 2][j];
				A22[i][j] = MatrixA[i + N / 2][j + N / 2];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + N / 2];
				B21[i][j] = MatrixB[i + N / 2][j];
				B22[i][j] = MatrixB[i + N / 2][j + N / 2];

			}
		}

		//现在计算M1~M7矩阵
		//M1[][]
		ADD(A11, A22, AResult, HalfSize);
		ADD(B11, B22, BResult, HalfSize);                //p5=(a+d)*(e+h)
		Strassen(HalfSize, AResult, BResult, M1); //now that we need to multiply this , we use the strassen itself .
		//M2[][]
		ADD(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11   p3=(c+d)*e
		Strassen(HalfSize, AResult, B11, M2);       //Mul(AResult,B11,M2);
		//M3[][]
		SUB(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)   p1=a*(f-h)
		Strassen(HalfSize, A11, BResult, M3);       //Mul(A11,BResult,M3);
		//M4[][]
		SUB(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)    p4=d*(g-e)
		Strassen(HalfSize, A22, BResult, M4);       //Mul(A22,BResult,M4);
		//M5[][]
		ADD(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22   p2=(a+b)*h
		Strassen(HalfSize, AResult, B22, M5);       //Mul(AResult,B22,M5);
		//M6[][]
		SUB(A21, A11, AResult, HalfSize);
		ADD(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)   p7=(c-a)(e+f)
		Strassen(HalfSize, AResult, BResult, M6);    //Mul(AResult,BResult,M6);
		//M7[][]
		SUB(A12, A22, AResult, HalfSize);
		ADD(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)    p6=(b-d)*(g+h)
		Strassen(HalfSize, AResult, BResult, M7);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADD(M1, M4, AResult, HalfSize);
		SUB(M7, M5, BResult, HalfSize);
		ADD(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADD(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADD(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADD(M1, M3, AResult, HalfSize);
		SUB(M6, M2, BResult, HalfSize);
		ADD(AResult, BResult, C22, HalfSize);

		//组合小矩阵C11,C12,C21,C22到一个大矩阵
		for (int i = 0; i < N / 2; i++)
		{
			for (int j = 0; j < N / 2; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + N / 2] = C12[i][j];
				MatrixC[i + N / 2][j] = C21[i][j];
				MatrixC[i + N / 2][j + N / 2] = C22[i][j];
			}
		}

		// 释放矩阵内存空间
		for (int i = 0; i < newLength; i++)
		{
			delete[] A11[i]; delete[] A12[i]; delete[] A21[i];
			delete[] A22[i];

			delete[] B11[i]; delete[] B12[i]; delete[] B21[i];
			delete[] B22[i];
			delete[] C11[i]; delete[] C12[i]; delete[] C21[i];
			delete[] C22[i];
			delete[] M1[i]; delete[] M2[i]; delete[] M3[i]; delete[] M4[i];
			delete[] M5[i]; delete[] M6[i]; delete[] M7[i];
			delete[] AResult[i]; delete[] BResult[i];
		}
		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;

	}//end of else

}

template<typename T>
void Strassen_class<T>::FillMatrix(T** MatrixA, T** MatrixB, int length)
{
	for (int row = 0; row < length; row++)
	{
		for (int column = 0; column < length; column++)
		{
			MatrixA[row][column] = rand() % 5 + 1;
			MatrixB[row][column] = rand() % 5 + 1;
		}

	}
}
template<typename T>
void Strassen_class<T>::PrintMatrix(T **MatrixA, int MatrixSize)
{
	cout << endl;
	for (int row = 0; row < MatrixSize; row++)
	{
		for (int column = 0; column < MatrixSize; column++)
		{
			cout << MatrixA[row][column] << " ";
		}
		cout << endl;

	}
	cout << endl;
}
#endif
