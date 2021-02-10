#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

#define dim 256

/*void common_matrix_product(int **&a,int**&b, int**&c)
{
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			for (int k = 0; k < dim; k++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}*/

typedef struct matrix {
	vector<vector<int>> data;
	matrix(int r, int c)
	{
		data.resize(c, vector<int>(r, 0));
	}
	//r,c表示行、列起始位置，rn,cn为行、列终止位置。
	void SetSubMatrix(int r, int c, int rn, int cn, const matrix &A, const matrix &B)
	{
		for (int i = c; i < cn; ++i)
		{
			for (int j = r; j < rn; j++)
			{
				data[i][j] = A.data[i - c][j - r] + B.data[i - c][j - r];
			}
		}
	}

	//ar,ac是A矩阵左上角元素的行与列，n表示该矩阵的大小
	static matrix SquareMultiplyRecursive(matrix &A, matrix &B, int ar, int ac, int br, int bc, int n)
	{
		matrix C(n, n);

		if (n == 1)
			C.data[0][0] = A.data[ac][ar] * B.data[bc][br];
		else
		{
			C.SetSubMatrix(0, 0, n / 2, n / 2, 
				SquareMultiplyRecursive(A, B, ar, ac, br, bc, n / 2),
				SquareMultiplyRecursive(A, B, ar, ac + (n / 2), br + (n / 2), bc, n / 2));

			C.SetSubMatrix(0, n / 2, n / 2, n, 
				SquareMultiplyRecursive(A, B, ar, ac, br, bc + (n / 2), n / 2),
				SquareMultiplyRecursive(A, B, ar, ac + (n / 2), br + (n / 2), bc + (n / 2), n / 2));
			
			C.SetSubMatrix(n / 2, 0, n, n / 2, 
				SquareMultiplyRecursive(A, B, ar + n / 2, ac, br, bc, n / 2),
				SquareMultiplyRecursive(A, B, ar + (n / 2), ac + (n / 2), br + (n / 2), bc, n / 2));
			
			C.SetSubMatrix(n / 2, n / 2, n, n, 
				SquareMultiplyRecursive(A, B, ar + (n / 2), ac, br, bc + (n / 2), n / 2),
				SquareMultiplyRecursive(A, B, ar + (n / 2), ac + (n / 2), br + (n / 2), bc + (n / 2), n / 2));

		}
		return C;
	}

	void Print()
	{
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				cout << data[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	static matrix common_matrix_product(const matrix &a, const matrix &b)
	{
		matrix c(dim, dim);
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					c.data[i][j] += a.data[i][k] * b.data[k][j];
				}
			}
		}
		return c;
	}
}matrix;


/*int main()
{
	matrix A(dim, dim);
	matrix B(dim, dim);
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			A.data[i][j] = rand()%5+1;
			B.data[i][j] = rand()%5+1;
		}
	}
	//A.Print();
	//B.Print();
	//clock_t start1 = clock();
	//matrix C(matrix::SquareMultiplyRecursive(A, B, 0, 0, 0, 0, dim));
	//clock_t end1 = clock();
	//C.Print();

	clock_t start2 = clock();
	matrix D(matrix::common_matrix_product(A, B));
	clock_t end2 = clock();
	//cout << "分治算法:"<<(double)(end1 - start1) / CLOCKS_PER_SEC << endl;
	cout << "一般算法:"<<(double)(end2 - start2) / CLOCKS_PER_SEC << endl;
	return 0;
}*/