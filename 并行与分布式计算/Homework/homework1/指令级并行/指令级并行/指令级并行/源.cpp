#include <immintrin.h>
#include <stdio.h>
#include<iostream>
#include<time.h>
#include<vector>
#include<Windows.h>
using namespace std;
const int N = 1000000;
int *result2 = new int[N]();//将每个单元初始化为0，多线程
int *result3 = new int[N]();//串行
//！！坑！不能返回int*,因为临时变量会消亡！！
void addvalues1(int *a, int *b, int *&result)//已知size为1000 000
{
	int nBlockWidth = 8;//每次计算8个int分量的相加
	int cntBlock = N / 8;//block数量（循环计数上限）
	__m256i loadData1, loadData2;
	__m256i resultData = _mm256_setzero_si256();
	__m256i *p1 = (__m256i *)a;//进行指针类型的转换
	__m256i *p2 = (__m256i *)b;
	__m256i *p3 = (__m256i *)result;
	for (int i = 0; i < cntBlock; i++)
	{
		loadData1 = _mm256_load_si256(p1);//加载数据1
		loadData2 = _mm256_load_si256(p2);//加载数据2
		resultData = _mm256_add_epi32(loadData1, loadData2);//将2数据相加
		_mm256_store_si256(p3, resultData);//卸载数据回结果数组result
		p1 += 1;//！！！是加1而不是加8！（p1+1=p1+1*8 int,"1"means one __m256i)
		p2 += 1;
		p3 += 1;
		//if (i % 125 == 0)
			//Sleep(100);
	}
	//128与256的不同就在于把函数前面的256给删掉。后面的256改成128
	/*int nBlockWidth = 4;//每次计算8个int分量的相加
	int cntBlock = N / 4;//block数量（循环计数上限）
	__m128i loadData1, loadData2, resultData;
	//__m128i resultData = _mm128_setzero_si128();
	__m128i *p1 = (__m128i *)a;//进行指针类型的转换
	__m128i *p2 = (__m128i *)b;
	__m128i *p3 = (__m128i *)result;
	for (int i = 0; i < cntBlock; i++)
	{
		loadData1 = _mm_load_si128(p1);//加载数据1
		loadData2 = _mm_load_si128(p2);//加载数据2
		resultData = _mm_add_epi32(loadData1, loadData2);//将2数据相加
		_mm_store_si128(p3, resultData);//卸载数据回结果数组result
		p1 += 1;//！！！是加1而不是加8！（p1+1=p1+1*8 int,"1"means one __m256i)
		p2 += 1;
		p3 += 1;
		if (i % 25 == 0)
			Sleep(1);
	}*/
	return;
}

void addvalues3(int *a, int *b, int *&result)
{
	for (int i = 0; i < N; i++)
	{
		result[i] = a[i] + b[i];
		//if (i % 1000 == 0)
			//Sleep(100);
	}
}

int main()
{
	
	int a[N];
	int b[N];
	for (int i = 0; i < N; i++)
	{
		a[i] = rand() % 100 + 1;
		b[i] = rand() % 100 + 1;
	}

	clock_t start2 = clock();
	addvalues1(a, b, result2);
	clock_t end2 = clock();

	clock_t start3 = clock();
	addvalues3(a, b, result3);
	clock_t end3 = clock();

	cout << "指令级并行时间(s)：" << (double)(end2 - start2) / CLOCKS_PER_SEC << endl;
	cout << "普通串行时间(s)：" << (double)(end3 - start3) / CLOCKS_PER_SEC << endl;
	cout << "加速比为" << (double)(end3 - start3) / (double)(end2 - start2) << endl;
	system("pause");
	return 0;
}
