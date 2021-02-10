#include <immintrin.h>
#include <stdio.h>
#include<iostream>
#include<time.h>
#include<Windows.h>
using namespace std;

const int N = 1000000;
const int num_of_CPUs = 4;//全局变量！！
const int num_of_CPUs1 = 6;
const int num_of_CPUs2 = 8;
int *result1 = new int[N]();//将每个单元初始化为0，多线程
int *result3 = new int[N]();//串行

struct ThreadParam {
	size_t begin;
	size_t end;
	int *result;//结果数组
	int *a;//向量1
	int *b;//向量2
	ThreadParam(size_t begin1, size_t end1, int *result1, int *a1, int *b1) :begin(begin1), end(end1), result(result1), a(a1), b(b1) {}
};

DWORD WINAPI addvalues2(LPVOID param) {
	ThreadParam*p = static_cast<ThreadParam*>(param);//传入的param是void*，所以要进行转换
	for (size_t i = p->begin; i < p->end; ++i)
	{
		p->result[i] = p->a[i] + p->b[i];
		if (i % 100 == 0)
			Sleep(10);
	}
	delete p;
	return 0;
}

void addvalues3(int *a, int *b, int *&result)
{
	for (int i = 0; i < N; i++)
	{
		result[i] = a[i] + b[i];
		if (i % 100 == 0)//当计算了100个后，沉睡0.01s
			Sleep(10);
	}
}


int main()
{
	HANDLE Threads[num_of_CPUs];
	int a[N];
	int b[N];
	for (int i = 0; i < N; i++)//利用随机数初始化数据
	{
		a[i] = rand() % 100 + 1;
		b[i] = rand() % 100 + 1;
	}

	clock_t start1 = clock();//程序段开始前取得系统运行时间(s)  

	for (int i = 0; i < num_of_CPUs; ++i)
	{
		ThreadParam* p = new ThreadParam(i*N / num_of_CPUs, (i + 1)*N / num_of_CPUs, result1, a, b);
		//在不能整除时，把后面的都处理掉
		if (i == num_of_CPUs - 1)
			p->end = N - 1;
		Threads[i] = CreateThread(NULL, 0, addvalues2, p, 0, NULL);
	}
	WaitForMultipleObjects(num_of_CPUs, Threads, true, INFINITE);
	clock_t end1 = clock();//程序段结束后取得系统运行时间(s)  

	
	/*clock_t start2 = clock();
	for (int i = 0; i < num_of_CPUs1; ++i)
	{
		ThreadParam* p = new ThreadParam(i*N / num_of_CPUs1, (i + 1)*N / num_of_CPUs1, result1, a, b);
		if (i == num_of_CPUs1 - 1)
			p->end = N - 1;//在不能整除时，把后面的都处理掉
		Threads[i] = CreateThread(NULL, 0, addvalues2, p, 0, NULL);
	}
	WaitForMultipleObjects(num_of_CPUs1, Threads, true, INFINITE);
	clock_t end2 = clock();

	clock_t start4 = clock();
	for (int i = 0; i < num_of_CPUs2; ++i)
	{
		ThreadParam* p = new ThreadParam(i*N / num_of_CPUs2, (i + 1)*N / num_of_CPUs2, result1, a, b);
		//在不能整除时，把后面的都处理掉：
		if (i == num_of_CPUs2 - 1)
			p->end = N - 1;
		Threads[i] = CreateThread(NULL, 0, addvalues2, p, 0, NULL);
	}
	WaitForMultipleObjects(num_of_CPUs2, Threads, true, INFINITE);
	clock_t end4 = clock();*/

	clock_t start3 = clock();
	addvalues3(a, b, result3);
	clock_t end3 = clock();
	cout << "4级线程级并行时间(s)：" << (double)(end1 - start1) / CLOCKS_PER_SEC << endl;
	//cout << "6级线程级并行时间(s)：" << (double)(end2 - start2) / CLOCKS_PER_SEC << endl;
	//cout << "8级线程级并行时间(s)：" << (double)(end4 - start4) / CLOCKS_PER_SEC << endl;
	cout << "普通串行时间(s)：" << (double)(end3 - start3) / CLOCKS_PER_SEC << endl;

	cout << "4级加速比为" << (double)(end3 - start3) / (double)(end1 - start1)<<endl;
	//cout << "6级加速比为" << (double)(end3 - start3) / (double)(end2 - start2) << endl;
	//cout << "8级加速比为" << (double)(end3 - start3) / (double)(end4 - start4) << endl;
	system("pause");
	return 0;
}
