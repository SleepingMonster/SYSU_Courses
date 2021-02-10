#include<iostream>
#include<time.h>
#include"MutliAccessQueue.h"

using namespace std;
MultiAccessQueue<int> q;
void producer(int cnt)//每个生产者cnt个数，
{
	for (int i = 0; i < cnt; ++i)
		q.push(i);
}
void consumer(int cnt)
{
	for (int i = 0; i < cnt; ++i)
		q.pop();
}

int main()
{
	int num = 0;
	cout << "请输入生产者-消费者的数量：";
	cin >> num;
	omp_init_lock(&(q.back_mutex));
	omp_init_lock(&(q.front_mutex));
	clock_t start = clock();
	omp_set_num_threads(omp_get_num_procs());//设置线程数
#pragma omp parallel for
	for (int i = 0; i < num; ++i)//多个生产者消费者并行
	{
		int thread_id = omp_get_thread_num();
		if (thread_id < omp_get_num_threads() / 2)
			producer(250);
		else
			consumer(250);
	}
	clock_t end = clock();
	//cout << q.cnt << endl;
	cout <<num<<"个生产者-消费者的时间为：(s)"<< (double)(end - start) / CLOCKS_PER_SEC << endl;
	omp_destroy_lock(&(q.front_mutex));
	omp_destroy_lock(&(q.back_mutex));
	return 0;
}
