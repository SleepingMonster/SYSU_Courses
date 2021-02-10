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
	omp_init_lock(&(q.back_mutex));
	omp_init_lock(&(q.front_mutex));
	clock_t start = clock();

#pragma omp parallel num_threads(2)
{
	#pragma omp sections
	{
		#pragma omp section
		{
			producer(250);
		}
		#pragma omp section
		{
			consumer(250);
		}
	}
}
	clock_t end = clock();
	cout << "1个生产者-消费者的时间为：(s)" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	omp_destroy_lock(&(q.front_mutex));
	omp_destroy_lock(&(q.back_mutex));
	return 0;
}