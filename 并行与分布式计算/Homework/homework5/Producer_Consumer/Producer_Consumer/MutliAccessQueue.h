#pragma once
#include<queue>
#include<iostream>
#include<omp.h>
using namespace std;
template<class T>
class MultiAccessQueue :queue<T>//继承queue
{	
public:
	omp_lock_t back_mutex;//push用锁
	omp_lock_t front_mutex;//pop用锁
	int cnt = 0;
	void push(T val)
	{
		omp_set_lock(&back_mutex);//获得锁
		queue<T>::push(val);
		omp_unset_lock(&back_mutex);
		return;
	}
	void pop()
	{
		omp_set_lock(&front_mutex);//锁要加在外面，否则可能会出错
		//while (queue<T>::empty());
		if (!queue<T>::empty())
		{
			queue<T>::pop();
			//cnt++;
		}
			
		omp_unset_lock(&front_mutex);
		return;
	}
};
