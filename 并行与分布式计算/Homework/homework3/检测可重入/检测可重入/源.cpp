#include<iostream>
#include<fstream>
#include<string>
using namespace std;
int main()
{ //用string.find来实现找有没有defind\@等
	fstream file("test.ll");
	string line;
	bool reentrant = true;//标志是否可重入
	if (file)
	{
		bool inside = false;//是否进入函数
		while (getline(file, line))
		{
			if (inside == false)//不在函数中
			{
				if (line.find("define") != line.npos)
					inside = true;
			}
			else//已经进入函数中
			{
				if (line == "}")//到达函数尾部
					inside = false;
				else if(line.find("call")!=line.npos)//调用了下面这些函数
				{
					if (line.find("@printf") != line.npos || line.find("@scanf")!=line.npos ||line.find("@puts") != line.npos || 
						line.find("@gets") != line.npos ||line.find("@malloc") != line.npos || line.find("@new") != line.npos ||
						line.find("@free") != line.npos)
					{
						reentrant = false;
					}
				}
				else//没有调入输入输出、开辟空间等函数,则判断有无全局变量
				{
					if (line.find("@")!=line.npos)
						reentrant = false;
				}
			}
		}
		if (reentrant)
			cout << "程序中不存在不可重入的函数" << endl;
		else
			cout<< "程序中存在不可重入的函数" << endl;
	}
	else
	{
		cout << "No such file!" << endl;
	}
	system("pause");
	return 0;
}