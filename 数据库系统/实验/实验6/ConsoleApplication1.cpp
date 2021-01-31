#define _CRT_SECURE_NO_WARNINGS
//#include "pch.h"
#include <mysql.h>
#include <stdio.h>
#include <winsock.h>
#include <iostream>

using namespace std;
MYSQL mysql;	//声明为全局变量
MYSQL_RES* result;

int create_course_table()
{
	char yn[2];
	result = mysql_list_tables(&mysql, "course");//删除表
	unsigned int rows = mysql_num_rows(result);
	mysql_free_result(result);
	if (rows > 0)
	{
		printf("The course table already exists. Do you want to delete it?\n");
		printf("Delete the table?(y--yes,n--no):");
		scanf("%s", &yn);
		if (yn[0] == 'y' || yn[0] == 'Y')	//考虑大小写
		{
			if (!mysql_query(&mysql, "drop table course:"))
			{
				printf("drop table course successfully!\n");
			}
			else
			{
				printf("error:drop table course failed.\n");
			}
		}
		else  //使用原来的表,exit directly
		{
			return 0;
		}
	}
	int num = mysql_query(&mysql, "create table course(cno char(10) primary key,cname char(20),cpno char(10) default null,ccredit char(10)) engine=innodb;");
	if (num == 0)
		printf("create table course successully!\n\n");
	else
		printf("ERROR: create table course\n\n");
	return 0;
}

int insert_rows_into_course_table()
{
	while (1)
	{
		char icno[10] = "0";
		char icname[20] = "xx";
		char icpno[10] = "100";
		char iccredit[10] = "3";
		char yn[2];	//用来判断是否继续插入
		char strquery[200] = "insert into course(cno,cname,cpno,ccredit) values('";
		//insert cno
		printf("Please input cno:");
		scanf("%s", icno);
		strcat(strquery, icno);
		strcat(strquery, "','");
		//insert cname
		printf("Please input cname:");	//注意！！用scanf不能读空格！所以只能打下划线连接课程名。
		scanf("%s", icname);
		strcat(strquery, icname);
		strcat(strquery, "','");
		//insert 先修课程号
		printf("Please input cpno:");
		scanf("%s", icpno);
		strcat(strquery, icpno);
		strcat(strquery, "','");
		//insert ccredit
		printf("Please input ccredit:");
		scanf("%s", iccredit);
		strcat(strquery, iccredit);
		strcat(strquery, "');");	//输到sql中以分号结尾
		printf("%s\n", strquery);
		//check whether execute successfully
		if (mysql_query(&mysql, strquery) == 0)
		{
			printf("execute successfully!\n");
		}
		else
		{
			printf("error:execute failed\n");
			printf("%s", mysql_error(&mysql));
		}
		//判断是否需要继续插入记录
		printf("Insert again?(y/n):");
		scanf("%s", &yn);
		printf("\n");
		if (yn[0] == 'y' || yn[0] == 'Y')
			continue;
		else
			break;
	}
	return 0;
}

int main(int argc, char** argv, char** envp)
{
	char func[2];//储存要执行的操作的变量
	mysql_init(&mysql);	//初始化一个MYSQL结构
	//mysql_real_connect()连接到MYSQL数据库服务器。"localhost"为服务器名，
	//"root"为连接用户名，123456为密码，hello为数据库名，3306位连接端口号
	mysql_options(&mysql, MYSQL_SET_CHARSET_NAME, "gbk");//插入这句话可以使得sql输入中文。
	if (mysql_real_connect(&mysql, "localhost", "root", "123456", "hello", 3306, 0, 0))
	{
		for (;;)
		{
			printf("Sample Embedded SQL for C application\n");
			printf("Please select one function to excute:\n\n");
			printf("0--exit.\n");
			printf("1--创建课程表  2--添加课程记录\n");
			scanf("%s", &func);
			if (func[0] == '0')
				break;
			else if (func[0] == '1')
				create_course_table();
			else if (func[0] == '2')
				insert_rows_into_course_table();
		}
	}
	else
	{
		printf("数据库不存在！\n");
	}
	mysql_close(&mysql);//访问完毕，关闭数据库mysql
	result = mysql_store_result(&mysql);
	mysql_free_result(result);
	system("pause");
	return 0;
}

