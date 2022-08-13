


import numpy as np
from util import *
from SGA_setting import *
from matplotlib import pyplot as plt
import math
import time





def initialize_group(initial_group_size,city_num):
    #根据初始种群大小，生成第一代个体
    #返回初始种群个体集合：groups[i] 第i个个体的基因序列
    groups=[]
    for i in range(initial_group_size):
        groups.append(initialize_path(city_num))
    return np.array(groups)


def fitness(seq,node_list):
    return 1/calculate_route(seq,node_list)


def wheel_select(group,node_list,size):
    #通过轮盘算法，得到选择之后的种群，返回group
    
    
    
    #通过轮盘算法得到生存的个体
    
    #计算每一个个体的适应值
    fitness_list=[]
    for item in group:
        fitness_list.append(fitness(item,node_list))
    fitness_sum=sum(fitness_list) #总的适应值
    #计算每一个个体被选择的累计概率
    sum_p=0 
    cumulative_p=[]#每一个个体被选择累计概率
    for item in fitness_list:
        p=item/fitness_sum
        cumulative_p.append(p+sum_p)
        sum_p+=p
    #print(cumulative_p)
    
    rand_num = np.random.uniform(low=0.0,high=1.0,size=size) #生成group_size个随机小数
    choice_index=np.searchsorted(cumulative_p,rand_num) #得到选出的个体对应的下标
    new_group=group[choice_index]
    return new_group

def elite_select(group,node_list,size):
    
    len_list=[]
    for item in group:
        len_list.append(calculate_route(item,node_list))
    choice_index=np.argsort(len_list)
    choice_index=choice_index[0:size]
    new_group=group[choice_index]
    return new_group

def select(group,node_list):
    if len(group)<=group_size: #总数比最大种群数小，不需要选择
        return group
    elite_size=math.ceil(elite_rate*group_size)
    new_group=elite_select(group,node_list,elite_size)
    new_group=np.vstack((new_group,wheel_select(group,node_list,(group_size-elite_size))))
    return new_group


def intersect_2_seq(seq1,seq2): 
    #对两个序列进行交叉操作
    pos1,pos2=get_2_points(len(seq1))
    #建立映射关系表
    dict1=dict() #seq1到seq2
    dict2=dict() #seq2到seq1
    for i in range(pos1,pos2):
        dict1[seq1[i]]=seq2[i]
        dict2[seq2[i]]=seq1[i]
    
    #生成两个新的序列
    new_seq1=[0]*len(seq1)
    new_seq2=[0]*len(seq2)
    for i in range(pos1):
        temp=seq1[i]
        while temp in dict2:
            temp=dict2[temp]        
        new_seq1[i]=temp
        
        temp=seq2[i]
        while temp in dict1:
            temp=dict1[temp]        
        new_seq2[i]=temp
        
    for i in range(pos1,pos2):
        temp=seq1[i]
        new_seq1[i]=seq2[i]
        new_seq2[i]=temp
        
    for i in range(pos2,len(seq1)):
        temp=seq1[i]
        while temp in dict2:
            temp=dict2[temp]        
        new_seq1[i]=temp
        
        temp=seq2[i]
        while temp in dict1:
            temp=dict1[temp]        
        new_seq2[i]=temp
    return new_seq1,new_seq2


def intersect(group):
    #根据交叉算法，对整个种群进行交叉，得到交叉之后的新种群
    new_group=[]
    for i in range(len(group)//2):
        rand_num=np.random.uniform(low=0.0,high=1.0)
        if rand_num>intersect_p:
            continue
        new_seq1,new_seq2=intersect_2_seq(group[2*i],group[2*i+1])
        new_group.append(new_seq1)
        new_group.append(new_seq2)
    return np.array(new_group)
    
def variation(group,choice):
    #对group内的每一个个体，以一定的概率进行变异
    #choice 变异类型
    new_group=[]
    for i in range(len(group)):
        rand_num=np.random.uniform(low=0.0,high=1.0)
        if rand_num>variation_p:
            new_group.append(groups[i])
            continue
        new_group.append(local_search(group[i],choice))
    return np.array(new_group)

def find_best_path(groups,city):
    #从种群中找到最优的路径和路径长度
    best_len=calculate_route(groups[0],city)
    index=0
    for i in range(len(groups)):
        len1=calculate_route(groups[i],city)
        if len1<best_len:
            best_len=len1
            index=i
    #print(index)
    return groups[index],best_len


if __name__ == '__main__':

    node_pos_list=read_data("../tc/lin105.tsp") #这里可以选../tc/lin105.tsp(测试样例1) 或者  ../tc/ch130.tsp(测试样例2)
    opt_route = get_opt_route("../tc/lin105.opt.tour") #这里可以选../tc/lin105.opt.tour(测试样例1) 或者  ../tc/ch130.opt.tour(测试样例2)
    city_num=len(node_pos_list)
    groups=initialize_group(initial_group_size,city_num)

    best_len_list=[]
    best_route_list=[]
    total_time=0
    a=time.time()
    for i in range(generation_times):
        child_groups1=intersect(groups)
        child_groups2=variation(child_groups1,choice)
        all_groups=np.vstack((child_groups1,child_groups2))
        all_groups=np.vstack((groups,all_groups))
        groups=select(all_groups,node_pos_list)
        best_route,best_len=find_best_path(groups,node_pos_list)
        if i%100==0:
            best_route_list.append(best_route)
        best_len_list.append(best_len)
    b=time.time()
    total_time+=(b-a)
    print("迭代%d次,总运行时间为%.2f,最短路径长度为%f"%(generation_times,total_time,best_len))

    #下面画图
    best_route = best_route.tolist()
    best_route.append(best_route[0])
    best_route = np.array(best_route)
    count = [i for i in range(len(best_len_list))]
    plt.plot(count, best_len_list)
    plt.title("Total distance(SGA)")
    plt.show()
    draw_route(best_route, node_pos_list, "blue", "Current best route of SGA")
    draw_route(opt_route, node_pos_list, "orange", "Optimal route")
    draw_2_routes(best_route, opt_route, node_pos_list, "SGA", "Optimal route", "Routes")




