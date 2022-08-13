import random
import math
import time
import matplotlib.pyplot as plt
# import fig
from util import *

# SA 参数
T_start = 50000     # initial tempreture
T_end = 1e-8         # terminal tempreture
N = 2000             # inner loop time(iteration time for each tempreture)


def Local_Search_only(choice, TEST):
    if TEST == 1:
        file1 = "../tc/lin105.tsp"
        file2 = "../tc/lin105.opt.tour"
    elif TEST == 2:
        file1 = "../tc/ch130.tsp"
        file2 = "../tc/ch130.opt.tour"
    city = read_data(file1)
    t1 = time.time()
    route = initialize_path(len(city))
    count = 0
    dist = []
    dist_index = []
    while count <= LOOP_TIME:
        new_route = local_search(route, choice)
        d1 = calculate_route(route, city)
        d2 = calculate_route(new_route, city)
        diff = d2 - d1
        if diff<0:   # good solution
            route = new_route
            dist.append(d2)
        else:
            dist.append(d1)
        count += 1
        dist_index.append(count)
    t2 = time.time()
    print("Duration for only Local Search(s): ", t2-t1)
    print("Length of the best path is :", calculate_route(route, city))
    print("The best solution for %d loops is :\n" %(LOOP_TIME), route)
    draw_route(route, city, color='blue', title='Current best route of Local Search only')
    opt_route = get_opt_route(file2)
    draw_route(opt_route, city, color='orange', title='Optimal route')
    temp_title = 'Routes(LOOP_TIME = '+ str(LOOP_TIME) +' for Local Search)'
    draw_2_routes(route, opt_route, city, label1='Local Search only', label2='Optimal route', title=temp_title)
    draw_dist(dist_index, dist, title='Total Distance (Local Search only)')


def SA(choice, TEST):
    if TEST == 1:
        file1 = "../tc/lin105.tsp"
        file2 = "../tc/lin105.opt.tour"
    elif TEST == 2:
        file1 = "../tc/ch130.tsp"
        file2 = "../tc/ch130.opt.tour"
    city = read_data(file1)
    t1 = time.time()
    route = initialize_path(len(city))
    count = 0
    dist = []
    dist_index = []
    T = T_start
    T_index = 0
    while T > T_end:
        for count in range(N):
            new_route = local_search(route, choice)
            d1 = calculate_route(route, city)
            d2 = calculate_route(new_route, city)
            diff = d2 - d1
            r = random.random()
            if diff>0 and math.exp(-diff / T) <= r:   # bad solution and not accepted
                dist.append(d1)
            else:          # good solution or bad one being accepted
                route = new_route
                dist.append(d2)
            dist_index.append(T_index * N + count)
        T = 1 / math.log10(1+1+T_index) * T
        T_index += 1
        
    t2 = time.time()
    loop_count = len(dist_index)
    print("Duration for Simulated Annealing(s): ", t2-t1)
    print("Length of the best path is :", calculate_route(route, city))
    print("The best solution for %d loops is :\n" %(loop_count), route)
    draw_route(route, city, color='blue', title='Current best route of SA')
    opt_route = get_opt_route(file2)
    draw_route(opt_route, city, color='orange', title='Optimal route')
    temp_title = 'Routes(T_start='+str(T_start)+',T_end='+str(T_end)+',N='+ str(N) +' for SA)'
    draw_2_routes(route, opt_route, city, label1='SA', label2='Optimal route', title=temp_title)
    draw_dist(dist_index, dist, title='Total Distance (SA)')


if __name__ == '__main__':
    method = "Local Search only"       # method = "Local Search only" or "SA"
    choice = 2
    TEST = 2   # 1-"Dataset/lin105.tsp/lin105.tsp";2 - "Dataset/ch130.tsp/ch130.tsp"
    if method == "Local Search only":
        Local_Search_only(choice=choice, TEST=TEST)
    elif method == "SA":
        SA(choice=choice, TEST=TEST)

