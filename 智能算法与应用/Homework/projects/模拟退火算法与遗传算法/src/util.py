import random
import math
import matplotlib.pyplot as plt

LOOP_TIME = 40000

def read_data(file_name):
    '''
    file_name: location of the .tsp file
    return: node_pos_list (the 2-D location of all nodes), node_pos_list[i]:[x_i, y_i]
    '''
    f = open(file_name)            
    line = f.readline()  
    count=0;
    node_pos_list=[] #节点的二维坐标 node_pos_list[i]表示节点 i+1 的[x_坐标，y_坐标]
    while line: 
        count+=1
        line = f.readline()
        if count>5 :
            temp=line.split()
            if len(temp)==3:
                temp1=[float(temp[1]),float(temp[2])]
                node_pos_list.append(temp1)
        
    f.close()
    return node_pos_list

def get_opt_route(file_name):
    f = open(file_name)
    line = f.readline()
    opt_node_list=[]   #存放最优路径的节点下标 （下标从1开始）
    count=0
    while line:
        #print(line)
        count+=1
        #print(line, end =" ")
        line = f.readline()
        if line=='-1\n':
            break
        if count>4 and line!='-1\n':
            line=line.strip('\n')
            if len(line) !=0:
                opt_node_list.append(int(line))
    f.close()
    opt_node_list.append(opt_node_list[0])
    return opt_node_list


def initialize_path(num):
    '''
    num = number of city；
    return initialized path
    '''
    a = range(num)
    b = random.sample(a, num)
    return b

def get_2_points(num):
    '''
    get 2 random points in the route
    return: 2 pos
    '''
    r1 = random.random()
    r2 = random.random()
    pos1 = (int)(num * r1)
    pos2 = (int)(num * r2)
    while(pos1==pos2):
        r1 = random.random()
        r2 = random.random()
        pos1 = (int)(num * r1)
        pos2 = (int)(num * r2)
    p1 = min(pos1, pos2)
    p2 = max(pos1, pos2)
    return p1, p2

def swap_2_points(route):
    '''
    To swap 2 random points in the route.
    return: new route
    '''
    pos1, pos2 = get_2_points(len(route))
    temp = route[pos1]
    route[pos1] = route[pos2]
    route[pos2] = temp
    return route

def reverse(route):
    '''
    reverse the route between two random points. Aka "2opt".
    return: new route
    '''
    pos1, pos2 = get_2_points(len(route))
    # print(p1, p2)
    i = pos1
    j = pos2
    while i<j:
        temp = route[i]
        route[i] = route[j]
        route[j] = temp
        i+=1
        j-=1
    return route

def swap_2_arrays(route):
    '''
    swap 2 arrays outside 2 random points.
    return: new route
    '''
    pos1, pos2 = get_2_points(len(route))
    # print(p1, p2)
    new_route = [0] * len(route)
    index = 0
    for i in range(pos2, len(route)):
        new_route[index] = route[i]
        index += 1
    for i in range(pos1, pos2):
        new_route[index] = route[i]
        index += 1
    for i in range(0, pos1):
        new_route[index] = route[i]
        index += 1
    return new_route
    
def extract_to_head(route):
    '''
    extract the array between two random points to the head of the route.
    return: new route
    '''
    pos1, pos2 = get_2_points(len(route))
    # print(p1, p2)
    index = 0
    new_route = [0] * len(route)
    for i in range(pos1, pos2):
        new_route[index] = route[i]
        index += 1
    for i in range(0, pos1):
        new_route[index] = route[i]
        index += 1
    for i in range(pos2, len(route)):
        new_route[index] = route[i]
        index += 1
    return new_route

def extract_to_tail(route):
    '''
    extract the array between two random points to the tail of the route.
    return: new route
    '''
    pos1, pos2 = get_2_points(len(route))
    # print(p1, p2)
    index = 0
    new_route = [0] * len(route)
    for i in range(0, pos1):
        new_route[index] = route[i]
        index += 1
    for i in range(pos2, len(route)):
        new_route[index] = route[i]
        index += 1
    for i in range(pos1, pos2):
        new_route[index] = route[i]
        index += 1
    return new_route


def local_search(route, choice):
    '''
    get the new route by using corresponding method.
    choice: method chosen
    return: new route
    '''
    result = []
    if choice == 1:        # swap
        result = swap_2_points(route[:])   # ！！！this can pass argument in value way
    elif choice == 2:      # 2opt
        result = reverse(route[:])
    elif choice == 3:      # swap arrays outside 2 points
        result = swap_2_arrays(route)
    elif choice == 4:      # extract array to head
        result = extract_to_head(route)
    elif choice == 5:      # extract array to tail
        result = extract_to_tail(route)
    return result


def calculate_distance(city, start, end):
    '''
    calculate the distance between two points: start and end
    '''
    x1 = city[start][0]
    y1 = city[start][1]
    x2 = city[end][0]
    y2 = city[end][1]
    return math.sqrt(pow(x1-x2, 2)+pow(y1-y2, 2))
    
def calculate_route(route, city):
    '''
    calculate the total distances of the whole route.
    '''
    result = 0
    for i in range(len(route)):
        if i==len(route)-1:
            result += calculate_distance(city, route[i], route[0])
        else:
            result += calculate_distance(city, route[i], route[i+1])
    return result


def draw_route(route, city, color, title):
    '''
    draw the route(current / optimal).
    route: serial order number of each city（存储城市序号）
    city: the city map
    color[!!]: blue-current best route(Local Search/SA/SGA); orange-optimal
    title: graph title
    '''
    X = []
    Y = []
    if color=='blue':
        for i in range(len(route)):
            X.append(city[route[i]][0])
            Y.append(city[route[i]][1])
    else:
        for i in range(len(route)):
            X.append(city[route[i]-1][0])
            Y.append(city[route[i]-1][1])
    plt.figure()
    plt.plot(X, Y, color=color, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()


def draw_2_routes(route, opt_route, city, label1, label2, title):
    '''
    draw two routes in one picture.
    '''
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(len(route)):
        X1.append(city[route[i]][0])
        Y1.append(city[route[i]][1])
    for i in range(len(opt_route)):
        X2.append(city[opt_route[i]-1][0])
        Y2.append(city[opt_route[i]-1][1])
    plt.figure()
    plt.plot(X1, Y1, label=label1)
    plt.plot(X2, Y2, label=label2, color='orange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

def draw_dist(x, y, title):
    '''
    draw the total distance-iteration number graph.
    x: iteration number, y: total distance
    '''
    plt.figure(4)
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('Iteration Number')
    plt.ylabel('Total Distance')
    plt.title(title)