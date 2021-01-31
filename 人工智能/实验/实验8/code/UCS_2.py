import heapq
import matplotlib.pyplot as plt

actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]  # up/down/left/right; x means up/down, y means left/right


class Node:
    def __init__(self, location, cost=0, father=None):
        self.location = location
        self.cost = cost    # min cost to arrive at this node
        self.father = father

    def __lt__(self, other):        # redefine '<'(used in heappush)
        return self.cost < other.cost


def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        maze = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()     # remove the blank
            s_col = line.find('S')
            e_col = line.find('E')
            if s_col != -1:
                S = Node([i, s_col])      # create a S node
            if e_col != -1:
                E = Node([i, e_col])
            maze.append(line)
    return maze, S, E


def usc_search(maze, S, E):       # USC search
    count = 1   # time-complexity
    length = 1  # space-complexity
    row, col = len(maze), len(maze[0])
    frontier = []   # use heapq to construct a priority-queue
    explored = [[False] * col for _ in range(row)]      # False:unexplored, True:explored
    heapq.heappush(frontier, S)     # push S to heapq
    X, Y = [], []   # to plot
    while 1:
        if len(frontier) == 0:  # frontier is empty
            return None, X, Y
        curNode = heapq.heappop(frontier)   # current explored node
        if curNode.location == E.location:  # find target node
            X.append(curNode.location[1])
            Y.append(-curNode.location[0])  # '-' since up/down is converse when constructing the maze(readfile)
            print("Time complexity is %d" % count)
            print("Space complexity is %d" % length)
            return curNode, X, Y
        # update explored for curNode
        explored[curNode.location[0]][curNode.location[1]] = True
        count += 1  # update time
        X.append(curNode.location[1])
        Y.append((-curNode.location[0]))

        for action in actions:
            new_location = [x + y for x, y in zip(curNode.location, action)]
            # judge whether new_location is valid or not
            if new_location[0] < 0 or new_location[0] >= row or new_location[1] < 0 or new_location[1] >= col:
                continue
            # if new_location isn't explored and can be explored
            if explored[new_location[0]][new_location[1]] is False and maze[new_location[0]][new_location[1]] != '1':
                flag = True     # symbolizes whether new_location is in frontier
                for node in frontier:       # check whether new_node is already in the frontier
                    if node.location == new_location and node.cost > curNode.cost + 1:  # if in and cost is less, update
                        node.cost = curNode.cost + 1
                        node.father = curNode
                        heapq.heapify(frontier)     # update heapq
                        flag = False
                        break
                if flag:    # if new_location not in frontier, insert into frontier
                    new_node = Node(new_location, curNode.cost + 1, curNode)
                    heapq.heappush(frontier, new_node)
                    if len(frontier) > length:      # update length
                        length = len(frontier)


def draw_explored(X, Y):
    plt.figure(2)
    plt.xlim(0, 36)     # 0~35
    plt.ylim(-18, 0)    # -17~0
    plt.title('Explored nodes of UCS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.annotate('S', xy=(S.location[1], -S.location[0]), xytext=(S.location[1] + 0.5, -S.location[0]+0.1))
    plt.annotate('E', xy=(E.location[1], -E.location[0]), xytext=(E.location[1] - 0.5, -E.location[0]+0.5))
    plt.scatter(X, Y)   # discrete nodes
    # plt.plot(X,Y)
    plt.show()


def draw_route(node, S, E):
    X, Y = [], []
    curNode = node
    while curNode is not None:
        X.append(curNode.location[1])
        Y.append(-curNode.location[0])
        curNode = curNode.father
    plt.figure(3)
    plt.xlim(0, 36)  # 0~35
    plt.ylim(-18, 0)  # -17~0
    plt.title('Route of UCS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.annotate('S', xy=(S.location[1], -S.location[0]), xytext=(S.location[1] + 0.5, -S.location[0] + 0.1))
    plt.annotate('E', xy=(E.location[1], -E.location[0]), xytext=(E.location[1] - 0.5, -E.location[0] + 0.5))
    plt.plot(X, Y)      # continuous graph
    plt.show()


def draw_maze(maze, S, E, node):
    X, Y = [], []
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=='1':
                X.append(j)
                Y.append(-i)
    X1, Y1 = [], []
    curNode = node
    while curNode is not None:
        X1.append(curNode.location[1])
        Y1.append(-curNode.location[0])
        curNode = curNode.father
    plt.figure(1)
    plt.xlim(0, 36)  # 0~35
    plt.ylim(-18, 0)  # -17~0
    plt.title('Maze')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.annotate('S', xy=(S.location[1], -S.location[0]), xytext=(S.location[1] + 0.5, -S.location[0] + 0.1))
    plt.annotate('E', xy=(E.location[1], -E.location[0]), xytext=(E.location[1] - 0.5, -E.location[0] + 0.5))
    plt.plot(X1, Y1, color='orange')  # continuous graph
    plt.scatter(X, Y)  # continuous graph
    plt.show()


if __name__ == "__main__":
    maze, S, E = readfile("MazeData.txt")
    result, X, Y = usc_search(maze, S, E)
    draw_explored(X, Y)     # plot the explored nodes
    if result is None:
        print("无路径可到达终点")
    else:
        draw_maze(maze, S, E, result)  # plot the maze
