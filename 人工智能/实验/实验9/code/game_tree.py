from graphics import *


AI_pos = set()  # ai下的棋的落子位置
Man_pos = set()  # 玩家下的棋的落子位置
All_pos = set()  # 当前整个棋局所有落子的位置
ChessBoard = set()  # 整个棋盘中可以落子的位置

GRID_WIDTH = 40     # 画图时的格子宽度
col = 11
row = 11
MAX_DEPTH = 3   # 表示考虑三步后分数最大的位置落子

first_hand = 1  # 1是人先走  0是机先走
AI_FIRST_DEFAULT = (col // 2, row // 2)  # 如果AI先手，默认移动的位置

# 评估分数：针对各个棋型给出分数
shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (50000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]


# 评价函数：计算当前棋局的得分（人是MAX节点，AI是MIN节点）
def evaluation():
    # 算玩家自己的得分
    score_all_arr_player = []  # 得分形状的位置 用于计算如果有相交 得分翻倍
    player_score = 0
    for pt in Man_pos:
        m = pt[0]   # 横坐标
        n = pt[1]   # 纵坐标
        # 计算四个方向的总得分
        player_score += cal_score(m, n, 0, 1, AI_pos, Man_pos, score_all_arr_player) # 水平左右方向
        player_score += cal_score(m, n, 1, 0, AI_pos, Man_pos, score_all_arr_player)   # 竖直上下方向
        player_score += cal_score(m, n, 1, 1, AI_pos, Man_pos, score_all_arr_player)    # 左下->右上方向
        player_score += cal_score(m, n, -1, 1, AI_pos, Man_pos, score_all_arr_player)   # 左上->右下方向

    #  算ai的得分， 并减去
    score_all_arr_ai = []
    AI_score = 0
    for pt in AI_pos:
        m = pt[0]
        n = pt[1]
        AI_score += cal_score(m, n, 0, 1, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, 1, 0, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, 1, 1, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, -1, 1, Man_pos, AI_pos, score_all_arr_ai)

    return player_score - AI_score * 0.1    # ！！！


# 计算当前步数得分：遍历每个方向
def cal_score(m, n, x_direct, y_direct, enemy_list, my_list, score_all_arr):
    add_score = 0  # 加分项
    # 在一个方向上， 只取最大的得分项
    max_score_shape = (0, None)     # 格式：max_score, 5个位置, 方向(delta_x, delta_y)。同样适用与score_all_arr

    # 如果此方向上，该点已经有得分形状，不重复计算
    for item in score_all_arr:
        for pt in item[1]:
            if m == pt[0] and n == pt[1] and x_direct == item[2][0] and y_direct == item[2][1]:
                return 0

    # 在落子点 左右方向上循环查找得分形状
    for offset in range(-5, 1):
        # offset = -2
        pos = []
        for i in range(0, 6):
            if (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in enemy_list:
                pos.append(2)  # 敌人标记为2
            elif (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in my_list:
                pos.append(1)  # 自己标记为1
            else:
                pos.append(0)  # 空标记为0
        tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])    # 取5个点
        tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])    # 取6个点

        for (score, shape) in shape_score:  # 评估分数
            if tmp_shap5 == shape or tmp_shap6 == shape:    # tmp_shape和评估份数中的相匹配
                if score > max_score_shape[0]:
                    max_score_shape = (score, ((m + (0 + offset) * x_direct, n + (0 + offset) * y_direct),
                                               (m + (1 + offset) * x_direct, n + (1 + offset) * y_direct),
                                               (m + (2 + offset) * x_direct, n + (2 + offset) * y_direct),
                                               (m + (3 + offset) * x_direct, n + (3 + offset) * y_direct),
                                               (m + (4 + offset) * x_direct, n + (4 + offset) * y_direct)),
                                       (x_direct, y_direct))

    # 计算两个形状相交， 如两个3活相交，得分增加（一个子的除外）
    if max_score_shape[1] is not None:
        for item in score_all_arr:  # 查看别的方向上的得分形状
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:  # 如果存在两个得分形状有点重合
                    if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]  # 将重合的形状得分翻倍

        score_all_arr.append(max_score_shape)   # 由于传的是引用，故可以成立

    return add_score + max_score_shape[0]


# 判断四周（8个方向）是否有棋子，如果都无就不考虑这个点，加快搜索速度
def neighbour(point):
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [1, 1, 1, 0, 0, -1, -1, -1]
    for i in range(8):
        if (point[0] + x[i], point[1] + y[i]) in All_pos:
            return True
    return False


# 判断上一步下完后,是否有一方取胜(往8个方向延伸5步并做边界检测，判断）
def game_win(point, is_ai):
    dir_list = ([(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(1, 1), (-1, -1)])  # 记录八个方向
    if is_ai is True:
        mylist = AI_pos
    else:
        mylist = Man_pos
    for dir_pair in dir_list:
        seq = 1
        for dir in dir_pair:
            cur = point
            while ((cur[0] + dir[0], cur[1] + dir[1]) in mylist) and row > cur[0] and cur[0]>= 0 and col > cur[1] and cur[1]>= 0:
                seq += 1
                if seq >= 5:
                    return True
                cur = (cur[0] + dir[0], cur[1] + dir[1])
    return False


def get_available_pos():
    return ChessBoard - All_pos


# 极大节点 人走
def MiniMax(last_step, depth, max_alpha, min_beta):
    # 判断是不是终止状态
    if game_win(last_step, True) or depth == 0:
        return evaluation(), (-1, -1)
    alpha = float('-inf')  # 设定alpha为无穷小
    pos = (-1, -1)
    # 获得可以走的位置tuple
    available_pos = get_available_pos()
    for pos1 in available_pos:   # 回溯的思想
        # 忽略周围没有棋子的位置（不值得下）
        if neighbour(pos1) is False:
            continue
        Man_pos.add(pos1)
        All_pos.add(pos1)
        alpha_temp, _ = MiniMin(pos1, depth - 1, max_alpha, min_beta)   # 查看子节点：depth-1！
        # 更新当前节点的alpha值
        if alpha < alpha_temp:
            alpha = alpha_temp
            pos = pos1
        Man_pos.remove(pos1)
        All_pos.remove(pos1)

        # 判断要不要alpha剪枝
        if alpha > min_beta:
            return alpha, pos
        # 如果要继续搜索，查看是否需要更新当前最大的alpha值（用来传给子节点用，可以剪枝）
        if alpha >= max_alpha:
            max_alpha = alpha
    return alpha, pos


# 极小节点 AI走
def MiniMin(last_step, depth, max_alpha, min_beta):
    # 判断是不是终止状态
    if game_win(last_step, False) or depth == 0:
        return evaluation(), (-1, -1)
    beta = float('inf')  # 设定beta为无穷大
    pos = (-1, -1)
    # 获得可以走的位置tuple
    available_pos = get_available_pos()
    for pos1 in available_pos:
        # 忽略周围没有棋子的位置（不值得下）
        if not neighbour(pos1):
            continue
        AI_pos.add(pos1)
        All_pos.add(pos1)
        beta_temp, _ = MiniMax(pos1, depth - 1, max_alpha, min_beta)
        if beta > beta_temp:  # 将子节点中最小的返回值赋值给当前节点的beta值
            beta = beta_temp
            pos = pos1
        AI_pos.remove(pos1)
        All_pos.remove(pos1)

        # beta 剪枝
        if beta < max_alpha:
            return beta, pos
        # 如果要继续搜索，更新当前最小的beta值（用来传给子节点用，可以剪枝）
        if beta < min_beta:
            min_beta = beta

    return beta, pos


def gobangwin():
    win = GraphWin("gobang game by try", GRID_WIDTH * col, GRID_WIDTH * row)
    win.setBackground("light yellow")
    i1 = 0

    while i1 <= GRID_WIDTH * col:
        l = Line(Point(i1, 0), Point(i1, GRID_WIDTH * col))
        l.draw(win)
        i1 = i1 + GRID_WIDTH
    i2 = 0

    while i2 <= GRID_WIDTH * row:
        l = Line(Point(0, i2), Point(GRID_WIDTH * row, i2))
        l.draw(win)
        i2 = i2 + GRID_WIDTH
    return win


def main():
    win = gobangwin()
    if first_hand:
        Man_color = 'black'
        AI_color = 'white'
    else:
        Man_color = 'white'
        AI_color = 'black'
    for pos in Man_pos:
        piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
        piece.setFill(Man_color)
        piece.draw(win)
    for pos in AI_pos:
        piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
        piece.setFill(AI_color)
        piece.draw(win)

    for i in range(col+1):
        for j in range(row+1):
            ChessBoard.add((i, j))

    count = 0
    is_terminal = 0
    m = 0
    n = 0
    last_step = (-1, -1)
    while is_terminal == 0:
        # AI下棋
        if count % 2 == first_hand:
            _, pos = MiniMin(last_step, MAX_DEPTH, -float("inf"), float("inf"))

            if pos in All_pos:
                message = Text(Point(200, 200), "不可用的位置" + str(pos[0]) + "," + str(pos[1]))
                message.draw(win)
                is_terminal = 1

            AI_pos.add(pos)
            All_pos.add(pos)
            print("第%d回合电脑落子得分为：" % (count // 2 + 1), evaluation())

            piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
            piece.setFill(AI_color)
            piece.draw(win)

            if game_win(pos, True):
                message = Text(Point(100, 100), "Computer WIN!")
                message.draw(win)
                is_terminal = 1
            count = count + 1
        # 玩家下棋
        else:
            p2 = win.getMouse()     # 获得鼠标
            if not ((round((p2.getY()) / GRID_WIDTH), round((p2.getX()) / GRID_WIDTH)) in All_pos):
                # round函数返回四舍五入值
                a2 = round((p2.getX()) / GRID_WIDTH)    # 获得坐标
                b2 = round((p2.getY()) / GRID_WIDTH)
                Man_pos.add((b2, a2))
                All_pos.add((b2, a2))
                print("第%d回合用户落子得分为：" % (count // 2 + 1), evaluation())
                last_step = (b2, a2)

                piece = Circle(Point(GRID_WIDTH * a2, GRID_WIDTH * b2), 16)     # 画图
                piece.setFill(Man_color)
                piece.draw(win)
                if game_win(last_step, False):
                    message = Text(Point(100, 100), "Player WIN!")
                    message.draw(win)
                    is_terminal = 1

                count = count + 1

    message = Text(Point(100, 120), "Click anywhere to quit.")
    message.draw(win)
    win.getMouse()
    win.close()


if __name__ == "__main__":
    first_hand = int(input("请选择玩家先手1 还是电脑先手0  请输入："))

    if first_hand:  # 此时是人先手 人是x
        print("人先手")
        AI_pos.add((5, 4))
        AI_pos.add((6, 5))
        Man_pos.add((5, 5))
        Man_pos.add((5, 6))
    else:
        print("电脑先手")
        Man_pos.add((5, 4))
        Man_pos.add((6, 5))
        AI_pos.add((5, 5))
        AI_pos.add((5, 6))
    All_pos.add((5,4))
    All_pos.add((6,5))
    All_pos.add((5,5))
    All_pos.add((5,6))

    main()