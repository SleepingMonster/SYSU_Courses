import numpy as np
import random
import matplotlib.pyplot as plt


height = 4  # 对应y
width = 12  # 对应x
epsilon = 0.1
gamma = 1
alpha = 0.5


def take_action(x, y, a):   # 返回才取action a之后的位置，和reward
    destination = False
    if x == width - 1 and y == 0:
        destination = True
    act_move = [[0, 1], [1, 0], [0, -1], [-1, 0]]   # means up,right,down, left
    temp_x = x + act_move[a][0]
    temp_y = y + act_move[a][1]
    if 0<=temp_x<width and 0<=temp_y<height:
        x = temp_x
        y = temp_y
    if destination is True:
        return x, y, 0      # q(terminal_state) = 0
    if 0 < x < width-1 and y==0:
        return 0, 0, -100
    else:
        return x, y, -1


def max_q(x, y, q):     # 返回(x,y)状态下可采取获得的最大利润的action的下标
    max_q = q[x][y][0]
    max_action = 0
    for i in range(1, 4):  # [1,4)
        if q[x][y][i] >= max_q:
            max_q = q[x][y][i]
            max_action = i
    return max_action


def epsilon_policy(x, y, q, epsilon):   # 利用episilon-policy选取下一个动作
    random_action = random.randint(0,3)     # 包含两端
    if random.random() < epsilon:   # 随机出来的数是小于epsilon,表示随机选取aciton
        action = random_action
    else:   # 随机出来的数是大于epsilon，表示选择best_reward_action
        action = max_q(x, y, q)
    return action


def sarsa(q):
    runs = 20
    rewards = np.zeros(500)
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            action = epsilon_policy(x, y, q, epsilon)   # 根据epsilon_policy选择action
            while 1:
                x_next, y_next, reward = take_action(x, y, action)
                reward_sum += reward
                action_next = epsilon_policy(x_next, y_next, q, epsilon)
                q[x][y][action] += alpha * (reward + gamma * q[x_next][y_next][action_next] - q[x][y][action])
                if x == width - 1 and y == 0:
                    break
                x = x_next
                y = y_next
                action = action_next
            rewards[i] += reward_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i + 1]))
    for i in range(10, len(rewards) + 1):
        avg_rewards.append(np.mean(rewards[i - 10:i]))

    return avg_rewards


def q_learning(q):
    runs = 20
    rewards = np.zeros(500)
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            while 1:
                action = epsilon_policy(x, y, q, epsilon)  # 根据epsilon_policy选择action
                x_next, y_next, reward = take_action(x, y, action)
                reward_sum += reward
                action_next = max_q(x_next, y_next, q)
                q[x][y][action] += alpha * (reward + gamma * q[x_next][y_next][action_next] - q[x][y][action])
                if x == width - 1 and y == 0:
                    break
                x = x_next
                y = y_next
            rewards[i] += reward_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10, len(rewards) + 1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards


def print_optimal_path(q):
    x = 0
    y = 0
    path = np.zeros([width, height]) - 1
    flag = True
    exist = np.zeros([width, height])
    while (x != width - 1 or y != 0) and flag is True:
        action = max_q(x, y, q)
        path[x][y] = action
        if exist[x][y] == 1:
            flag = False
        exist[x][y] = 1
        x, y, reward = take_action(x, y, action)
    for j in range(height-1, -1, -1):
        for i in range(width):
            if i == width - 1 and j == 0:
                print("G ", end="")  # 利用end=空字符串可以不输出换行符
                continue
            temp = path[i][j]
            if temp == -1:
                print("* ", end="")
            elif temp == 0:
                print("↑ ", end="")
            elif temp == 1:
                print("→ ", end="")
            elif temp == 2:
                print("↓ ", end="")
            elif temp == 3:
                print("← ", end="")
        print("")


def main():
    q1 = np.zeros([12, 4, 4])
    q2 = np.zeros([12, 4, 4])
    sarsa_rewards = sarsa(q1)
    q_learning_rewards = q_learning(q2)
    # 画图:rewards
    plt.plot(range(len(sarsa_rewards)), sarsa_rewards, label="sarsa")
    plt.plot(range(len(q_learning_rewards)), q_learning_rewards, label="q_learning")
    plt.grid()
    plt.legend(loc="lower right")   # 把图标放在右下角
    plt.title('Reward per episode (average)')
    plt.xlabel('episode number')
    plt.ylabel('Reward')
    plt.show()
    print("Sarsa Optimal Path:")
    print_optimal_path(q1)
    print("\nQ-learning Optimal Path:")
    print_optimal_path(q2)


if __name__ == "__main__":
    main()
