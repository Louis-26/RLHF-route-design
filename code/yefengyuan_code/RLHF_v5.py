import numpy as np
import random
import matplotlib.pyplot as plt

map = np.zeros((10, 10))
image = np.zeros((10, 10, 3))
l = list(range(100))

random.seed(1)
# reward 
reward = random.sample(l, k=10)
# block
block = random.sample(l, k=30)

for i in range(len(reward)):
    x = reward[i] // 10
    y = reward[i] % 10
    map[x][y] = random.randint(10, 50)
    # map[x][y] = 1000

for i in range(len(block)):
    x = block[i] // 10
    y = block[i] % 10
    map[x][y] = random.randint(-50, -10)
    # map[x][y] = 1000

reward_max = max(max(row) for row in map)
block_max = min(min(row) for row in map)

map[0][0] = 0
map[5][9] = 10000

# visualization
for i in range(100):
    x = i // 10
    y = i % 10
    if map[x][y] == 0:
        image[x][y][0] = 1.0
        image[x][y][1] = 1.0
        image[x][y][2] = 1.0
    elif map[x][y] == 10000:
        image[x][y][0] = 0
        image[x][y][1] = 0
        image[x][y][2] = 0
    elif map[x][y] > 0:
        image[x][y][0] = 1 - map[x][y] / reward_max
        image[x][y][1] = 1
        image[x][y][2] = 1
    elif map[x][y] < 0:
        image[x][y][0] = 1
        image[x][y][1] = 1 - map[x][y] / block_max
        image[x][y][2] = 1 - map[x][y] / block_max

print(map)
plt.imshow(image)
plt.show()
print(reward)


# print(block)
# print(l)

def compute_reward(route, gamma):
    t = 0
    reward_sum = 0
    for i, j in route:
        if map[i][j] > 0:
            map_val = map[i][j] * gamma ** t
        else:
            map_val = map[i][j]

        t += 1
        reward_sum += map_val
    return reward_sum


# human route demo
best_route_person = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                     (0, 7), (0, 8), (1, 8), (2, 8), (3, 8), (3, 8), (4, 9), (5, 9)]
# get the reward of the route
print(compute_reward(best_route_person, gamma=0.95))
# visualize the route


plt.imshow(image)
# x1 = np.array([0, 1, 2, 3])
# y1 = np.array([0, 0, 0, 0])
# 在网格上可视化路径
# plt.imshow(image)

x1 = np.array([coord[1] for coord in best_route_person])  # 从路径中提取 x 坐标
y1 = np.array([coord[0] for coord in best_route_person])  # 从路径中提取 y 坐标

# 在网格上绘制路径
plt.plot(x1, y1, color="r", marker='o')  # 使用标记在路径单元格上绘制
plt.show()


# plt.plot(x1, y1, color="r")
# plt.show()
def calculate_rewards(map_data, goal,t=0):
    rows, cols = map_data.shape
    rewards = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if map_data[i][j] > 0:
                rewards[i][j] = 100 - 10 * (abs(goal[0] - i) + abs(goal[1] - j))+map_data[i][j]*0.95**t
            else:
                rewards[i][j] = -5 * (abs(goal[0] - i) + abs(goal[1] - j))+map_data[i][j]

    return rewards

def q_learning(map_data, start, end, gamma=0.8, alpha=0.1, epsilon=0.1, max_iterations=10000):
    rows, cols = map_data.shape
    Q = np.zeros((rows * cols, 4))
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()  # Keep track of visited states
    current_path = [start]

    # rewards = calculate_rewards(map_data, end)
    t = 0
    while start != end and t < max_iterations:
        t += 1
        current_state = start[0] * cols + start[1]
        visited.add(current_state)

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(Q[current_state])

        new_row = start[0] + actions[action][0]
        new_col = start[1] + actions[action][1]

        if 0 <= new_row < rows and 0 <= new_col < cols:
            new_state = new_row * cols + new_col

            if map_data[new_row][new_col] < 0 or new_state in visited:
                # If the new state is a block or already visited, choose a different action
                continue

            # reward = calculate_rewards(map_data, end, t)[new_row][new_col]
            reward = compute_reward(current_path, 0.95)
            Q[current_state][action] = Q[current_state][action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[current_state][action])

            start = (new_row, new_col)
            current_path.append(start)
        else:
            continue

    return current_path

start_point = (0, 0)
end_point = (5, 9)
q_learning_path = q_learning(map, start_point, end_point)
print("Best Route (Q-learning):", q_learning_path)

# print(q_learning_path)
x2 = np.array([coord[1] for coord in q_learning_path])  # 从路径中提取 x 坐标
y2 = np.array([coord[0] for coord in q_learning_path])  # 从路径中提取 y 坐标

# 在网格上绘制路径
plt.imshow(image)
plt.plot(x2, y2, color="r", marker='o')  # 使用标记在路径单元格上绘制
plt.show()
