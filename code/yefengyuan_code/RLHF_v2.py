import numpy as np
import random
import matplotlib.pyplot as plt

map = np.zeros((10,10))
image = np.zeros((10,10,3))
l = []
# 创建map
for i in range(100):
    l.append(i)

# reward 
reward = random.sample(l, k=10)
# block
block = random.sample(l, k=30)

for i in range(len(reward)):
    x = reward[i]//10
    y = reward[i]%10
    map[x][y] = random.randint(10, 50)
    # map[x][y] = 1000

for i in range(len(block)):
    x = block[i]//10
    y = block[i]%10
    map[x][y] = random.randint(-50, -10)
    # map[x][y] = 1000

reward_max = max(max(row) for row in map)
block_max = min(min(row) for row in map)

map[0][0] = 0
map[5][9] = 10000

# 可视化map

for i in range(100):
    x = i//10
    y = i%10
    if map[x][y] == 0:
        image[x][y][0] = 1.0
        image[x][y][1] = 1.0
        image[x][y][2] = 1.0
    elif map[x][y] == 10000:
        image[x][y][0] = 0
        image[x][y][1] = 0
        image[x][y][2] = 0
    elif map[x][y] > 0:
        image[x][y][0] = 1-map[x][y]/reward_max
        image[x][y][1] = 1
        image[x][y][2] = 1
    elif map[x][y] < 0:
        image[x][y][0] = 1
        image[x][y][1] = 1-map[x][y]/block_max
        image[x][y][2] = 1-map[x][y]/block_max

print(map)
plt.imshow(image)
plt.show()
# print(reward)
# print(block)
# print(l)