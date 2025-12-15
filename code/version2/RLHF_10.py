import numpy as np
import random
import matplotlib.pyplot as plt

map = np.zeros((10, 10))
image = np.zeros((10, 10, 3))
l = list(range(100))

random.seed(0)
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

# define a function to draw the route, given all the nodes to pass by
def draw_route(img,node_li):
    x_li=[]
    y_li=[]
    for x,y in node_li:
        x_li.append(x)
        y_li.append(y)
    x_coor=np.array(x_li)
    y_coor=np.array(y_li)
    plt.imshow(img)
    plt.plot(x_coor,y_coor,color="r")
    plt.show()

# human route demo
best_route_person = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
                     (7, 0), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (9, 4), (9, 5)]

draw_route(image, best_route_person)
# get the reward of the route
print(compute_reward(best_route_person, gamma=0.95))
# visualize the route

# plt.imshow(image)
# x1 = np.array([0, 1, 2, 3])
# y1 = np.array([0, 0, 0, 0])
#
# plt.plot(x1, y1, color="r")
# plt.show()
