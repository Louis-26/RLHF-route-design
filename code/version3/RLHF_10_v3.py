import numpy as np
import random
import matplotlib.pyplot as plt

# initialize the map and image size
n = 10
# map is the environment setting where the agent will walk in, with the size of a square n*n
# image is used to visualize the map, different color denotes different rewards, they can be positive or negative

map = np.zeros((n, n))
image = np.zeros((n, n, 3))
l = list(range(n * n))


# define a function to set a reward value for the whole map
# use gaussian noise to simulate the reward,
# which is quite natural because fewer location will have extreme positive or negative values
def generate_reward(n, seed=0, mean=0, sigma=10):
    # set the random seed to ensure the experiment is repeatable
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=sigma, size=(n, n))


map += generate_reward(n, seed=0, mean=0, sigma=30)
reward_max = map.max()
reward_min = map.min()

# the starting point is (0,0), and the terminal point is (5,9)
map[0][0] = 0
map[5][9] = 10000
print(map)

# visualize the map
# blue color denotes positive reward, the darker the blue color is, the larger the reward is
# red color denotes negative reward, the darker the red color is, the smaller the reward is

for i in range(100):
    # get the x and y coordinate of the node
    x = i // 10
    y = i % 10
    # set the color of the initial node as yellow
    if (x == 0 and y == 0):
        image[x][y][0] = 1
        image[x][y][1] = 1
        image[x][y][2] = 0
    # set the color of the terminal node as black
    elif (x == 5 and y == 9):
        image[x][y][0] = 0
        image[x][y][1] = 0
        image[x][y][2] = 0
    # set the color of the nodes with positive reward
    elif map[x][y] > 0:
        image[x][y][0] = 1 - map[x][y] / reward_max
        image[x][y][1] = 1 - map[x][y] / reward_max
        image[x][y][2] = 1
    # set the color of nodes with negative reward
    elif map[x][y] < 0:
        image[x][y][0] = 1
        image[x][y][1] = 1 - map[x][y] / reward_min
        image[x][y][2] = 1 - map[x][y] / reward_min


# plt.imshow(image)
# plt.show()

# define a function to compute the reward corresponding to a route
# the route is given
def compute_reward(route, gamma=0.95):
    t = 0
    reward_sum = 0
    # (i,j) is the coordinate in the map
    # the corresponding value restored in the map is in map[j][i]
    # for example, the node (5,0) is stored as the first row of the map, and the fifth value of the map
    for i, j in route:
        if map[i][j] > 0:
            map_val = map[i][j] * gamma ** t
        else:
            map_val = map[i][j]
        t += 1
        print(map_val)
        reward_sum += map_val
    return reward_sum


# define a function to draw the route, given all the nodes to pass by
def draw_route(img, node_li, title=None):
    x_li = []
    y_li = []
    for x, y in node_li:
        x_li.append(x)
        y_li.append(y)
    x_coor = np.array(x_li)
    y_coor = np.array(y_li)

    plt.imshow(img)
    plt.plot(y_coor, x_coor, color="r")
    plt.title(title)
    plt.show()


# give the best route designed and initialize it
# best_route = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (5, 1),
#               (6, 1), (6, 2), (6, 3), (7, 3), (7, 4), (7, 5), (8, 5), (9, 5)]
best_route = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5),
              (1, 6), (2, 6), (3, 6), (3, 7), (4, 7), (5, 7), (5, 8), (5, 9)]
# get the reward of the route
print(compute_reward(best_route, gamma=0.95))
# visualize the route
draw_route(image, best_route, "the human designed best route")


# check whether a node is in the map, n is the length of map
def in_range_map(n, x, y):
    if (0 <= x <= n - 1 and 0 <= y <= n - 1):
        return True
    else:
        return False


# define a function to get the next state, given current state and the action
def get_next_state(current_state, action):
    new_state = tuple(sum(x) for x in zip(current_state, action))
    return new_state


# simulate the Q learning algorithm to train a route and evaluate it
# n is the size of the map, gamma is the discount factor, alpha is the learning rate
# epsilon is the percentage of time when the agent will choose the action with largest Q value,
# episode_num is the total time of episodes
# t_init is a parameter to control the training time
def Q_learning_train(n, gamma=0.95, alpha=0.1, epsilon=0.9, episode_num=1000, t_terminal=100):
    # initialize the action list
    action_alternative = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # initialize the Q value table
    Q_table = np.zeros((n, n, len(action_alternative)))
    for episode in range(episode_num):
        print(episode)
        # initialize the state, we always start from (0,0)
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        # in each single episode
        t = 0
        while not (x == 5 and y == 9) and t <= t_terminal:
            t += 1
            # identify the current state
            state = (x, y)
            # select the action
            # with the probability of epsilon to select the best action with highest Q value
            # with the probability of 1-epsilon to select a random action
            rand_num = np.random.random()

            if rand_num < epsilon:
                # print(x, y)
                action_id = np.argmax(Q_table[x, y])
                action = action_alternative[action_id]
                # print(action,"arg")
            else:
                action = random.choice(action_alternative)
                # print(action,"rand")
            # invalid action
            new_x, new_y = get_next_state(state, action)
            if not in_range_map(n, new_x, new_y):
                continue


            # valid action, update the Q table
            else:
                # evaluate the action
                new_state = get_next_state(state, action)
                prev_x, prev_y = x, y
                x, y = new_state
                previous_Q_val = Q_table[prev_x][prev_y][action_alternative.index(action)]
                # compute the reward
                reward = map[x][y]
                # compute temporal difference for that action
                temporal_difference = reward + gamma * np.max(Q_table[x][y]) - previous_Q_val
                # learn from this temporal difference according to its learning rate alpha
                new_Q_val = previous_Q_val + alpha * temporal_difference
                # update the Q table for the previous state and action
                Q_table[prev_x][prev_y][action_alternative.index(action)] = new_Q_val
    return Q_table


def get_route(Q_table, x=0, y=0, t_terminal=100):
    route = []
    action_alternative = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # initialize the starting point, we always start from 5,9

    route.append((x, y))
    t = 0
    while not (x == 5 and y == 9) and t <= t_terminal:
        t += 1
        # find out the best action according to the Q_table
        action = action_alternative[np.argmax(Q_table[x][y])]
        x, y = get_next_state((x, y), action)
        # print(x, y)
        route.append((x, y))
    return route


# train to get the Q table, use default arguments
obtained_Q_table = Q_learning_train(n, episode_num=1000)
print(obtained_Q_table)
# utilize the trained Q table to get the route
obtained_route = get_route(obtained_Q_table)
print(obtained_route)
draw_route(image, obtained_route, "route without human feedback")

# without human feedback, it fails, because the agent will be limited to the high reward if it simply keeps walking
# between (0,3), and (0,4)
# It is hard to overcome that and explores the much larger reward on (5,9)
# (This is due to the limitation of common RL)


# apply human feedback to enhance the training performance
## give a better route which is similar to the trained route
# when the agent was stuck between (0,3) and (0,4) for more than 10 times,
# I keep all of its trained route before that and keep (0,3), (0,4) only once,
# and point out that you should go down once, and go right twice
partial_route_trained = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
guided_route = [(1, 4), (1, 5), (1, 6)]
# then the initial state for it to be trained will be at (1,6)

route_trained_2 = get_route(obtained_Q_table, x=1, y=6)
print(route_trained_2)
draw_route(image, route_trained_2, "route with human feedback")

# print(compute_reward(route_trained_2,0.95))
