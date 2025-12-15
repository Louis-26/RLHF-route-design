import numpy as np
import random
m=(1,2)
n=(4,2)
print(tuple(sum(x) for x in zip(m,n)))

def in_range_map(n, x, y):
    if (0 <= x <= n - 1 and 0 <= y <= n - 1):
        return True
    else:
        return False

l=(4,2)
print(in_range_map(10, *l))

n=10
#
# Q_table = np.zeros((n, n, 4))
# print(Q_table[*l])
# print(*l)

action_alternative = [(-1, 0), (1, 0), (0, -1), (0, 1)]
print(random.choice(action_alternative))
Q_table=np.zeros((n,n,len(action_alternative)))
print(Q_table)
print(Q_table[0][0])
print(Q_table.shape)

# define a function to get the next state, given current state and the action
def get_next_state(current_state,action):
    new_state = tuple(sum(x) for x in zip(current_state, action))
    return new_state

c1=(4,5)
c2=(-1,0)
print(get_next_state(c1,c2))
act=(1,0)
print(action_alternative.index(act))