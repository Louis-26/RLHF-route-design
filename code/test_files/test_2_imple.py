import numpy as np
env_row=11
env_col=11
# init Q value
q_val=np.zeros((env_row,env_col,4))
print(q_val)

# define actions
action=[(0,1),(1,0),(0,-1),(0,-1)]

# define rewards
rewards=np.full((env_row,env_col),-100)
rewards[0,5]=100

# define aisle locations
aisles={}
aisles[1]=list(range(1,10))
aisles[2]=[1,7,9]
aisles[3]=list(range(1,8))
aisles[3].append(9)
aisles[4]=[3,7]
aisles[5]=list(range(11))
aisles[6]=[5]
aisles[7]=list(range(1,10))
aisles[8]=[3,7]
aisles[9]=list(range(11))

for row_id in range(1,10):
    for col_id in aisles[row_id]:
        rewards[row_id,col_id]=-1

print(rewards)
