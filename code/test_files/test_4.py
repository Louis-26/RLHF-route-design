a = True
b = True
print(a + b)
my_route = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3),
            (0, 4), (0, 3), (0, 4), (0, 3), (0, 4), (0, 3)]

my_route_2 = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3),
              (0, 4), (0, 5), (0, 4), (0, 3), (0, 4), (0, 5),
              (0, 4), (0, 3), (0, 4), (0, 5)]

best_route = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5),
              (1, 6), (2, 6), (3, 6), (3, 7), (4, 7), (5, 7), (5, 8), (5, 9)]


def find_stuck_point(route):
    result = []
    count_dict = {}
    for node in route:
        count_dict[node] = count_dict.get(node, 0) + 1
    for key, val in count_dict.items():
        if val >= 3:
            result.append(key)
    # if there is no stuck point, return None
    if len(result) == 0:
        result = None
    # determine the stuck node order
    else:
        temp = []
        for node in result:
            order_in_route = route.index(node)
            temp.append((order_in_route, node))
        temp=sorted(temp)
        result=[i[1] for i in temp]
    return result


print(find_stuck_point(my_route))
print(find_stuck_point(my_route_2))
print(find_stuck_point(best_route))

print(my_route.index((0,4)))
my_test_1=[(1,(3,4)),(4,(2,1)),(3,(2,7))]
print(sorted(my_test_1))

# the potential action can be go up/down/left/right one step each time
action_alternative = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# define a function to get the next state, given current state and the action
def get_next_state(current_state, action):
    new_state = tuple(sum(x) for x in zip(current_state, action))
    return new_state

def route_after_human_feedback(n, trained_route):
    modified_route = []
    # find the stuck point
    stuck_point_li = find_stuck_point(trained_route)
    terminal_node_training = stuck_point_li[-1]
    # get the first part of route
    partial_route_1 = []
    for node in trained_route:
        partial_route_1.append(node)
        if node == terminal_node_training:
            break

    # apply human feedback here
    print(f"Now the agent is stuck between several nodes. Its terminal node now is {terminal_node_training}."
          "Please give feedback to help it out.")
    guide_route = []
    curr_state = terminal_node_training
    while True:
        human_action = input("Please enter one action(0: go up one step, 1: go down one step, "
                             "2: go left one step, 3: go right one step):")
        if human_action == "quit":
            break
        elif human_action in ["1", "2", "3", "4"]:
            action_choice = int(human_action)
            action = action_alternative[action_choice]
            curr_state = get_next_state(curr_state, action)
            guide_route.append(curr_state)
        else:
            print("Invalid input! Please enter again")
            continue
    print(f"With the guidance of human feedback, the agent transmit to state{curr_state}.")
    print(guide_route)
    return modified_route

n=10
# route_after_human_feedback(n,my_route)
s1=[1,3,4]
s2=[5,7,8]
s3=[10,11]
