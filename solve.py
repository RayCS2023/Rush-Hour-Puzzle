from board import *
import copy
import heapq as hq
import matplotlib.pyplot as plt


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    depth = 0
    frontier = []
    explored = []
    init_state = State(init_board, hfn, hfn(init_board), depth)

    hq.heappush(frontier, ((init_state.f, init_state.id, 0),init_state))

    while len(frontier) != 0:
        state = hq.heappop(frontier)[1]

        if not any(s.id == state.id for s in explored):
            explored.append(state)
            if is_goal(state):
                path = get_path(state)
                return path, len(path)-1

            neighbours = get_successors(state)

            for neighbour in neighbours:
                hq.heappush(frontier, ((neighbour.f, neighbour.id, neighbour.parent.id), neighbour))
            # frontier.extend(neighbours)
            # frontier = sorted(frontier, key = lambda n: (n.f, n.id, n.parent.id), reverse=True)
    
    return [], -1

def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    depth = 0
    frontier = [State(init_board, zero_heuristic, 0, depth)]
    explored = []

    while len(frontier) != 0:
        state = frontier.pop()

        if not any(s.id == state.id for s in explored):
            explored.append(state)
            if is_goal(state):
                path = get_path(state)
                return path, len(path)-1

            neighbours = get_successors(state)
            neighbours.sort(key=lambda n: n.id, reverse=True)
            frontier.extend(neighbours)
    
    return [], -1


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    neighbours = []
    cars = state.board.cars
    for i in range(len(cars)):
        if cars[i].orientation == "h":
            addHorizontalNeighbours(state, i, neighbours)
        else:
            addVerticalNeighbours(state, i, neighbours)

    return neighbours

def addVerticalNeighbours(state: State, i: int, neighbours: list):
    board = state.board
    car = board.cars[i]
    varCoord = car.var_coord

    # moving up
    for yCoord in range(varCoord-1,-1,-1):
        if board.grid[yCoord][car.fix_coord] == ".":
            carsCopy = copy.deepcopy(board.cars)
            carsCopy[i].set_coord(yCoord)
            addNeighbour(state, carsCopy, neighbours)
        else:
            break

    # moving down
    varCoord = car.var_coord + car.length
    for yCoord in range(varCoord, board.size):
        if board.grid[yCoord][car.fix_coord] == ".":
            carsCopy = copy.deepcopy(board.cars)
            carsCopy[i].set_coord(yCoord - car.length + 1)
            addNeighbour(state, carsCopy, neighbours)
        else:
            break

def addHorizontalNeighbours(state: State, i: int, neighbours: list):
    board = state.board
    car = board.cars[i]
    varCoord = car.var_coord

    # moving left
    for xCoord in range(varCoord-1,-1,-1):
        if board.grid[car.fix_coord][xCoord] == ".":
            carsCopy = copy.deepcopy(board.cars)
            carsCopy[i].set_coord(xCoord)
            addNeighbour(state, carsCopy, neighbours)
        else:
            break

    # moving right
    varCoord = car.var_coord + car.length
    for xCoord in range(varCoord, board.size):
        if board.grid[car.fix_coord][xCoord] == ".":
            carsCopy = copy.deepcopy(board.cars)
            carsCopy[i].set_coord(xCoord - car.length + 1)
            addNeighbour(state, carsCopy, neighbours)
        else:
            break

def addNeighbour(state: State, cars: list, neighbours: list):
    board = state.board
    newBoard = Board(board.name, board.size, cars)
    newState = State(newBoard, state.hfn, 0, state.depth+1, state)
    newState.f = newState.hfn(newBoard) + newState.depth
    neighbours.append(newState)

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    return state.board.grid[2][4] == "<" and state.board.grid[2][5] == ">"


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    path = []
    s = state

    while s is not None:
        path.append(s)
        s = s.parent

    path.reverse()
    return path


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    
    # check if we are in goal state
    if board.grid[2][4] == "<" and board.grid[2][5] == ">":
        return 0

    exitRow = board.grid[2]
    count=1
    for i in range(exitRow.index(">")+1, board.size):
        if (exitRow[i] != "."):
            count+=1

    return count


def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    num_cars_blocking = blocking_heuristic(board)

    for i in range(board.grid[2].index(">")+1, board.size):
        num_cars_blocking_up = len(board.cars) + 1
        num_cars_blocking_down = len(board.cars) + 1
        if (board.grid[2][i] != "."):
            if (board.grid[2][i] != "|" and (board.grid[2][i] != "v" or board.grid[1][i] != "|") and (board.grid[2][i] != "^" or board.grid[3][i] != "|")):
                num_top_cars = 0
                if (board.grid[1][i] == "v" or board.grid[1][i] == ">" or board.grid[1][i] == "<" or board.grid[1][i] == "-"):
                    num_top_cars += 1

                if (board.grid[0][i] == "v" or board.grid[0][i] == ">" or board.grid[0][i] == "<" or board.grid[0][i] == "-"):
                    num_top_cars += 1

                num_cars_blocking_up = num_top_cars
        
            num_bot_cars = 0

            if (board.grid[3][i] == "^" or board.grid[3][i] == ">" or board.grid[3][i] == "<" or board.grid[3][i] == "-"):
                num_bot_cars += 1

            if (board.grid[4][i] == "^" or board.grid[4][i] == ">" or board.grid[4][i] == "<" or board.grid[4][i] == "-"):
                num_bot_cars += 1

            if (board.grid[5][i] == "^" or board.grid[5][i] == ">" or board.grid[5][i] == "<"or board.grid[5][i] == "-"):
                num_bot_cars += 1
            
            num_cars_blocking_down = num_bot_cars

            num_cars_blocking =  num_cars_blocking + min(num_cars_blocking_up, num_cars_blocking_down)
            break

    return num_cars_blocking

def a_star_node_expanded(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    count = 0
    depth = 0
    frontier = []
    explored = []
    init_state = State(init_board, hfn, hfn(init_board), depth)

    hq.heappush(frontier, ((init_state.f, init_state.id, 0),init_state))

    while len(frontier) != 0:
        state = hq.heappop(frontier)[1]
        count = count + 1

        if not any(s.id == state.id for s in explored):
            explored.append(state)
            if is_goal(state):
                path = get_path(state)
                return path, len(path)-1, count

            neighbours = get_successors(state)

            for neighbour in neighbours:
                hq.heappush(frontier, ((neighbour.f, neighbour.id, neighbour.parent.id), neighbour))
            # frontier.extend(neighbours)
            # frontier = sorted(frontier, key = lambda n: (n.f, n.id, n.parent.id), reverse=True)
    
    return [], -1, -1
        

# boards = from_file("/Users/raymond/Desktop/a1_files/code_posted/jams_posted.txt")
# x = [i for i in range(len(boards))]
# a_star_blocking = [4,3239,1761,3450,464,3635,3909,18490,4474,1808,11793,4665,4550,24403,27020,3148,16385,15291,9781,3352,1461,1352,16704,8773,41348,57850,32183,16721,8194,37189,6947,29482,2676,19678,33584,32665,16427,14152,22236,23999,18267]
# a_star_advanced = [4,1019,755,3351,159,702,2117,17606,3404,1352,10781,4395,4053,20182,20736,3136,15692,12533,7242,3267,654,1133,11997,7244,41158,53904,31402,15285,5758,36560,6338,28633,2630,17030,32053,32502,15717,13147,21195,23893,14429]

# for i in range(len(boards)):
#     p, c, count1 = a_star_node_expanded(boards[i], blocking_heuristic)
#     p, c, count2 = a_star_node_expanded(boards[i], advanced_heuristic)
#     a_star_blocking.append(count1)
#     a_star_advanced.append(count2)
#     print(count1, count2)

# plt.title("Number of Expanded Nodes on Test Set with 2 Different Heuristics")
# plt.xlabel("Puzzle Number: Jam-(#)")
# plt.ylabel("Nodes Expanded")
# plt.scatter(x, a_star_blocking)
# plt.scatter(x, a_star_advanced)
# plt.legend(('Blocking Heuristic', 'Advance Heuristic'),
#            scatterpoints=1,
#            loc='lower right',
#            ncol=1,
#            fontsize=8)
# plt.show()

# for i in range(len(boards)):
#     print("Nodes expanded with advance for Jam-" + str(i) +": " + str(a_star_advanced[i]))




