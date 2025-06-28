from random import choice, randint

four_directional_vector = [(1, 0), (0, -1), (-1, 0), (0, 1)]
start_cor = (0, 0)
grid_size = 4
pits = 2
goal = 1
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}


def set_random_pits(num_pits):
    """
    Description:
        Randomly generates coordinates for pits in the grid world.
    Parameters:
        num_pits (int): Number of pits to place in the grid.
    Returns:
        list[tuple]: List of pit coordinates.
    """
    pits_cors = []
    while True:
        pit_cor = (randint(0, grid_size - 1), randint(0, grid_size - 1))
        if pit_cor not in pits_cors and pit_cor != start_cor:
            pits_cors.append(pit_cor)
        if len(pits_cors) == num_pits:
            return pits_cors


def set_random_goal(pits_cors):
    """
    Description:
        Randomly generates a goal coordinate in the grid world, ensuring it does not overlap with pits or the start position.
    Parameters:
        pits_cors (list[tuple]): List of pit coordinates.
    Returns:
        tuple: Goal coordinate.
    """
    while True:
        goal_cor = (randint(0, grid_size), randint(0, grid_size - 1))
        if goal_cor not in pits_cors and goal_cor != start_cor:
            return goal_cor


def print_grid(pits_cors: list[tuple], goal_cor: tuple, agent_cor: tuple):
    """
    Description:
        Prints the current state of the grid world, including the agent, goal, and pits.
    Parameters:
        pits_cors (list[tuple]): List of pit coordinates.
        goal_cor (tuple): Goal coordinate.
        agent_cor (tuple): Agent's current position.
    Returns:
        None
    """
    for x_cor in range(grid_size):
        row = ''
        for y_cor in range(grid_size):
            cord = (x_cor, y_cor)
            if cord == agent_cor:
                row += 'A '
            elif cord == goal_cor:
                row += 'G '
            elif cord in pits_cors:
                row += 'P '
            else:
                row += '- '
        print(row)
    print()


def step(pos, action, goal_cor, pits_cors):
    """
    Description:
        Simulates a single step in the grid world based on the agent's action.
    Parameters:
        pos (tuple): Current position of the agent.
        action (str): Action taken by the agent ('up', 'down', 'left', 'right').
        goal_cor (tuple): Goal coordinate.
        pits_cors (list[tuple]): List of pit coordinates.
    Returns:
        tuple:
    """
    # Calculate new position
    move = ACTION_MAP[action]
    new_pos = (pos[0] + move[0], pos[1] + move[1])
    # Stay in bounds
    if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
        pos = new_pos
    # Check for reward or end
    if pos == goal_cor:
        return pos, 1, True
    if pos in pits_cors:
        return pos, -1, True
    return pos, 0, False


def run_episode(random_moves=True):
    """
    Description:
        Simulates an episode in the grid world, where the agent moves until it reaches the goal or falls into a pit.
    Parameters:
        random_moves (bool): If True, the agent moves randomly; otherwise, it follows a rule-based strategy.
    Returns:
        None
    """
    pos = start_cor
    pits_cors = set_random_pits(pits)
    goal_cor = set_random_goal(pits_cors)
    print("Pits:", pits_cors)
    print("Goal:", goal_cor)
    done = False
    total_reward = 0
    print_grid(pits_cors, goal_cor, pos)
    steps = 0

    while not done:
        if random_moves:
            action = choice(ACTIONS)
        else:
            # Rule-based: try right, else down
            if pos[1] < grid_size - 1:
                action = 'right'
            elif pos[0] < grid_size - 1:
                action = 'down'
            else:
                action = choice(ACTIONS)
        pos, reward, done = step(pos, action, goal_cor, pits_cors)
        print(f"Step {steps + 1}: Move {action}, Reward: {reward}")
        print_grid(pits_cors, goal_cor, pos)
        total_reward += reward
        steps += 1

    print(f"Episode ended. Total reward: {total_reward}. Number of steps: {steps}")


run_episode(random_moves=True)
