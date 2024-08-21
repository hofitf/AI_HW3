from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def find_max_action(mdp: MDP, i, j, U_local):
    max_sum = float('-inf')
    max_action = None
    if (i, j) in mdp.terminal_states:
        return 0, None
    for action in mdp.actions.keys():
        sum_action = 0
        count = 0
        for action2 in mdp.actions.keys():
            next_state = mdp.step((i, j), action2)
            sum_action += mdp.transition_function[action][count] * U_local[next_state[0]][next_state[1]]
            count += 1
        if max_sum < sum_action:
            max_sum = sum_action
            max_action = action
    return max_sum, max_action


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    U_local = U_init
    U_final = []
    for i in range(mdp.num_row):
        U_final.append([0] * mdp.num_col)
    while True:
        delta = 0
        for i in range(3):
            for j in range(4):
                U_final[i][j] = U_local[i][j]
        for i in range(3):
            for j in range(4):
                if mdp.board[i][j] != 'WALL':
                    max_sum_action = find_max_action(mdp, i, j, U_final)
                    U_local[i][j] = float(mdp.board[i][j]) + mdp.gamma * max_sum_action[0]
                    if abs(U_local[i][j] - U_final[i][j]) > delta:
                        delta = abs(U_local[i][j] - U_final[i][j])
                else:
                    U_local[i][j] = None
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    return U_final


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy = []
    for i in range(mdp.num_row):
        policy.append([0] * mdp.num_col)
    for i in range(3):
        for j in range(4):
            max_action = None
            if mdp.board[i][j] == 'WALL' or (i, j) in mdp.terminal_states:
                policy[i][j] = None
            else:
                max_utility = float('-inf')
                for action in mdp.actions.keys():
                    next_state = mdp.step((i, j), action)
                    if mdp.board[next_state[0]][next_state[1]] == 'WALL':
                        continue
                    if max_utility < U[next_state[0]][next_state[1]]:
                        max_utility = U[next_state[0]][next_state[1]]
                        max_action = action
                policy[i][j] = max_action
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    P = np.zeros((mdp.num_row * mdp.num_col, mdp.num_row * mdp.num_col))
    R = np.zeros((mdp.num_row * mdp.num_col))
    I = np.eye(mdp.num_row * mdp.num_col)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] != 'WALL' and policy[i][j] is not None:
                count = 0
                for action in mdp.actions.keys():
                    next_state = mdp.step((i, j), action)
                    if mdp.transition_function[policy[i][j]][count] is not None:
                        P[mdp.num_col * i + j][mdp.num_col * next_state[0] + next_state[1]] += \
                            mdp.transition_function[policy[i][j]][count]
                    count += 1
            if mdp.board[i][j] != 'WALL':
                R[mdp.num_col * i + j] = float(mdp.board[i][j])

    V = np.linalg.solve(I - mdp.gamma * P, R)
    U_final = []
    for i in range(mdp.num_row):
        U_final.append([0] * mdp.num_col)

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] != 'WALL':
                U_final[i][j] = V[mdp.num_col * i + j]
            else:
                U_final[i][j] = None

    return U_final


def turn_to_action(action):
    if action == 'UP':
        return Action.UP
    elif action == 'DOWN':
        return Action.DOWN
    elif action == 'RIGHT':
        return Action.RIGHT
    elif action == 'LEFT':
        return Action.LEFT
    return None


def turn_action_to_string(action):
    if action == Action.UP:
        return 'UP'
    elif action == Action.DOWN:
        return 'DOWN'
    elif action == Action.RIGHT:
        return 'RIGHT'
    elif action == Action.LEFT:
        return 'LEFT'
    return None


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = []
    for i in range(mdp.num_row):
        optimal_policy.append([None] * mdp.num_col)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            optimal_policy[i][j] = turn_to_action(policy_init[i][j])
    changed = False
    while not changed:
        U = policy_evaluation(mdp, optimal_policy)
        changed = True
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                sum_action = 0
                count = 0
                if mdp.board[i][j] != 'WALL' and (i, j) not in mdp.terminal_states:
                    for action in mdp.actions.keys():
                        next_state = mdp.step((i, j), action)
                        if optimal_policy[i][j] is not None:
                            sum_action += mdp.transition_function[optimal_policy[i][j]][count] * U[next_state[0]][
                                next_state[1]]
                        count += 1
                    max_sum_action = find_max_action(mdp, i, j, U)
                    if max_sum_action[0] > sum_action:
                        optimal_policy[i][j] = max_sum_action[1]
                        changed = False
        if changed:
            break
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            optimal_policy[i][j] = turn_action_to_string(optimal_policy[i][j])
    return optimal_policy


def adp_algorithm(
        sim: Simulator,
        num_episodes: int,
        num_rows: int = 3,
        num_cols: int = 4,
        actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return reward_matrix, transition_probs
