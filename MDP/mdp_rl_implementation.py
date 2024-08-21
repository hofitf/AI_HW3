from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def find_max_action(mdp: MDP, i, j, U_local):
    max_sum = float('-inf')
    if (i, j) in mdp.terminal_states:
        return 0
    for action in mdp.actions.keys():
        sum_action = 0
        count = 0
        for action2, value2 in mdp.actions.items():
            next_state = mdp.step((i, j), action2)
            y = mdp.transition_function[action][count]
            x = U_local[next_state[0]][next_state[1]]
            sum_action += mdp.transition_function[action][count] * U_local[next_state[0]][next_state[1]]
            count += 1
        max_sum = max(max_sum, sum_action)
    return max_sum


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
                    U_local[i][j] = float(mdp.board[i][j]) + mdp.gamma * find_max_action(mdp, i, j, U_final)
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
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
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
