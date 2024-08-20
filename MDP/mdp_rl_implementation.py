from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def find_max_action(mdp: MDP, i, j, U_local):
    max_sum = float('-inf')
    for action, i in enumerate(mdp.actions):
        sum_action = 0
        for action2, j in enumerate(mdp.actions):
            next_state = mdp.step((i, j), action2)
            sum_action += mdp.transition_function[i][j] * U_local[next_state[0]][next_state[i]]
        max_sum = max(sum_action, sum_action)
    return max_sum


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:  # TODO wall
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    U_local = U_init
    U_final = None
    while True:
        delta = 0
        U_final = U_local
        for i in range(3):
            for j in range(4):
                U_local[i][j] = mdp.board[i][j] + mdp.gamma * find_max_action(mdp, i, j, U_final)
                if abs(U_local[i][j] - U_final[i][j]) > delta:
                    delta = abs(U_local[i][j] - U_final[i][j])
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    return U_final


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy = None
    for i in range(3):
        for j in range(4):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                policy[i][j] = None
            else:
                max_action = None
                max_utility = float('-inf')
                for action in mdp.actions:
                    next_state = mdp.step((i, j), action)
                    if max_utility > U[next_state[0], next_state[1]]:
                        max_utility = U[next_state[0], next_state[1]]
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
