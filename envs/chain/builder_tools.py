"""
Code inspired from https://github.com/iosband/TabulaRL/blob/master/src/environment.py
"""

import numpy as np
from envs.chain.river_swim import RiverSwim


def make_riverSwim(ep_len: int = 10,
                   num_states: int = 7,
                   spawn_prob: float = 0.95,
                   p_fail: float = 0.05,
                   p_stay_a: float = 0.6,
                   p_stay_b: float = 0.05,
                   seed: int = 0):
    """
    Makes the benchmark RiverSwim MDP and returns it.

    :param ep_len:
    :param num_states:
    :param spawn_prob:
    :param p_fail:
    :param p_stay_a:
    :param p_stay_b:
    :param seed:
    :return:
    """
    num_actions = 2
    left_action = 0
    right_action = 1

    # hard code the MDP's interesting state
    start_state_a = 0
    start_state_b = 1
    intermed_state = num_states//2

    # -------------------------------------------------
    #       Define the mu_z
    # -------------------------------------------------
    mu_a = np.ones(num_states) * (1. - spawn_prob) / (num_states - 1)
    mu_a[start_state_a] = spawn_prob

    mu_b = np.ones(num_states) * (1. - spawn_prob) / (num_states - 1)
    mu_b[start_state_b] = spawn_prob

    mu_z = np.vstack([mu_a, mu_b])

    # -------------------------------------------------
    #       Define the R
    # -------------------------------------------------
    R = np.zeros((num_states, num_actions))

    # reward for staying in the start state
    R[0, left_action] = 0.01
    # reward for staying in the goal state
    R[num_states-1, right_action] = 1.

    # reward for reaching middle state
    R[intermed_state-1, right_action] = 0.1
    R[intermed_state+1, left_action] = 0.1

    # -------------------------------------------------
    #       Define the P_a
    # -------------------------------------------------
    P_a = np.zeros((num_states, num_actions, num_states))
    P_b = np.zeros((num_states, num_actions, num_states))

    # get the prob of transitions for right
    # for A
    p_fail_a = p_fail
    p_success_a = 1. - (p_stay_a + p_fail_a)
    # for B
    p_fail_b = p_fail
    p_success_b = 1. - (p_stay_b + p_fail_b)
    assert p_success_a >= 0 and p_success_b >= 0, "Probabilities don't sum to 1"
    assert (p_success_a + p_stay_a + p_fail_a == 1.0) and \
            (p_success_b + p_stay_b + p_fail_b == 1.0), "Probabilities don't sum to 1"

    # left action deterministic
    for s in range(num_states):
        P_a[s, left_action, max(0, s - 1)] = 1.0
        P_b[s, left_action, max(0, s - 1)] = 1.0

    # right action stochastic
    for s in range(1, num_states-1):
        # for A
        P_a[s, right_action, min(num_states - 1, s + 1)] = p_success_a # successfully going right
        P_a[s, right_action, s] = p_stay_a #staying in one place
        P_a[s, right_action, max(0, s-1)] = p_fail_a # going to left state instead
        # for B
        P_b[s, right_action, min(num_states - 1, s + 1)] = p_success_b  # successfully going right
        P_b[s, right_action, s] = p_stay_b  # staying in one place
        P_b[s, right_action, max(0, s - 1)] = p_fail_b  # going to left state instead

    # for first and last state
    # for A
    P_a[0, right_action, 0] = 0.4 # stay
    P_a[0, right_action, 1] = 0.6
    P_a[num_states-1, right_action, num_states-1] = 0.6 # stay
    P_a[num_states - 1, right_action, num_states - 2] = 0.4
    # for B
    P_b[0, right_action, 0] = 0.4
    P_b[0, right_action, 1] = 0.6
    P_b[num_states - 1, right_action, num_states - 1] = 0.6
    P_b[num_states - 1, right_action, num_states - 2] = 0.4

    # -------------------------------------------------
    #       Create the environment and return it
    # -------------------------------------------------

    return RiverSwim(P_a=P_a, P_b=P_b, R=R, mu_a=mu_a, mu_b=mu_b, ep_len=ep_len, seed=seed)

