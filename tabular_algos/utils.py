import numpy as np


def evaluate_ns_ep_policy(P: np.ndarray,
                          R: np.ndarray,
                          mu: np.ndarray,
                          H: int,
                          pi: np.ndarray,
                          ):
    """
    Calculate the return for a non-stationary policy for the episodic MDP.
    Works also for stationary policies by treating them as non-stationary.

    :param P: transition matrix of shape [|S|,|A|,|S|]
    :param R: reward function of shape [|S|,|A|]
    :param mu: inital state distribution of shape [|S|]
    :param H: the horizon or length of the episode
    :param pi: the policy to evaluate [|S|, |A|]

    :return: expected return (float)
    """
    n_states, n_actions = R.shape

    # occupancy measure
    d = np.zeros((H, n_states, n_actions))

    assert len(pi.shape) == 2 or len(pi.shape) == 3 , "wrong shape of pi"

    # calculate the occupancy measures
    for h in range(H):
        if h == 0:
            for s in range(n_states):
                for a in range(n_actions):
                    if len(pi.shape) == 2:
                        d[h, s, a] = mu[s] * pi[s, a]
                    else:
                        d[h, s, a] = mu[s] * pi[h, s, a]
        else:
            for s in range(n_states):
                for a in range(n_actions):
                    if len(pi.shape) == 2:
                        d[h, s, a] = pi[s, a] * np.sum(np.multiply(P[:, :, s], d[h - 1]))
                    else:
                        d[h, s, a] = pi[h, s, a] * np.sum(np.multiply(P[:, :, s], d[h - 1]))

    # calculate the returns
    J_pi = 0

    for h in range(H):
        J_pi += np.sum(np.multiply(d[h], R))

    return J_pi


def backward_policy_eval(pi: np.ndarray,
                         P: np.ndarray,
                         R: np.ndarray,
                         mu: np.ndarray,
                         H: int):
    """
    Do policy evaluation via backward induction

    :param pi:
    :param P:
    :param R:
    :param mu:
    :param H:
    :return:
    """
    n_states, n_actions = R.shape

    V_pi = np.zeros((H, n_states))

    if len(pi.shape) == 3:
        # non-stationary, depends on H
        assert pi.shape[0] == H

        for h in range(H - 1, -1, -1):
            pi_h = pi[h]
            P_pi_h = np.einsum('sat,sa -> st', P, pi_h)
            R_pi_h = np.einsum('sa,sa -> s', R, pi_h)

            if h == H - 1:
                # last/terminal state
                V_pi[h] = R_pi_h
            else:
                V_pi[h] = R_pi_h + np.einsum('st, t -> s', P_pi_h, V_pi[h + 1])

    elif len(pi.shape) == 2:
        P_pi = np.einsum('sat,sa -> st', P, pi)
        R_pi = np.einsum('sa,sa -> s', R, pi)

        for h in range(H - 1, -1, -1):
            if h == H - 1:
                # last/terminal state
                V_pi[h] = R_pi
            else:
                V_pi[h] = R_pi + np.einsum('st, t -> s', P_pi, V_pi[h + 1])

    else:
        raise Exception('wrong pi shape')

    # return the value of this policy for this subgroup
    ret = np.sum(np.multiply(mu, V_pi[0]))
    return ret
