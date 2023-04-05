import cvxpy as cp
import numpy as np


def opt_fair_ep_mdp_lp(P_a: np.ndarray,
                       P_b: np.ndarray,
                       R: np.ndarray,
                       mu_a: np.ndarray,
                       mu_b: np.ndarray,
                       H: int,
                       eps: float,
                       ):
    """
    Calculates the optimal non-stationary policy for an MDP!

    :param P_a: transition matrix for subgroup A, shape [|S|,|A|,|S|]
    :param P_b: transition matrix for subgroup B, shape [|S|,|A|,|S|]
    :param R: reward function of shape [|S|,|A|]
    :param mu_a: inital state distribution for subgroup A, shape [|S|]
    :param mu_b: inital state distribution for subgroup B, shape [|S|]
    :param H: the horizon or length of the episode
    :param eps: the epsilon parameter for the demographic fairness

    Returns the optimal policy as well as its corresponding return
    """
    nS, nA = R.shape

    # placeholder for final solution
    pi_opt_a = np.zeros((H, nS, nA))
    d_opt_a = np.zeros((H, nS, nA))
    J_pi_opt_a = 0

    pi_opt_b = np.zeros((H, nS, nA))
    d_opt_b = np.zeros((H, nS, nA))
    J_pi_opt_b = 0

    # create the optimization variable (occupancy measure)
    d_a = {}
    d_b = {}
    for h in range(H):
        d_a[h] = cp.Variable(shape=(nS, nA), nonneg=True)
        d_b[h] = cp.Variable(shape=(nS, nA), nonneg=True)

    # define the objective
    ret_a = 0
    ret_b = 0
    for h in range(H):
        ret_a += cp.sum(cp.multiply(d_a[h], R))
        ret_b += cp.sum(cp.multiply(d_b[h], R))

    # maximizing the returns is the objective
    obj = cp.Maximize(ret_a + ret_b)

    # placeholder for constraints
    constr = []

    # add the Markov flow constraints
    for h in range(H):
        if h == 0:
            # only for the initial distribution
            constr += [cp.sum(d_a[h][s]) == mu_a[s] for s in range(nS)]
            constr += [cp.sum(d_b[h][s]) == mu_b[s] for s in range(nS)]
        else:
            constr += [cp.sum(d_a[h][s]) == cp.sum(cp.multiply(P_a[:, :, s], d_a[h - 1])) for s in range(nS)]
            constr += [cp.sum(d_b[h][s]) == cp.sum(cp.multiply(P_b[:, :, s], d_b[h - 1])) for s in range(nS)]

    # add the fairness constraint
    constr += [cp.abs(ret_a - ret_b) <= eps]

    # solve the LP
    prob = cp.Problem(obj, constr)
    # prob.solve(verbose=True)
    prob.solve()

    # Calculate the optimal policy
    for h in range(H):
        d_h_a = d_a[h].value
        d_h_b = d_b[h].value
        # calculate policy
        pi_opt_a[h] = d_h_a / d_h_a.sum(axis=1)[:, None]
        pi_opt_b[h] = d_h_b / d_h_b.sum(axis=1)[:, None]
        # update the placeholders
        d_opt_a[h] = d_h_a
        d_opt_b[h] = d_h_b
        # add to the final return
        J_pi_opt_a += np.sum(np.multiply(d_h_a, R))
        J_pi_opt_b += np.sum(np.multiply(d_h_b, R))

    # return the optimal policy and its return
    return pi_opt_a, pi_opt_b, J_pi_opt_a, J_pi_opt_b

