import cvxpy as cp
import numpy as np


def opu_fair_lp_solver(P_a: np.ndarray,
                       P_b: np.ndarray,
                       r_main: np.ndarray,
                       r_opt: np.ndarray,
                       r_pess: np.ndarray,
                       mu_a: np.ndarray,
                       mu_b: np.ndarray,
                       H: int,
                       eps: float,
                       ):
    """
    Calculates the optimal non-stationary policy for an MDP!

    :param P_a: transition matrix for subgroup A, shape [|S|,|A|,|S|]
    :param P_b: transition matrix for subgroup B, shape [|S|,|A|,|S|]

    Note: R (r_main, r_opt, r_pess) are now of shape [|Z|,|S|,|A|]

    :param mu_a: inital state distribution for subgroup A, shape [|S|]
    :param mu_b: inital state distribution for subgroup B, shape [|S|]
    :param H: the horizon or length of the episode
    :param eps: the epsilon parameter for the demographic fairness

    Returns the optimal policy as well as its corresponding return
    """
    n_groups, nS, nA = r_main.shape

    idx_A = 0
    idx_B = 1

    # placeholder for final solution
    pi_opt_a = np.zeros((H, nS, nA))
    d_opt_a = np.zeros((H, nS, nA))

    pi_opt_b = np.zeros((H, nS, nA))
    d_opt_b = np.zeros((H, nS, nA))

    # create the optimization variable (occupancy measure)
    d_a = {}
    d_b = {}
    for h in range(H):
        d_a[h] = cp.Variable(shape=(nS, nA), nonneg=True)
        d_b[h] = cp.Variable(shape=(nS, nA), nonneg=True)

    # define the objective
    ret_main_A = 0.
    ret_main_B = 0.
    for h in range(H):
        ret_main_A += cp.sum(cp.multiply(d_a[h], r_main[idx_A]))
        ret_main_B += cp.sum(cp.multiply(d_b[h], r_main[idx_B]))

    # maximizing the returns is the objective
    obj = cp.Maximize(ret_main_A + ret_main_B)

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

    # define the returns wrt opt and pess rewards
    ret_opt_A = 0.
    ret_opt_B = 0.
    ret_pess_A = 0.
    ret_pess_B = 0.

    for h in range(H):
        # optimistic returns
        ret_opt_A += cp.sum(cp.multiply(d_a[h], r_opt[idx_A]))
        ret_opt_B += cp.sum(cp.multiply(d_b[h], r_opt[idx_B]))
        # pessimistic returns
        ret_pess_A += cp.sum(cp.multiply(d_a[h], r_pess[idx_A]))
        ret_pess_B += cp.sum(cp.multiply(d_b[h], r_pess[idx_B]))

    # add the fairness constraint
    constr += [(ret_opt_A - ret_pess_B) <= eps]
    constr += [(ret_opt_B - ret_pess_A) <= eps]

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

    # return the optimal policy and its return
    return pi_opt_a, pi_opt_b

