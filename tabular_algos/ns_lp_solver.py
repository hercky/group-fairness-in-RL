import numpy as np
import cvxpy as cp


def opt_ns_mdp_lp(P: np.ndarray,
                  R: np.ndarray,
                  mu: np.ndarray,
                  H: int,
                  ):
    """
    Calculates the optimal non-stationary policy for an MDP!

    :param P: transition matrix of shape [|S|,|A|,|S|]
    :param R: reward function of shape [|S|,|A|]
    :param mu: inital state distribution of shape [|S|]
    :param H: the horizon or length of the episode

    Returns the optimal policy as well as its corresponding return
    """
    nS, nA = R.shape

    # placeholder for final solution
    pi_opt = np.zeros((H, nS, nA))
    d_opt = np.zeros((H, nS, nA))
    J_pi_opt = 0

    # create the optimization variable (occupancy measure)
    d = {}
    for h in range(H):
        d[h] = cp.Variable(shape=(nS, nA), nonneg=True)

    # define the objective
    ret = 0
    for h in range(H):
        ret += cp.sum(cp.multiply(d[h], R))
    # maximizing the returns is the objective
    obj = cp.Maximize(ret)

    # placeholder for constraints
    constr = []

    # add the Markov flow constraints
    for h in range(H):
        if h == 0:
            # only for the initial distribution
            constr += [cp.sum(d[h][s]) == mu[s] for s in range(nS)]
        else:
            constr += [cp.sum(d[h][s]) == cp.sum(cp.multiply(P[:, :, s], d[h - 1])) for s in range(nS)]

    # solve the LP
    prob = cp.Problem(obj, constr)
    # prob.solve(verbose=True)
    prob.solve()

    # Calculate the optimal policy
    for h in range(H):
        d_h = d[h].value
        # calculate policy
        pi_opt[h] = d_h / d_h.sum(axis=1)[:, None]
        # update the placeholders
        d_opt[h] = d_h
        # add to the final return
        J_pi_opt += np.sum(np.multiply(d_h, R))

    # return the optimal policy and its return
    return pi_opt, J_pi_opt