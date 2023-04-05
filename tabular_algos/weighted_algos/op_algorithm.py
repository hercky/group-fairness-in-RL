"""
A baseline that uses MLE estimates with the LP solver
"""
import cvxpy as cp
import numpy as np

from tabular_algos.weighted_algos.base_algo import BaseTabularAlgo
from tabular_algos.utils import evaluate_ns_ep_policy


def opu_fair_lp_solver(P_a: np.ndarray,
                       P_b: np.ndarray,
                       r_main: np.ndarray,
                       r_opt: np.ndarray,
                       r_pess: np.ndarray,
                       mu_a: np.ndarray,
                       mu_b: np.ndarray,
                       H: int,
                       eps: float,
                       prob_a: float,
                       prob_b: float,
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
    obj = cp.Maximize(prob_a * ret_main_A + prob_b * ret_main_B)

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
    constr += [(prob_a * ret_opt_A - prob_b * ret_pess_B) <= eps]
    constr += [(prob_b * ret_opt_B - prob_a * ret_pess_A) <= eps]

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



class OptPess_Fair_Algo(BaseTabularAlgo):
    """

    """
    def __init__(self,
                 n_groups: int,
                 n_states: int,
                 n_actions: int,
                 pi_init: np.ndarray, # shape ZxSxA or
                 delta=0.1,
                 beta_scale_factor=1.,
                 seed=0):
        """

        """
        super().__init__(n_groups, n_states, n_actions, seed)
        self.delta = delta
        self.beta_scale_factor = beta_scale_factor

        assert pi_init.shape[0] == self.n_groups, "wrong shape of the initial policy"
        assert len(pi_init.shape) == 3 or len(pi_init.shape) == 4, "wrong shape of the initial policy"
        self.pi_init = pi_init

    def estimate_beta(self, H:int, K:int):
        """
        Compute the beta required for reward shaping here based on counts
        :return:
        """
        # caluclate the factor inside the beta term
        C = np.log((4. * self.n_groups**2 * self.n_states**2 * self.n_actions * H * K)/self.delta)

        # beta placeholder
        beta = np.zeros((self.n_groups, self.n_states, self.n_actions))

        # estimate beta for both subgroups
        for z in range(self.n_groups):
            # calculate N(z, s,a) = \max{N(z,s,a), 1}
            normalized_count = np.maximum(np.sum(self.count_P[z], 2), 1.)
            # has shape (S x A)
            beta[z] = np.sqrt(1./normalized_count * C)

        # Scaling trick typically used in optimism based algorithms
        # Note (!): should this be uniform or non-uniform
        beta = self.beta_scale_factor * beta

        return beta

    def compute_best_policy(self,
                            est_P: np.ndarray,
                            R: np.ndarray,
                            mu_a,
                            mu_b,
                            H,
                            K,
                            eps,
                            eps_0,
                            eta,
                            prob_a,
                            prob_b,
                            C = None,
                            ):
        """
        Compute the policy using MLE models

        :return: output_policy, baseline_return_flag
        """
        P_a = est_P[0]
        P_b = est_P[1]

        # get the beta (shape [|Z|,|S|,|A|])
        beta = self.estimate_beta(H, K)

        # placeholders for new rewards
        r_main = np.zeros((self.n_groups, self.n_states, self.n_actions))
        r_opt = np.zeros((self.n_groups, self.n_states, self.n_actions))
        r_pess = np.zeros((self.n_groups, self.n_states, self.n_actions))

        # do the reward shaping for r_main
        alpha = 1. + (self.n_groups * self.n_states * H) + (8. * H * (1. + self.n_groups * self.n_states * H))/eta
        for z in range(self.n_groups):
            if C is not None:
                r_main[z] = C.copy() + alpha * beta[z]
            else:
                r_main[z] = R.copy() + alpha * beta[z]

        # for optimistic and pessimistic reward
        op_scale_coefficient = 1. + self.n_groups * self.n_states * H
        for z in range(self.n_groups):
            # optimistic reward
            r_opt[z] = R.copy() + op_scale_coefficient * beta[z]
            # pessimistic reward
            r_pess[z] = R.copy() - op_scale_coefficient * beta[z]

        # if any of the violation conditions are true, return the pi_init

        # optimistic estimates wrt pi_init
        J0_opt_a = evaluate_ns_ep_policy(P=P_a, R=r_opt[0], mu=mu_a, H=H, pi=self.pi_init[0])
        J0_opt_b = evaluate_ns_ep_policy(P=P_b, R=r_opt[1], mu=mu_b, H=H, pi=self.pi_init[1])
        # pessimistic estimates wrt pi_init
        J0_pess_a = evaluate_ns_ep_policy(P=P_a, R=r_pess[0], mu=mu_a, H=H, pi=self.pi_init[0])
        J0_pess_b = evaluate_ns_ep_policy(P=P_b, R=r_pess[1], mu=mu_b, H=H, pi=self.pi_init[1])

        cond_1 = (prob_a * J0_opt_a - prob_b * J0_pess_b > (eps + eps_0)/2.)
        cond_2 = (prob_b * J0_opt_b - prob_a * J0_pess_a > (eps + eps_0)/2.)

        if cond_1 or cond_2:
            return self.pi_init, True

        # Else we will compute the policy from the feasible set
        pi_opt_a, pi_opt_b = opu_fair_lp_solver(P_a, P_b,
                                                r_main, r_opt, r_pess,
                                                mu_a, mu_b,
                                                H, eps,
                                                prob_a,
                                                prob_b)

        pi_opt = np.zeros((self.n_groups, H, self.n_states, self.n_actions))
        pi_opt[0] = pi_opt_a
        pi_opt[1] = pi_opt_b

        return pi_opt, False
