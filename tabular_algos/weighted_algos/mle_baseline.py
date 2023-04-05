"""
A baseline that uses MLE estimates with the LP solver
"""
import numpy as np

from tabular_algos.weighted_algos.base_algo import BaseTabularAlgo
from tabular_algos.weighted_algos.fair_lp_solver import opt_fair_ep_mdp_lp

class MLE_Baseline(BaseTabularAlgo):
    """

    """
    def __init__(self,
                 n_groups: int,
                 n_states: int,
                 n_actions: int,
                 seed=0):
        """

        """
        super().__init__(n_groups, n_states, n_actions, seed)


    def compute_best_policy(self,
                            est_P: np.ndarray,
                            R,
                            mu_a,
                            mu_b,
                            H,
                            eps,
                            prob_a,
                            prob_b,
                            C = None,
                            ):
        """
        Compute the policy using MLE models

        :return:
        """
        P_a = est_P[0]
        P_b = est_P[1]

        pi_opt_a, pi_opt_b, _, _, _, _ = opt_fair_ep_mdp_lp(P_a, P_b, R, mu_a, mu_b,
                                                      H, eps, prob_a, prob_b, C=C)

        pi_opt = np.zeros((self.n_groups, H, self.n_states, self.n_actions))
        pi_opt[0] = pi_opt_a
        pi_opt[1] = pi_opt_b

        return pi_opt


