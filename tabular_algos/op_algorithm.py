"""
A baseline that uses MLE estimates with the LP solver
"""
import numpy as np

from tabular_algos.base_algo import BaseTabularAlgo
from tabular_algos.opu_fair_lp_solver import opu_fair_lp_solver
from tabular_algos.utils import evaluate_ns_ep_policy

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

        # Note (!), scaling trick typically used in optimism based algorithms
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

        cond_1 = (J0_opt_a - J0_pess_b > (eps + eps_0)/2.)
        cond_2 = (J0_opt_b - J0_pess_a > (eps + eps_0)/2.)

        if cond_1 or cond_2:
            return self.pi_init, True

        # Else we will compute the policy from the feasible set
        pi_opt_a, pi_opt_b = opu_fair_lp_solver(P_a, P_b,
                                                r_main, r_opt, r_pess,
                                                mu_a, mu_b,
                                                H, eps)

        pi_opt = np.zeros((self.n_groups, H, self.n_states, self.n_actions))
        pi_opt[0] = pi_opt_a
        pi_opt[1] = pi_opt_b

        return pi_opt, False
