"""
Define an experiment and run it
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.logx import Logger, setup_logger_kwargs
from envs.econ.lending import Lending, make_lending_env

from tabular_algos.utils import evaluate_ns_ep_policy
from tabular_algos.weighted_algos.fair_lp_solver import opt_fair_ep_mdp_lp
from tabular_algos.weighted_algos.mle_baseline import MLE_Baseline
from tabular_algos.weighted_algos.op_algorithm import OptPess_Fair_Algo

import matplotlib.pyplot as plt
import pickle

class Tracker(object):

    def __init__(self,
                name: str,
                j_opt: float, j_opt_a: float, j_opt_b: float,
                j_init: float,
                epsilon: float, delta: float,
                log_interval: int,
                plot_path: str = "/tmp/"):

        self.name = "Ours" if name == "op" else "MLE"
        self.J_opt = j_opt
        self.j_opt_a = j_opt_a
        self.j_opt_b = j_opt_b
        self.J_init = j_init
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.delta = delta
        self.path = plot_path

        # initialize the placeholders for all algorithms
        self.cum_regret = 0.
        self.cum_fair_violation = 0.
        self.hist_regret = []
        self.hist_fair_violation = []
        self.hist_avg_fair_violation = []
        self.pi_init_regret = 0.
        self.cum_pi_init_running = 0.
        self.hist_fair_gap = []
        self.hist_ret_a = []
        self.hist_ret_b = []
        self.hist_pi_init_util = []
        self.hist_pi_init_regret = []
        self.hist_regret_wo_norm = []
        self.hist_pi_init_regret_wo_norm = []

        self.results = []

    def update(self, bank_a, bank_b, loan_a, loan_b, epoch, init_pi = False):
        """
        returns are already weighted by the probability of the groups
        :return:
        """

        # store the returns

        # calculate the stats
        J_k = bank_a + bank_b
        assert not np.isnan(J_k), "Why is nan coming around?"

        eps_k = np.abs(loan_a - loan_b)
        fair_k = 1. - float(eps_k <= self.epsilon)  # unfair violation or not

        # updated the records
        self.cum_regret += (self.J_opt - J_k)
        self.cum_fair_violation += float(fair_k)
        self.cum_pi_init_running += float(init_pi)
        # Note: an alternate form of regret wrt pi_init can be defined as (\pi^k - \pi^0)
        self.pi_init_regret += (self.J_init - J_k)

        # add to the history log so far at
        if (epoch - 1) % self.log_interval == 0:  # because epoch starts at 1
            # calculate epoch specific stats
            norm_cum_regret = self.cum_regret / np.sqrt(epoch)
            num_fair_violation = self.cum_fair_violation
            avg_pi_init_until = self.cum_pi_init_running / epoch

            # update the dicts
            self.hist_regret.append(norm_cum_regret)
            self.hist_fair_violation.append(num_fair_violation)
            self.hist_avg_fair_violation.append(num_fair_violation / epoch)
            self.hist_regret_wo_norm.append(self.cum_regret)

            self.hist_fair_gap.append(eps_k)
            self.hist_ret_a.append(bank_a)
            self.hist_ret_b.append(bank_b)
            self.hist_pi_init_util.append(avg_pi_init_until)
            self.hist_pi_init_regret.append(self.pi_init_regret / np.sqrt(epoch))
            self.hist_pi_init_regret_wo_norm.append(self.pi_init_regret)

            # save stats
            self.results.append([self.name, epoch, J_k, eps_k, fair_k, self.cum_regret,
                            norm_cum_regret, num_fair_violation])

    def get_df(self):
        """

        :return:
        """

        df = pd.DataFrame({
            "hist_regret_wo_norm": self.hist_regret_wo_norm,
            "hist_fair_violation": self.hist_fair_violation,
            "hist_avg_fair_violation": self.hist_avg_fair_violation,
            "hist_fair_gap": self.hist_fair_gap,
            "hist_ret_a": self.hist_ret_a,
            "hist_ret_b": self.hist_ret_b,
            "hist_pi_init_regret_wo_norm": self.hist_pi_init_regret_wo_norm,
            "hist_pi_init_util": self.hist_pi_init_util,
        })

        # add other attributes
        df.j_opt_a = self.j_opt_a
        df.j_opt_b = self.j_opt_b

        return df


    def plot_regret_and_violations(self, show=False):
        """
        """
        width = 12 * 2  # 6
        height = 5 * 2  # 4.5

        fig, axes = plt.subplots(2, 4, figsize=(width, height))

        # num of iters
        x_range = np.arange(len(self.hist_regret))
        axis_title_font_size = 16

        # ------ first row ------------
        # regret plots
        axes[0, 0].plot(x_range, self.hist_regret_wo_norm, label=self.name, linestyle="--", )
        axes[0, 0].set_ylabel("Cumulative Regret", fontsize=axis_title_font_size)
        axes[0, 0].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 0].set_title("Cumulative Regret")
        axes[0, 0].legend(loc="upper left", fancybox=True, prop={'size': 12})

        # regret wrt baseline
        axes[0, 1].plot(x_range, self.hist_pi_init_regret_wo_norm, label=self.name, linestyle="--", )
        axes[0, 1].set_ylabel("Cumulative Regret w.r.t. $\pi^{0}$ ", fontsize=axis_title_font_size)
        axes[0, 1].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 1].set_title("Regret w.r.t. the initial fair policy ($\pi^{0}$)")
        axes[0, 1].legend(loc="upper right", fancybox=True, prop={'size': 12})

        # ret-A
        axes[0, 2].plot(x_range, self.hist_ret_a, label=self.name, linestyle="--", )
        axes[0, 2].axhline(y=self.j_opt_a, label="$J^{\pi^\star}_{1}$", linestyle=":", c='black')
        axes[0, 2].set_ylabel("Return for first subgroup,  $J^{k}_1(r,P)$", fontsize=axis_title_font_size)
        axes[0, 2].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 2].set_title("Return for first subgroup")
        axes[0, 2].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # ret-B
        axes[0, 3].plot(x_range, self.hist_ret_b, label=self.name, linestyle="--", )
        axes[0, 3].axhline(y=self.j_opt_b, label="$J^{\pi^\star}_2$", linestyle=":", c='black')
        axes[0, 3].set_ylabel("Return for second subgroup,  $J^{k}_2(r,P)$", fontsize=axis_title_font_size)
        axes[0, 3].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 3].set_title("Return for second subgroup")
        axes[0, 3].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # ------ Second row ------------
        # fairness violation
        axes[1, 0].plot(x_range, self.hist_fair_violation, label=self.name, linestyle="--", )
        axes[1, 0].set_ylabel(f"Cumulative fairness violation", fontsize=axis_title_font_size)
        axes[1, 0].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 0].set_title("Fairness violations ")
        axes[1, 0].legend(loc="upper left", fancybox=True, prop={'size': 12})

        # avg fair violation
        axes[1, 1].plot(x_range, self.hist_avg_fair_violation, label=self.name, linestyle="--", )
        axes[1, 1].axhline(y=self.delta, label="delta", linestyle=":", c='black')
        axes[1, 1].set_ylabel(f"Average fairness violation", fontsize=axis_title_font_size)
        axes[1, 1].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 1].set_title("Avg Fairness violations")
        axes[1, 1].legend(loc="upper left", fancybox=True, prop={'size': 12})
        axes[1, 1].set_ylim([-0.1, 1.1])

        # fairness gap
        axes[1, 2].plot(x_range, self.hist_fair_gap, label=self.name, linestyle="--", )
        axes[1, 2].axhline(y=self.epsilon, label="Fairness threshold ($\epsilon$)", linestyle=":", c='black')
        axes[1, 2].set_ylabel(f"Fairness gap $\epsilon^k$", fontsize=axis_title_font_size)
        axes[1, 2].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 2].set_title("Fairness gap")
        axes[1, 2].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # how many times pi^0 was used

        alpha_vis = 0.0 if self.name == "MLE" else 1.0
        axes[1, 3].plot(x_range, self.hist_pi_init_util, label=self.name, alpha=alpha_vis)  # phantom line
        axes[1, 3].set_ylabel("Avg amount of times $\pi^{0}$ was executed", fontsize=axis_title_font_size)
        axes[1, 3].set_xlabel(f"Epochs, 1 unit = {self.log_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 3].set_title("Initial fair policy usage rate")

        fig.tight_layout()

        if show:
            plt.show()
        else:
            # save figure
            plt.savefig(self.path + '/plot.png')

def benchmark_bank(K,
                       eta,
                       mode,
                       group_dist=[0.5, 0.5],
                       eps_0=0.01,
                       beta_scale_factor=1.,
                       # Experiment methodology parameters
                       num_trajs=1,
                       delta=0.1,
                       # Run specific parameters
                       seed=123,
                       logger=None,
                       logger_kwargs=dict(),
                       # Plotting params
                       plotting=False,
                       plot_interval=10,
                       num_of_pts=1000,
                       ):
    """

    """
    # =========================================================================
    #  Prepare logger, seed, and result store for this run
    # =========================================================================

    logger = Logger(**logger_kwargs) if logger is None else logger
    # logger also creates the output dir for storing things
    # logger_kwargs contains
    #   - output_dir
    #   - exp_name
    #   - output_filename
    #   - print on std output or only in logs
    # save the experiment variables in a config file
    logger.save_config(locals())

    # set the seed for numpy
    np.random.seed(seed)

    # define the logging interval, the number of pts in x-axis for the plot
    log_interval = K / num_of_pts
    assert log_interval > 0, "log_interval needs to be positive"
    logger.log(f"Logging interval is set to {log_interval}")
    logger.log(f"Prob A : {group_dist[0]}, B : {group_dist[1]}")

    # only supports two algorithms
    assert mode in ["mle", "op"], "Unknown algorithm mode"

    # =========================================================================
    #  Creates the environment
    # =========================================================================
    env = make_lending_env()

    # get the environment data
    P_a, P_b, R, BankR, mu_a, mu_b = env.get_mdp_paramters()
    H = env.ep_len
    num_groups = 2  # hard coded
    num_states = R.shape[0]
    num_actions = R.shape[1]

    prob_a = group_dist[0]
    prob_b = group_dist[1]

    # =========================================================================
    #  Set up the pi_inital and benchmark it
    # =========================================================================

    # compute the optimal pi corresponding to the input eps
    pi_init_a, pi_init_b, _, _, loan_a, loan_b = opt_fair_ep_mdp_lp(P_a, P_b,
                                                                  R,
                                                                  mu_a, mu_b,
                                                                  H, eps_0,
                                                                  prob_a,
                                                                  prob_b)

    bank_init_a = evaluate_ns_ep_policy(P=P_a, R=BankR, mu=mu_a, H=H, pi=pi_init_a) * prob_a
    bank_init_b = evaluate_ns_ep_policy(P=P_b, R=BankR, mu=mu_b, H=H, pi=pi_init_b) * prob_b

    # set the initial policy to this
    pi_initial = np.zeros((num_groups, H, num_states, num_actions))
    pi_initial[0] = pi_init_a
    pi_initial[1] = pi_init_b

    # TODO: pi_initial can be of shape (Z,S,A) or (Z,H,S,A) depending on context
    J_init = bank_init_a + bank_init_b
    logger.log("Bench-marked pi_initial")
    logger.log(f"Eps 0 : {eps_0:.4f}, J_init : {J_init:0.4f}")

    # =========================================================================
    #  Calculate the optimal fair policy
    # =========================================================================

    # get the final target eps
    epsilon = eps_0 + eta
    logger.log(f"Target epsilon: {epsilon}")

    # compute the optimal fair policy using true MDP parameters
    pi_star_a, pi_star_b, _, _, loan_a, loan_b = opt_fair_ep_mdp_lp(P_a, P_b, R,
                                                                    mu_a, mu_b,
                                                                    H, epsilon,
                                                                    prob_a,
                                                                    prob_b,
                                                                    C=BankR)

    # get the bank returns now
    bank_opt_a = evaluate_ns_ep_policy(P=P_a, R=BankR, mu=mu_a, H=H, pi=pi_star_a) * prob_a
    bank_opt_b = evaluate_ns_ep_policy(P=P_b, R=BankR, mu=mu_b, H=H, pi=pi_star_b) * prob_b

    J_opt = bank_opt_a + bank_opt_b
    logger.log("Computed optimal-fair policy")
    logger.log(f"Optimal-fair return: {J_opt:.4f}")
    logger.log(f"Optimal-fair fairness gap: {np.abs(loan_a - loan_b):.4f}")
    logger.log(f"Group A: {bank_opt_a:.4f}, Group B: {bank_opt_b:.4f}")

    # =========================================================================
    #  Initialize the algorithms and logging utils
    # =========================================================================

    algo = None

    # MLE baseline ('mle')
    if mode == "mle":
        algo = MLE_Baseline(n_groups=num_groups,
                            n_states=num_states,
                            n_actions=num_actions,
                            seed=seed)

    elif mode == "op":
        # Opt-Pess Algorithm ('op')
        algo = OptPess_Fair_Algo(n_groups=num_groups,
                                 n_states=num_states,
                                 n_actions=num_actions,
                                 delta=delta,
                                 pi_init=pi_initial,
                                 beta_scale_factor=beta_scale_factor,
                                 seed=seed)
    else:
        raise Exception("Unknown algorithm.")

    tracker = Tracker(name=mode,
                      j_opt=J_opt,
                      j_opt_a=bank_opt_a,
                      j_opt_b=bank_opt_b,
                      j_init=J_init,
                      epsilon=epsilon,
                      delta=delta,
                      log_interval=log_interval,
                      plot_path=logger.output_dir)

    # =========================================================================
    #  Run the algorithms for K episodes
    # =========================================================================

    # set the initial policies to baseline polices for all algorithms
    pi_k = np.copy(pi_initial)
    init_pi_running = True

    bank_a = bank_init_a
    bank_b = bank_init_b

    for k in tqdm(range(1, K+1), desc="num of episodes", disable=1-plotting):

        # sample a group here
        z = np.random.choice(np.arange(2), p=np.array(group_dist))

        # update the model if required
        if algo.check_double_experience_collected():

            # if collected enough experience for update,
            #   get the new policy based on new data

            est_P = algo.estimate_model()
            est_R = algo.obs_R.copy()

            prev_bank_a = bank_a
            prev_bank_b = bank_b
            prev_pi_k = pi_k

            if mode == "op":
                prev_init_pi_running = init_pi_running

            # get the update policy to run from this iteration
            if mode == "mle":
                pi_k = algo.compute_best_policy(est_P, est_R,
                                                mu_a, mu_b,
                                                H, epsilon,
                                                prob_a, prob_b,
                                                C=BankR)
            elif mode == "op":
                pi_k, init_pi_running = algo.compute_best_policy(est_P=est_P, R=est_R,
                                                                 mu_a=mu_a, mu_b=mu_b,
                                                                 H=H, K=K,
                                                                 eps=epsilon, eps_0=eps_0,
                                                                 eta=eta,
                                                                 prob_a=prob_a, prob_b=prob_b,
                                                                 C=BankR)

            # save the counts related for this policy update
            algo.save_current_counts()

            # evaluate pi_k based on true env parameters
            # banks
            bank_a = evaluate_ns_ep_policy(P=P_a, R=BankR, mu=mu_a, H=H, pi=pi_k[0]) * prob_a
            bank_b = evaluate_ns_ep_policy(P=P_b, R=BankR, mu=mu_b, H=H, pi=pi_k[1]) * prob_b
            # loans
            loan_a = evaluate_ns_ep_policy(P=P_a, R=R, mu=mu_a, H=H, pi=pi_k[0]) * prob_a
            loan_b = evaluate_ns_ep_policy(P=P_b, R=R, mu=mu_b, H=H, pi=pi_k[1]) * prob_b

            # TODO: if return is nan, don't make the switch to new policy
            if np.isnan(bank_a) or np.isnan(bank_b):
                bank_a = prev_bank_a
                bank_b = prev_bank_b
                pi_k = prev_pi_k
                if mode == "op":
                    init_pi_running = prev_init_pi_running

        # collect more data for this subgroup
        # run the policy in env and collect the transitions
        batch_traj = algo.collect_trajectories(num_trajs=num_trajs, env=env, z=z, pi=pi_k)
        # update the counts
        algo.update_counts(z=z, batch=batch_traj)

        if mode == "mle":
            tracker.update(bank_a, bank_b, loan_a, loan_b, k)
        else:
            tracker.update(bank_a, bank_b, loan_a, loan_b, k, init_pi=init_pi_running)


        # =========================================================================
        # Plotting during learning
        # =========================================================================
        if plotting and k % plot_interval == 0:
            tracker.plot_regret_and_violations(show=True)

    # =========================================================================
    # done with training
    # =========================================================================

    # save the tracker
    with open(f"{tracker.path}/track.pkl", 'wb') as save_file:
        pickle.dump(tracker, save_file, pickle.HIGHEST_PROTOCOL)

    # save the results
    # Dump the results in a dataframe
    df = tracker.get_df()
    df.to_csv(path_or_buf=f"{tracker.path}/data.csv")

    logger.log(f"Df saved to {tracker.path}!")

    # create a plot, and save that too!
    tracker.plot_regret_and_violations(show=False)
    logger.log(f"Created a plot in {logger.output_dir}.")

    logger.log(f"Experiment finished!")


