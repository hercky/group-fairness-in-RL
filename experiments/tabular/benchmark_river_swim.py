"""
Define an experiment and run it
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.logx import Logger, setup_logger_kwargs
from envs.chain.builder_tools import make_riverSwim

from tabular_algos.fair_lp_solver import opt_fair_ep_mdp_lp
from tabular_algos.utils import evaluate_ns_ep_policy, backward_policy_eval
from tabular_algos.mle_baseline import MLE_Baseline
from tabular_algos.op_algorithm import OptPess_Fair_Algo

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=True)

def benchmark_river(
                   K,
                   eta,
                   eps_0 = 0.1,
                   pi_init_type="input",    # random/input
                   beta_scale_factor = 1.,
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

    :param env_name:
    :param K:
    :param eta:
    :param pi_init_type:
    :param num_trajs:
    :param delta:
    :param env_layout_path:
    :param ep_length:
    :param successful_transition_prob_A:
    :param successful_transition_prob_B:
    :param goal_reward:
    :param per_step_reward:
    :param seed:
    :param logger:
    :param logger_kwargs:
    :param plotting:
    :param plot_interval:
    :return:
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
    # log_interval = K/num_pts
    log_interval = K / num_of_pts
    assert log_interval > 0, "log_interval needs to be positive"

    # Containers to store the results over learning
    results = []

    cum_regret = {}
    cum_fair_violation = {}
    pi_init_regret = {}
    cum_pi_init_running = {}
    j_a = {}
    j_b = {}

    hist_regret = {}
    hist_regret_wo_norm = {}
    hist_fair_violation = {}
    hist_avg_fair_violation = {}
    hist_pi_init_regret = {}    # regret wrt pi^0
    hist_pi_init_regret_wo_norm = {}
    hist_fair_gap = {}  # eps_k
    hist_pi_init_util = {}  # whether OP algorithm is using pi^0 or not
    hist_ret_a = {}
    hist_ret_b = {}



    # =========================================================================
    #  Creates the environment
    # =========================================================================

    # use default settings for now
    # env = make_riverSwim(seed=seed)
    env = make_riverSwim(seed=seed, spawn_prob=0.9999)

    # get the environment data
    P_a, P_b, R, mu_a, mu_b = env.get_mdp_paramters()
    H = env.ep_len
    num_groups = 2  # hard coded
    num_states = R.shape[0]
    num_actions = R.shape[1]

    # =========================================================================
    #  Set up the pi_inital and benchmark it
    # =========================================================================
    # j_init_a = j_init_b = None
    # pi_initial = None

    if pi_init_type == "random":
        # create a uniform policy and use that for initialization
        pass # for now
        # pi_random = np.ones((num_states, num_actions)) / num_actions
        # pi_initial = np.zeros((num_groups, num_states, num_actions))
        # for z in range(num_groups):
        #     pi_initial[z] = pi_random
        #
        # # get the initial eps corresponding to the initial policy
        # j_init_a = evaluate_ns_ep_policy(P=P_a, R=R, mu=mu_a, H=H, pi=pi_initial[0])
        # j_init_b = evaluate_ns_ep_policy(P=P_b, R=R, mu=mu_b, H=H, pi=pi_initial[1])
        # eps_0 = np.abs(j_init_a - j_init_b)

    elif pi_init_type == "input":
        # modify the original exploration policy
        R_wo_last = R.copy()
        R_wo_last[num_states - 1, 1] = 0.0

        # compute the optimal pi corresponding to the input eps
        pi_init_a, pi_init_b, j_init_a, j_init_b = opt_fair_ep_mdp_lp(P_a, P_b, R_wo_last,
                                                                      mu_a, mu_b, H,
                                                                      eps_0)

        # # compute the optimal pi corresponding to the input eps
        # pi_init_a, pi_init_b, j_init_a, j_init_b = opt_fair_ep_mdp_lp(P_a, P_b, R,
        #                                                               mu_a, mu_b, H,
        #                                                               eps_0)

        # set the initial policy to this
        pi_initial = np.zeros((num_groups, H, num_states, num_actions))
        pi_initial[0] = pi_init_a
        pi_initial[1] = pi_init_b
    else:
        raise Exception("Invalid initial pi given as an input!")

    # Note: pi_initial can be of shape (Z,S,A) or (Z,H,S,A) depending on context
    J_init = j_init_a + j_init_b
    logger.log("Bench-marked pi_initial")
    logger.log(f"Eps 0 : {eps_0:.4f}, J_init : {J_init:0.4f}")

    # =========================================================================
    #  Calculate the optimal fair policy
    # =========================================================================

    # get the final target eps
    epsilon = eps_0 + eta
    logger.log(f"Target epsilon: {epsilon}")

    # compute the optimal fair policy using true MDP parameters
    _, _, j_opt_a, j_opt_b = opt_fair_ep_mdp_lp(P_a, P_b, R, mu_a, mu_b, H, epsilon)
    J_opt = j_opt_a + j_opt_b

    logger.log("Computed optimal-fair policy")
    logger.log(f"Optimal-fair return: {J_opt:.4f}")
    logger.log(f"Optimal-fair fairness: {np.abs(j_opt_a - j_opt_b):.4f}")

    # =========================================================================
    #  Initialize the algorithms and logging utils
    # =========================================================================
    # MLE baseline ('mle')
    mle_baseline = MLE_Baseline(n_groups=num_groups,
                                n_states=num_states,
                                n_actions=num_actions,
                                seed=seed)

    # Opt-Pess Algorithm ('op')
    op_algo = OptPess_Fair_Algo(n_groups=num_groups,
                                n_states=num_states, n_actions=num_actions,
                                delta=delta,
                                pi_init=pi_initial,
                                beta_scale_factor=beta_scale_factor,
                                seed=seed)

    # TODO: add the opt-CMDP baseline here

    # set the initial policies to baseline polices for all algorithms
    mle_pi_k = np.copy(pi_initial)
    op_pi_k = np.copy(pi_initial)
    op_init_pi_running = True

    # initialize the placeholders for all algorithms
    for alg_name in ["mle", "op"]:
        cum_regret[alg_name] = 0.
        cum_fair_violation[alg_name] = 0.
        hist_regret[alg_name] = []
        hist_fair_violation[alg_name] = []
        hist_avg_fair_violation[alg_name] = []
        pi_init_regret[alg_name] = 0.
        cum_pi_init_running[alg_name] = 0.
        hist_fair_gap[alg_name] = []
        hist_ret_a[alg_name] = []
        hist_ret_b[alg_name] = []
        hist_pi_init_util[alg_name] = []
        hist_pi_init_regret[alg_name] = []
        j_a[alg_name] = 0.
        j_b[alg_name] = 0.
        hist_regret_wo_norm[alg_name] = []
        hist_pi_init_regret_wo_norm[alg_name] = []

    # define the logging util function
    def update_results(alg: str, epoch: int, init_pi = False):
        """
        helper method to update the global containers and log

        :param alg: name of algorithm
        :param epoch: iteration/k
        :param ret_a: j_a
        :param ret_b: j_b
        :param init_pi: flag that tells whether the initial

        :return: None
        """
        # get the estimated returns corresponding to algorithm
        ret_a = j_a[alg]
        ret_b = j_b[alg]

        # calculate the stats
        J_k = ret_a + ret_b

        assert not np.isnan(J_k), "Why is nan coming around?"

        eps_k = np.abs(ret_a - ret_b)
        fair_k = 1. - float(eps_k <= epsilon)  # unfair violation or not

        # updated the records
        cum_regret[alg] += (J_opt - J_k)
        cum_fair_violation[alg] += float(fair_k)
        cum_pi_init_running[alg] += float(init_pi)
        # Note: an alternate form of regret wrt pi_init can be defined as (\pi^k - \pi^0)
        pi_init_regret[alg] += (J_init - J_k)

        # add to the history log so far at
        if (epoch-1) % log_interval == 0:   # because epoch starts at 1
            # calculate epoch specific stats
            norm_cum_regret = cum_regret[alg] / np.sqrt(epoch)
            num_fair_violation = cum_fair_violation[alg]
            avg_pi_init_until = cum_pi_init_running[alg] / epoch

            # update the dicts
            hist_regret[alg].append(norm_cum_regret)
            hist_fair_violation[alg].append(num_fair_violation)
            hist_avg_fair_violation[alg].append(num_fair_violation / epoch)
            hist_regret_wo_norm[alg].append(cum_regret[alg])

            hist_fair_gap[alg].append(eps_k)
            hist_ret_a[alg].append(ret_a)
            hist_ret_b[alg].append(ret_b)
            hist_pi_init_util[alg].append(avg_pi_init_until)
            hist_pi_init_regret[alg].append(pi_init_regret[alg]/np.sqrt(epoch))
            hist_pi_init_regret_wo_norm[alg].append(pi_init_regret[alg])

            logger.log(f"Episode #: {epoch}")
            logger.log(f"Return of {alg} baseline: {J_k}")
            logger.log(f"Fair violation?: {fair_k}")

            # save stats
            results.append([alg, epoch, J_k, eps_k, fair_k, cum_regret[alg],
                            norm_cum_regret, num_fair_violation])

    # =========================================================================
    #  Run the algorithms for K episodes
    # =========================================================================
    for k in tqdm(range(1, K+1), desc="num of episodes", disable=1-plotting):

        # =========================================================================
        # MLE Baseline
        # =========================================================================
        # if (k - 1) % log_interval == 0:
        if mle_baseline.check_double_experience_collected():

            # if collected enough experience for update, do the PI update
            # this is true for k==1
            logger.log(f"------- MLE baseline -----------")
            # get the model estimates
            est_P = mle_baseline.estimate_model()
            # Note (!): R is deterministic for now, so we are just going to use the original R
            # est_R = R.copy()
            est_R = mle_baseline.obs_R.copy()

            prev_j_a = j_a['mle']
            prev_j_b = j_b['mle']
            prev_mle_pi_k = mle_pi_k

            # get the update policy to run from this iteration
            mle_pi_k = mle_baseline.compute_best_policy(est_P, est_R, mu_a, mu_b, H, epsilon)

            # save the counts related for this policy update
            mle_baseline.save_current_counts()

            # evaluate pi_k based on true env parameters
            # Note: Can be done via backward policy eval also (commented out for now)
            j_a['mle'] = evaluate_ns_ep_policy(P=P_a, R=R, mu=mu_a, H=H, pi=mle_pi_k[0])
            j_b['mle'] = evaluate_ns_ep_policy(P=P_b, R=R, mu=mu_b, H=H, pi=mle_pi_k[1])
            # logger.log(f"NS eval: {j_a}, {j_b} ")
            # j_a = backward_policy_eval(pi=mle_pi_k[0], P=P_a, R=R, mu=mu_a, H=H)
            # j_b = backward_policy_eval(pi=mle_pi_k[1], P=P_b, R=R, mu=mu_b, H=H)
            # logger.log(f"backward eval: {j_a}, {j_b} ")

            # TODO: if return is nan, don't make the switch to new policy
            if np.isnan(j_a['mle']) or np.isnan(j_b['mle']):
                j_a['mle'] = prev_j_a
                j_b['mle'] = prev_j_b
                op_pi_k = prev_mle_pi_k

        # run the policy in env and collect the transitions
        batch_traj = mle_baseline.collect_trajectories(num_trajs=num_trajs, env=env, pi=mle_pi_k)

        # update the counts
        mle_baseline.update_counts(batch=batch_traj)

        update_results('mle', k)

        # =========================================================================
        # Opt-Pess Algorithm
        # =========================================================================
        # if (k - 1) % log_interval == 0: # alt update rule
        if op_algo.check_double_experience_collected():

            logger.log(f"------- OptPess baseline -----------")
            # update the models
            est_P = op_algo.estimate_model()
            # Note (!) R is deterministic this case, so we are just going to use the original R
            # est_R = R.copy()
            est_R = op_algo.obs_R.copy()

            # get the policy to run at this iteration
            prev_j_a = j_a['op']
            prev_j_b = j_b['op']
            prev_op_init_pi_running = op_init_pi_running
            prev_op_pi_k = op_pi_k

            op_pi_k, op_init_pi_running = op_algo.compute_best_policy(est_P=est_P, R=est_R,
                                                                      mu_a=mu_a, mu_b=mu_b,
                                                                      H=H, K=K,
                                                                      eps=epsilon, eps_0=eps_0,
                                                                      eta=eta)

            # save the counts related for this policy update
            op_algo.save_current_counts()

            # evaluate pi_k based on true env parameters
            j_a['op'] = evaluate_ns_ep_policy(P=P_a, R=R, mu=mu_a, H=H, pi=op_pi_k[0])
            j_b['op'] = evaluate_ns_ep_policy(P=P_b, R=R, mu=mu_b, H=H, pi=op_pi_k[1])

            # if the new value contains nan due to numerical precision errors
            # revert to previous policy only
            if np.isnan(j_a['op']) or np.isnan(j_b['op']):

                j_a['op'] = prev_j_a
                j_b['op'] = prev_j_b
                op_pi_k = prev_op_pi_k
                op_init_pi_running = prev_op_init_pi_running


        # run the policy in env and collect the transitions
        batch_traj = op_algo.collect_trajectories(num_trajs=num_trajs, env=env, pi=op_pi_k)

        # update the counts
        op_algo.update_counts(batch=batch_traj)

        # update the logs
        update_results('op', k, init_pi=op_init_pi_running)

        # =========================================================================
        # Plotting during learning
        # =========================================================================
        if plotting and k % plot_interval == 0:
            # for checking the plots while running this function in a notebook
            plot_regret_and_violations(hist_regret, hist_fair_violation, hist_avg_fair_violation,
                                       hist_fair_gap, hist_ret_a, hist_ret_b, hist_pi_init_util, hist_pi_init_regret,
                                       hist_regret_wo_norm, hist_pi_init_regret_wo_norm,
                                       delta, j_opt_a, j_opt_b, epsilon,
                                       logging_interval=log_interval,
                                       show=True)

        # plot anyways after every tick if plotting disabled
        if not plotting and (k - 1) % (log_interval * 10) == 0:
            # instead of plotting 1k times, we'll only plot 100 times (10x less times)
            plot_regret_and_violations(hist_regret, hist_fair_violation, hist_avg_fair_violation,
                                       hist_fair_gap, hist_ret_a, hist_ret_b, hist_pi_init_util, hist_pi_init_regret,
                                       hist_regret_wo_norm, hist_pi_init_regret_wo_norm,
                                       delta, j_opt_a, j_opt_b, epsilon,
                                       logging_interval=log_interval,
                                       path=logger.output_dir,
                                       show=False)

    # =========================================================================
    # done with training
    # =========================================================================

    # save the results
    # Dump the results in a dataframe
    df = pd.DataFrame(results, columns=['algorithm', 'episode_num', 'alg_return', 'eps', 'fair',
                                        'cum_regret', 'norm_cum_regret', 'cum_fair_violation',
                                        ])

    # Save the files here
    logger.dump_df_as_xlsx(df)
    df.to_csv(path_or_buf=logger.result_file + ".csv")
    logger.log(f"{len(results)} lines saved to {logger.result_file} in .xlsx and .csv")

    # create a plot, and save that too!
    plot_regret_and_violations(hist_regret, hist_fair_violation, hist_avg_fair_violation,
                               hist_fair_gap, hist_ret_a, hist_ret_b, hist_pi_init_util, hist_pi_init_regret,
                               hist_regret_wo_norm, hist_pi_init_regret_wo_norm,
                               delta, j_opt_a, j_opt_b, epsilon,
                               logging_interval=log_interval,
                               show=False, path=logger.output_dir)
    logger.log(f"Created a plot in {logger.output_dir}.")

    logger.log(f"Experiment finished!")


def plot_regret_and_violations(regret, fair_violation, avg_fair_violation,
                               fair_gap, ret_a, ret_b, avg_init_pi, init_pi_regret,
                               regret_wo_norm, init_pi_regret_wo_norm,
                               delta, opt_ret_a, opt_ret_b, epsilon,
                               logging_interval,
                               show=False,
                               path="/tmp/"):
        """
        plot how the training looks for debugging


        hist_fair_gap[alg].append(eps_k)
            hist_ret_a[alg].append(ret_a)
            hist_ret_b[alg].append(ret_b)
            hist_pi_init_util[alg].append(avg_pi_init_until)
            # Note: the def of regret wrt pi_init has changed not (\pi^k - \pi^0)
            hist_pi_init_regret[alg].append(J_k - J_init)

        """
        # Create sub-plot for 2 plots; Cum-regret and fariness violation
        # safety, duality gap
        # Bottom R_perf, C_perf (w/ baseline performance)
        width = 12 * 2 # 6
        height = 5  * 2 # 4.5
        # fig, axes = plt.subplots(4, 2, figsize=(width, height))
        fig, axes = plt.subplots(2, 4, figsize=(width, height))

        # num of iters
        x_range = np.arange(len(regret['mle']))

        axis_title_font_size = 16

        # ------ first row ------------
        # regret plots
        axes[0, 0].plot(x_range, regret_wo_norm['mle'], label="MLE", linestyle="--",)
        axes[0, 0].plot(x_range, regret_wo_norm['op'], label="Ours",)
        axes[0, 0].set_ylabel("Cumulative Regret", fontsize=axis_title_font_size)
        axes[0, 0].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 0].set_title("Cumulative Regret")
        axes[0, 0].legend(loc="upper left", fancybox=True, prop={'size': 12})

        # regret wrt baseline
        axes[0, 1].plot(x_range, init_pi_regret_wo_norm['mle'], label="MLE", linestyle="--", )
        axes[0, 1].plot(x_range, init_pi_regret_wo_norm['op'], label="Ours", )
        axes[0, 1].set_ylabel("Cumulative Regret w.r.t. $\pi^{0}$ ", fontsize=axis_title_font_size)
        axes[0, 1].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[0, 1].set_title("Regret w.r.t. the initial fair policy ($\pi^{0}$)")
        axes[0, 1].legend(loc="upper right", fancybox=True, prop={'size': 12})

        # ret-A
        axes[0, 2].plot(x_range, ret_a['mle'], label="MLE", linestyle="--",)
        axes[0, 2].plot(x_range, ret_a['op'], label="Ours", )
        axes[0, 2].axhline(y=opt_ret_a, label="$J^{\pi^\star}_{1}$", linestyle=":", c='black')
        axes[0, 2].set_ylabel("Return for first subgroup,  $J^{k}_1(r,P)$", fontsize=axis_title_font_size )
        axes[0, 2].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size )
        axes[0, 2].set_title("Return for first subgroup")
        axes[0, 2].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # ret-B
        axes[0, 3].plot(x_range, ret_b['mle'], label="MLE", linestyle="--",)
        axes[0, 3].plot(x_range, ret_b['op'], label="Ours")
        axes[0, 3].axhline(y=opt_ret_b, label="$J^{\pi^\star}_2$", linestyle=":", c='black')
        axes[0, 3].set_ylabel("Return for second subgroup,  $J^{k}_2(r,P)$", fontsize=axis_title_font_size )
        axes[0, 3].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size )
        axes[0, 3].set_title("Return for second subgroup")
        axes[0, 3].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # ------ Second row ------------
        # fairness violation
        axes[1, 0].plot(x_range, fair_violation['mle'], label="MLE", linestyle="--", )
        axes[1, 0].plot(x_range, fair_violation['op'], label="Ours",)
        axes[1, 0].set_ylabel(f"Cumulative fairness violation", fontsize=axis_title_font_size)
        axes[1, 0].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 0].set_title("Fairness violations ")
        axes[1, 0].legend(loc="upper left", fancybox=True, prop={'size': 12})

        # avg fair violation
        axes[1, 1].plot(x_range, avg_fair_violation['mle'], label="MLE", linestyle="--",)
        axes[1, 1].plot(x_range, avg_fair_violation['op'], label="Ours", )
        axes[1, 1].axhline(y=delta, label="delta", linestyle=":", c='black')
        axes[1, 1].set_ylabel(f"Average fairness violation", fontsize=axis_title_font_size)
        axes[1, 1].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 1].set_title("Avg Fairness violations")
        axes[1, 1].legend(loc="upper left", fancybox=True, prop={'size': 12})
        axes[1, 1].set_ylim([-0.1, 1.1])

        # fairness gap
        axes[1, 2].plot(x_range, fair_gap['mle'], label="MLE", linestyle="--",)
        axes[1, 2].plot(x_range, fair_gap['op'], label="Ours",)
        axes[1, 2].axhline(y=epsilon, label="Fairness threshold ($\epsilon$)", linestyle=":", c='black')
        axes[1, 2].set_ylabel(f"Fairness gap $\epsilon^k$", fontsize=axis_title_font_size)
        axes[1, 2].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 2].set_title("Fairness gap")
        axes[1, 2].legend(loc="lower right", fancybox=True, prop={'size': 12})

        # how many times pi^0 was used
        axes[1, 3].plot(x_range, avg_init_pi['op'], label="MLE", alpha=0.0) # phantom line
        axes[1, 3].plot(x_range, avg_init_pi['op'], label="Ours",)
        axes[1, 3].set_ylabel("Avg amount of times $\pi^{0}$ was executed", fontsize=axis_title_font_size)
        axes[1, 3].set_xlabel(f"Epochs, 1 unit = {logging_interval} episodes", fontsize=axis_title_font_size)
        axes[1, 3].set_title("Initial fair policy usage rate")


        fig.tight_layout()

        if show:
            plt.show()
        else:
            # save figure
            plt.savefig(path+'/plot.png')

