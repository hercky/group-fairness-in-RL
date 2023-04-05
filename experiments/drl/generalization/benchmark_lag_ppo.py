import os
import random
import time

import gym

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from deepRL_algos.base_agent import Agent
from deepRL_algos.data_generator import DataGenerator
from deepRL_algos.lag_ppo import LagrangianPPO

from experiments.drl.env_creation_utils import make_indv_env, get_envs_ids
from experiments.drl.utils import parse_args

from collections import deque
from itertools import combinations

import pandas as pd
from experiments.drl.generalization.test_rollout_generator import sample_return

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # ------------------------------------------------------- #
    #                       Init
    #           ( Do not modify this section )
    # ------------------------------------------------------- #
    args = parse_args()
    run_name = f"{args.env_case}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Tracking params
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=f"{args.log_path}",
        )

    writer = SummaryWriter(f"{args.log_path}/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set cuda if avail
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ------------------------------------------------------- #
    #            Setup Environments
    # ------------------------------------------------------- #

    # get env-ids
    env_ids = get_envs_ids(args.env_case)
    num_subgroups = len(env_ids)

    # Subgroup env setup
    envs = []
    lag_ppo = []
    data_gen = []

    for z in range(num_subgroups):
        # create the env
        envs.append(gym.vector.SyncVectorEnv(
            [make_indv_env(env_ids[z], args.seed + i, i, args.capture_video, run_name, args.log_path, z) for i in
             range(args.num_envs)]
        ))

        assert isinstance(envs[z].single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Setup Lagrangian PPO agent
        lag_ppo.append(LagrangianPPO(args=args, subgroup=z, num_subgroups=num_subgroups,
                                     agent=Agent(envs[z]).to(device)))

        # Setup Data Generators
        data_gen.append(DataGenerator(args=args, device=device,
                                      single_observation_space=envs[z].single_observation_space,
                                      single_action_space=envs[z].single_action_space,
                                      subgroup=z)
                        )

    # ------------------------------------------------------- #
    #           Create the test environments
    # ------------------------------------------------------- #
    test_envs = [[] for z in range(num_subgroups)]
    num_test_envs = 10

    for z in range(num_subgroups):
        for i in range(num_test_envs):
            # create the env
            test_envs[z].append(make_indv_env(env_ids[z], args.seed + i * 100, -1, False, run_name, args.log_path, z)())
            assert isinstance(test_envs[z][i].action_space, gym.spaces.Box), "only continuous action space is supported"

    test_results = []

    def update_test_results(step):
        """

        :param step:
        :return:
        """
        for i in range(num_test_envs):
            test_returns = np.zeros(num_subgroups, )
            for z in range(num_subgroups):
                test_returns[z] = sample_return(lag_ppo[z].agent, test_envs[z][i], device)
            # sampled once

            # Fairness related metrics
            test_fair_return = np.sum(test_returns)
            # maximum fair gap between different groups
            test_gap_between_returns = []
            for pair in combinations(range(num_subgroups), 2):
                test_gap_between_returns.append(abs(test_returns[pair[0]] - test_returns[pair[1]]))
            test_fair_gap = max(test_gap_between_returns)

            # add this in the df
            test_results.append({
                'fair_gap' : test_fair_gap,
                'fair_return' : test_fair_return,
                'seed': i*100 + args.seed,
                'global_step' : step,
                'algo' : "lag-ppo",
            })



    # ------------------------------------------------------- #
    #           Training Loop
    # ------------------------------------------------------- #

    start_time = time.time()
    subgroup_returns = np.zeros(num_subgroups,)
    cum_fair_violations = 0.
    # Queues for storing metrics
    running_cum_fair_return = deque(maxlen=100)
    running_fair_gap = deque(maxlen=100)
    running_subgroup_returns = [deque(maxlen=100) for _ in range(num_subgroups)]

    num_updates = args.total_timesteps // args.batch_size

    rollouts = []

    # collect the rollouts for each subgroup
    for z in range(num_subgroups):
        rollouts.append(data_gen[z].collect_data_and_targets(lag_ppo[z].agent, envs[z], writer))
        writer.add_scalar(f"charts/{z}/SPS", int(data_gen[z].global_step / (time.time() - start_time)),
                          data_gen[z].global_step)

    for update in range(1, num_updates + 1):

        # ---------- Log the statistics after one update --------------
        # get current estimated difference in returns
        for z in range(num_subgroups):
            subgroup_returns[z] = rollouts[z]['avg_return']

        # Fairness related metrics
        cum_fair_return = np.sum(subgroup_returns)
        # maximum fair gap between different groups
        gap_between_returns = []
        for pair in combinations(range(num_subgroups), 2):
            gap_between_returns.append(abs(subgroup_returns[pair[0]] - subgroup_returns[pair[1]]))
        fair_gap = max(gap_between_returns)
        # extra statistics
        fair_violation = float(fair_gap > args.epsilon)
        cum_fair_violations += fair_violation

        assert data_gen[0].global_step == data_gen[1].global_step, "Unequal amount of samples for both subgroups"

        writer.add_scalar(f"fair/fair_return", cum_fair_return, data_gen[0].global_step)
        writer.add_scalar(f"fair/fair_gap", fair_gap, data_gen[0].global_step)
        writer.add_scalar(f"fair/cum_violations", cum_fair_violations, data_gen[0].global_step)
        # log the average returns for the last policy for both subgroups
        for z in range(num_subgroups):
            writer.add_scalar(f"fair/{z}/avg_return", subgroup_returns[z], data_gen[z].global_step)

        # queue based plotting logic
        # add the metrics in a queue
        running_cum_fair_return.append(cum_fair_return)
        running_fair_gap.append(fair_gap)
        writer.add_scalar(f"queue/fair_return", np.mean(running_cum_fair_return), data_gen[0].global_step)
        writer.add_scalar(f"queue/fair_gap", np.mean(running_fair_gap), data_gen[0].global_step)
        # log the average returns for the last policy for both subgroups
        for z in range(num_subgroups):
            running_subgroup_returns[z].append(subgroup_returns[z])
            writer.add_scalar(f"queue/{z}/avg_return", np.mean(running_subgroup_returns[z]), data_gen[z].global_step)

        # -------- add the generalization results -----------
        update_test_results(data_gen[0].global_step)

        # ---------- Update the agents --------------
        # Note: can also udpate in a random update order

        for z0 in range(num_subgroups):

            return_diff = []
            # get diff of first wrt other
            for z1 in range(num_subgroups):
                if z0 == z1:
                    continue
                return_diff.append(rollouts[z0]['avg_return'] - rollouts[z1]['avg_return'])

            # update the subgroup
            lag_ppo[z0].update_params(rollouts[z0], return_diff, data_gen[z0].global_step, writer)

            # get new rollouts
            rollouts[z0] = data_gen[z0].collect_data_and_targets(lag_ppo[z0].agent, envs[z0], writer)
            writer.add_scalar(f"charts/{z0}/SPS", int(data_gen[z0].global_step / (time.time() - start_time)),
                              data_gen[z0].global_step)

        # # Update FIRST subgroup here
        # return_diff = rollouts[0]['avg_return'] - rollouts[1]['avg_return']
        # lag_ppo[0].update_params(rollouts[0], return_diff, data_gen[0].global_step, writer)
        #
        # # Get the new rollouts
        # rollouts[0] = data_gen[0].collect_data_and_targets(lag_ppo[0].agent, envs[0], writer)
        # writer.add_scalar(f"charts/{0}/SPS", int(data_gen[0].global_step / (time.time() - start_time)),
        #                   data_gen[0].global_step)

        # Note: changing the order of how we get updates determines if we are updating
        #    independently (update one after other and then new data)
        #    vs sequentially (update one -> collect new return -> use that to update the other).
        #    (move the rollout for first agent below the update for second group to be consistent with older version

    # done with everything
    for z in range(num_subgroups):
        envs[z].close()

    writer.close()


    # save the test_results as a dataframe
    df = pd.DataFrame(test_results)
    # save the df
    store_path = f"{args.log_path}/generalization/{run_name}"
    os.makedirs(store_path, exist_ok=True)
    df.to_csv(f"{store_path}/results.csv")
    df.to_pickle(path=f"{store_path}/results.pkl")
