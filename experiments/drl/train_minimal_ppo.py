import os
import random
import time

import gym

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from deepRL_algos.base_agent import Agent
from deepRL_algos.data_generator import DataGenerator
from deepRL_algos.minimal_ppo import MinimalPPO

from experiments.drl.env_creation_utils import make_indv_env, get_envs_ids
from experiments.drl.utils import parse_args

from collections import deque
from itertools import combinations

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
    mppo = []
    data_gen = []

    for z in range(num_subgroups):
        # create the env
        envs.append(gym.vector.SyncVectorEnv(
            [make_indv_env(env_ids[z], args.seed + i, i, args.capture_video, run_name, args.log_path, z) for i in
             range(args.num_envs)]
        ))

        assert isinstance(envs[z].single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Setup FCPO agent
        mppo.append(MinimalPPO(args=args, subgroup=z, agent=Agent(envs[z]).to(device)))

        # Setup Data Generators
        data_gen.append(DataGenerator(args=args, device=device,
                                      single_observation_space=envs[z].single_observation_space,
                                      single_action_space=envs[z].single_action_space,
                                      subgroup=z)
                        )

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
        rollouts.append(data_gen[z].collect_data_and_targets(mppo[z].agent, envs[z], writer))
        writer.add_scalar(f"charts/{z}/SPS", int(data_gen[z].global_step / (time.time() - start_time)),
                          data_gen[z].global_step)

    for update in range(1, num_updates + 1):

        # No annealing for fcpo

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


        # ---------- Update the agents --------------
        for z in range(num_subgroups):
            mppo[z].update_params(rollouts[z], data_gen[z].global_step, writer)
            # Update the rollouts
            rollouts[z] = data_gen[z].collect_data_and_targets(mppo[z].agent, envs[z], writer)
            writer.add_scalar(f"charts/{z}/SPS", int(data_gen[z].global_step / (time.time() - start_time)),
                              data_gen[z].global_step)


    # done with everything
    for z in range(num_subgroups):
        envs[z].close()

    writer.close()
