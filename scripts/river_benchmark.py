"""
usage
export PYTHONPATH=$PYTHONPATH:~/code/fpo

python river_benchmark.py --env_name ENV --K <num of episodes> --eta <eta>
    --delta DELTA --seed SEED --env_path <PATH>
    --ep_length EP_LENGTH --exp_name EXP_NAME --num_trajs NUM_TRAJS
"""
#!/usr/bin/env python

import argparse

from experiments.tabular.benchmark_river_swim import benchmark_river
from experiments.logx import setup_logger_kwargs

if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # env specific arguments
    parser.add_argument('--env_name', type=str, default='river')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--eps_init', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--num_trajs', type=int, default=1)
    parser.add_argument('--ep_length', type=int, default=10)
    # exp logging specific args
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='gridworld')
    parser.add_argument('--out_dir', type=str, default='/tmp/fair-rl/')
    # parse args
    args = parser.parse_args()

    # Starting the experiment now
    print(f"Starting the experiment: {args.exp_name}")

    # Run all baselines for only one env
    env_name = "/".join([args.env_name,
                         "beta_" + str(args.beta),
                         "K_"+str(args.K)])

    print(f"Environment: {env_name}")

    # Prepare a logger
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name,
                                        env_name=env_name,
                                        seed=args.seed,
                                        data_dir=args.out_dir,
                                        print_along=False,
                                        timestamp=True)
    # launch the experiment
    benchmark_river(K=args.K,
                   pi_init_type="input",
                   eps_0=args.eps_init,
                   beta_scale_factor=args.beta,
                   eta=args.eta,
                   num_trajs=args.num_trajs,
                   seed=args.seed,
                   logger_kwargs=logger_kwargs,
                   plotting=False,)

    print(f"Experiment finished")