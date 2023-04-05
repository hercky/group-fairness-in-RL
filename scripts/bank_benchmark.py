#!/usr/bin/env python

import argparse

from experiments.tabular.benchmark_bank import benchmark_bank
from experiments.logx import setup_logger_kwargs

if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # env specific arguments
    parser.add_argument('--algo', type=str, default='mle')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--eps_init', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--num_trajs', type=int, default=1)
    parser.add_argument('--group_dist', nargs=2, type=float, default=[0.5, 0.5])
    # exp logging specific args
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='/tmp/fair-rl/')
    # parse args
    args = parser.parse_args()

    # Starting the experiment now
    print(f"Starting the experiment: {args.exp_name}")

    env_name = f"{args.group_dist[0]}_{args.group_dist[1]}/K_{args.K}/{args.algo}/beta_{args.beta}"
    # Prepare a logger
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name,
                                        env_name=env_name,
                                        seed=args.seed,
                                        data_dir=args.out_dir,
                                        print_along=False,
                                        timestamp=True)
    # launch the experiment
    benchmark_bank(K=args.K,
                       eta=args.eta,
                       mode=args.algo,
                       group_dist=args.group_dist,
                       eps_0=args.eps_init,
                       beta_scale_factor=args.beta,
                       num_trajs=args.num_trajs,
                       seed=args.seed,
                       logger_kwargs=logger_kwargs,
                       plotting=False, )

    print(f"Experiment finished")