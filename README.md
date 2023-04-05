# Group Fairness in Reinforcement Learning  

Accompanying codebase for the submission to TMLR 2022

### Setup

Create a conda environment using the corresponding `_requirements.txt` to install all the dependent libraries.

```
$ conda create --name <env> --file <path/to/requirements.txt>
```

Use the `tabular_requirements.txt` to create conda environment for tabular experiments (Section 3), and use `deepRL_requirements.txt` to create conda environment for Deep-RL experiments (Section 4).

### Code Organization

Here is the high level structure of the components for this project:

- `envs` contains the implementations of different RL environments: RiverSwim, MuJoCo and Credit Lending environments
- `tabular_algos` contains the implementation for different algorithms for the  non-stationary finite horizon case (Section 3). The `weighted_algos` denote the extended implementations for the tabular algorithms for the setting when the group sampling distribution ($\Delta_{Z}$) is non-uniform.

- `deepRL_algos` contains the implementation for algorithms described for the DeepRL setting (Section 4).

- `scripts` contains the procedure for benchmarking different algorithms for the tabular setting (RiverSwim, CreditLending).

- `experiments` has the main driver code that utilizes the above components to run an experiment.

- `plots` contain a sample utility notebook for plotting the results for the HalfCheetah based experiments.

  
### Usage examples 

The first step is to add the current path to the python path, 
```
export PYTHONPATH=$PYTHONPATH:/path/to/codebase
``` 

#### DeepRL experiments

Run the corresponding scripts in `experiments/drl/` to launch an experiment. For instance,
to run Lagrangian-PPO baseline for the Half-Cheetah based tasks:
 
   
* Launch the experiment using the command,
    ```
    python -W ignore experiments/drl/train_lag_ppo.py  --exp-name <exp_name>  --log-path <log/path> --total-timesteps 2000000  --wandb-project-name <name> --wandb-entity <username> --track   --capture-video  --epsilon 1000   --env-case cheetah-family   --nu-max 1000.0   --nu-lr 0.01   --seed 13
    
    ```

* Similarly, it is possible to launch the FOC-PPO experiment for the Navigation task using:
    ```
    python -W ignore experiments/drl/train_fcpo.py  --exp-name <exp_name>  --log-path <log/path>  --total-timesteps 4000000  --num-steps 512  --num-envs 16  --num-minibatches 4  --anneal-lr False  --wandb-project-name <name> --wandb-entity <username>  --track   --capture-video  --epsilon 0.5   --seed 3   --env-case u-maze   --nu-max 1000.0   --nu-lr 0.01   --lam 1.0 
    ```

* For the economy tasks and PPO baseline, use the following:
    ```
    python -W ignore experiments/drl/train_ppo.py  --exp-name <exp_name>  --log-path <log/path>  --total-timesteps 2000000  --num-steps 512  --num-envs 16  --num-minibatches 4  --anneal-lr False  --wandb-project-name <name> --wandb-entity <username> --track  --env-case econs-low   --seed 2
    ```

#### Tabular experiments

Use the following command format for launching the RiverSwim experiments: 
```
python -W ignore scripts/river_benchmark.py --out_dir <logs>   --exp_name <name>  --seed 123   --K 20000  --beta 1e-04
 
```


