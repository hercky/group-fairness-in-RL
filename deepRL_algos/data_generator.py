"""
Based on the code by CleanRL (Huang et al 2021), and FOCOPS (Zhang et al 2020)

"""
import numpy as np
import torch

class DataGenerator:
    """
    To collect and store the on-policy data, i.e., rollout trajectories from a policy in an environment
    Also computes targets and GAE estimation
    """
    def __init__(self, args, device, single_observation_space, single_action_space, subgroup:int):
        """

        :param args:
        :param device:
        :param single_observation_space:
        :param single_action_space:
        :param subgroup:
        """

        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.device = device
        self.gae = args.gae
        self.gae_lambda = args.gae_lambda
        self.gamma = args.gamma
        self.subgroup = subgroup

        self.global_step = 0

        # storage setup
        # stores the batch: batch_size = num_steps x num_envs
        self.obs = torch.zeros((self.num_steps, self.num_envs) + single_observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + single_action_space.shape).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)


    def collect_data_and_targets(self, agent, envs, writer):
        """

        :return:
        """
        ret_hist = []

        next_obs = torch.Tensor(envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for step in range(0, self.num_steps):
            self.global_step += 1 * self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # log a sample return this particular global step
            for item in info:
                # if episode ended
                if "episode" in item.keys():
                    eps_return = item['episode']['r']
                    eps_len = item["episode"]["l"]
                    # print(f"subgroup={self.subgroup} global_step={global_step}, episodic_return={eps_return}")
                    writer.add_scalar(f"charts/{self.subgroup}/episodic_return", eps_return, self.global_step)
                    writer.add_scalar(f"charts/{self.subgroup}/episodic_length", eps_len, self.global_step)
                    break

            # Alternate storing logic
            # store the average statistics for this particular global step
            for item in info:
                if "episode" in item.keys():
                    eps_return = item['episode']['r']
                    # append the return to the history
                    ret_hist.append(eps_return)

        # average return under this policy
        avg_return = np.mean(ret_hist)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return {'obs': b_obs,
                'logprobs': b_logprobs,
                'actions': b_actions,
                'advantages': b_advantages,
                'returns': b_returns,
                'values': b_values,
                'avg_return': avg_return,
        }