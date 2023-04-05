"""
Based on the code by CleanRL (Huang et al 2021), and FOCOPS (Zhang et al 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from deepRL_algos.base_agent import Agent



class FCPO:

    def __init__(self, args, subgroup, num_subgroups, agent):

        self.subgroup = subgroup
        self.num_subgroups = num_subgroups

        self.agent = agent

        self.actor_optimizer = optim.Adam([*self.agent.actor_mean.parameters(), self.agent.actor_logstd],
                                          lr=args.learning_rate, eps=1e-5)

        self.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.batch_size = args.batch_size
        self.update_epochs = args.update_epochs
        self.minibatch_size = args.minibatch_size
        self.norm_adv = args.norm_adv
        self.clip_coef = args.clip_coef

        # self.clip_vloss = args.clip_vloss
        # self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        # self.target_kl = args.target_kl

        self.nu = np.ones((num_subgroups-1, 2)) * args.nu_init
        self.nu_max = args.nu_max
        self.nu_lr = args.nu_lr
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.lam = args.lam
        self.eta = args.eta

        self.l2_reg = args.l2_reg

        self.num_steps = args.num_steps
        self.num_envs = args.num_envs



    def project_nu(self, nu):
        """
        project nu in [0, nu_max] range

        :param nu:
        :return:
        """
        if nu < 0:
            nu = 0
        elif nu > self.nu_max:
            nu = self.nu_max
        return nu

    def gaussian_kl(self, mean1, std1, mean2, std2):
        """
        Calculate KL-divergence between two Gaussian distributions N(mu1, sigma1) and N(mu2, sigma2)
        """
        normal1 = Normal(mean1, std1)
        normal2 = Normal(mean2, std2)
        return torch.distributions.kl.kl_divergence(normal1, normal2).sum(-1, keepdim=True)

    def update_params(self, rollouts, return_diff, device, global_step, writer):

        b_obs = rollouts['obs']
        b_logprobs = rollouts['logprobs']
        b_actions = rollouts['actions']
        b_advantages = rollouts['advantages']
        b_returns = rollouts['returns']
        b_values = rollouts['values']

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        # Get the log likelihood, mean, and std of current policy
        # containers for KL calculation
        # b_old_logprob = torch.zeros((self.num_steps, self.num_envs)).to(device).reshape(-1)
        # b_old_mean = torch.zeros((self.num_steps, self.num_envs)).to(device).reshape(-1)
        # b_old_std = torch.zeros((self.num_steps, self.num_envs)).to(device).reshape(-1)

        with torch.no_grad():
            b_old_logprob, b_old_mean, b_old_std = self.agent.get_dist(b_obs, b_actions)
            # Note: passes the following checks:
            #   - the device of tensor is correct (seems correct!)
            #   - detached (require_grad is False ok!)

        # Update nu values
        assert len(return_diff) == self.num_subgroups - 1, "Incorrect number of return difference between subgroups"

        for z in range(self.num_subgroups-1):
            self.nu[z][0] -= self.nu_lr * (self.epsilon - return_diff[z])
            self.nu[z][0] = self.project_nu(self.nu[z][0])

            self.nu[z][1] -= self.nu_lr * (self.epsilon + return_diff[z])
            self.nu[z][1] = self.project_nu(self.nu[z][1])

        # Update in Minibatches
        for epoch in range(self.update_epochs):
            # shuffle the batch
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # ----------------------------------------
                #       Value loss
                # ----------------------------------------
                newvalue = self.agent.get_value(b_obs[mb_inds])
                newvalue = newvalue.view(-1)
                # new way of calculating the losses
                v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()
                # vs
                # v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # weight decay
                for param in self.agent.critic.parameters():
                    v_loss += param.pow(2).sum() * self.l2_reg
                # TODO: maybe weight decay is unnecessary here

                # update the critic
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                self.critic_optimizer.step()

                # ----------------------------------------
                #       Policy loss
                # ----------------------------------------
                new_logprob, new_mean, new_std = self.agent.get_dist(b_obs[mb_inds], b_actions[mb_inds])

                kl_new_old = self.gaussian_kl(new_mean, new_std, b_old_mean[mb_inds], b_old_std[mb_inds])

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # advantage multiplier
                adv_coefficient = 1.
                for z in range(self.num_subgroups - 1):
                    adv_coefficient += (-self.nu[z][0] + self.nu[z][1])

                pg_loss = (kl_new_old - (1 / self.lam) * ratio * mb_advantages * adv_coefficient) \
                          * (kl_new_old.detach() <= self.eta).type(torch.float32)
                # pg_loss = (kl_new_old - (1 / self.lam) * ratio * mb_advantages * (1. - self.nu_1 + self.nu_2)) \
                #                * (kl_new_old.detach() <= self.eta).type(torch.float32)

                actor_loss = pg_loss.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.actor_mean.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.actor_logstd, self.max_grad_norm)
                self.actor_optimizer.step()

            # mini-batch over
            # Early stopping
            logprob, mean, std = self.agent.get_dist(b_obs, b_actions)
            kl_val = self.gaussian_kl(mean, std, b_old_mean, b_old_std).mean().item()
            if kl_val > self.delta:
                print(f"Break at epoch {epoch+1} because KL value {kl_val:.4f} larger than {self.delta:.4f}")
                break


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(f"charts/{self.subgroup}/pi_learning_rate", self.actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar(f"charts/{self.subgroup}/vf_learning_rate", self.critic_optimizer.param_groups[0]["lr"],
                          global_step)
        writer.add_scalar(f"losses/{self.subgroup}/value_loss", v_loss.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/policy_loss", actor_loss.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/explained_variance", explained_var, global_step)
