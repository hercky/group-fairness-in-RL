"""
Based on the code by CleanRL (Huang et al 2021), and FOCOPS (Zhang et al 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MinimalPPO:

    def __init__(self, args, subgroup, agent):

        self.subgroup = subgroup

        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        self.batch_size = args.batch_size
        self.update_epochs = args.update_epochs
        self.minibatch_size = args.minibatch_size
        self.norm_adv = args.norm_adv
        self.clip_coef = args.clip_coef

        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl
        self.l2_reg = args.l2_reg

    def update_params(self, rollouts, global_step, writer):

        b_obs = rollouts['obs']
        b_logprobs = rollouts['logprobs']
        b_actions = rollouts['actions']
        b_advantages = rollouts['advantages']
        b_returns = rollouts['returns']
        b_values = rollouts['values']

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        # Update in Minibatches
        for epoch in range(self.update_epochs):
            # shuffle the batch
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    # the above stats are serving as diagnostics

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # No value clipping in the minimal PPO version
                v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                for param in self.agent.critic.parameters():
                    v_loss += param.pow(2).sum() * self.l2_reg

                loss = pg_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(f"charts/{self.subgroup}/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar(f"losses/{self.subgroup}/value_loss", v_loss.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar(f"losses/{self.subgroup}/explained_variance", explained_var, global_step)
