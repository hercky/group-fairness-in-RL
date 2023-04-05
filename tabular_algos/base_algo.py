"""
Skeleton for the different tabular algorithms
"""

import numpy as np


class BaseTabularAlgo(object):
    """
    Abstract Tabular Algorithm wrapper.
    """

    def __init__(self, n_groups: int,
                 n_states: int, n_actions: int, seed=0):
        """
        :param seed: A seed for the random number generator.
        """
        self.n_groups = n_groups
        self.n_states = n_states
        self.n_actions = n_actions
        # init the initial counts
        self.count_P = np.zeros((self.n_groups, self.n_states, self.n_actions, self.n_states))
        self.prev_count_P = np.zeros((self.n_groups, self.n_states, self.n_actions, self.n_states))
        # the reward estimator
        self.obs_R = np.zeros((self.n_states, self.n_actions))
        # the seed
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def update_counts(self, batch):
        """
        Update the empirical counts based on the observe trajectory

        :param batch: a dict with keys |Z|, each with value |num_trajectories|
        :param subgroup: the subgroup whose model is being updated
        :return:
        """
        for z in range(self.n_groups):
            assert z in batch , 'subgroup trajectory not collected in batch'
            for traj in batch[z]:
                for transition in traj:
                    state = transition[0]
                    action = transition[1]
                    reward = transition[2]
                    next_state = transition[-1]
                    # update the count
                    self.count_P[z, state, action, next_state] += 1.0
                    # update the reward
                    # as the reward function is deterministic, observing a reward once is enough
                    if reward > 0. and self.obs_R[state, action] == 0.:
                        self.obs_R[state, action] = reward

    def estimate_model(self, zero_unseen=False):
        """
        build the MLE transition \hat{P} here

        :return: MLE estimate for each subroups's transition dynamics, shape: [|Z|,|S|,|A|,|S|]
        """
        est_P = np.zeros((self.n_groups, self.n_states, self.n_actions, self.n_states))

        for z in range(self.n_groups):
            if zero_unseen:
                # N(s,a) = \max{N(s,a), 1}
                normalized_count = np.maximum(np.sum(self.count_P[z], 2), 1.)
                # do the normalization here
                est_P[z] = self.count_P[z] / normalized_count[:, :, np.newaxis]
            else:
                est_P[z] = self.count_P[z] / np.sum(self.count_P[z], 2)[:, :, np.newaxis]
                est_P[z][np.isnan(est_P[z])] = 1.0 / self.n_states

        return est_P

    def collect_trajectories(self,
                             num_trajs: int,
                             env,
                             pi: np.ndarray):
        """
        Collect the trajectories for both subgroups from the environment

        :param env:
        :param pi: shape [Z, H, S, A] or [Z,S,A]
        :return:
        """
        batch = {}
        assert len(pi.shape) == 3 or len(pi.shape) == 4, "Incorrect shape of pi"
        for z in range(self.n_groups):
            # collect trajectories for this subgroup
            trajectories = []
            for _ in range(num_trajs):
                traj = []
                # sample a single traj here
                state = env.reset(subgroup=z)
                done = False
                h = 0
                while not done:
                    if len(pi.shape) == 4:
                        action = np.random.choice(self.n_actions, p=pi[z, h, state])
                    else:
                        action = np.random.choice(self.n_actions, p=pi[z, state])
                    next_state, reward, done, info = env.step(action)
                    traj.append([state, action, reward, next_state])
                    state = next_state
                    h += 1

                trajectories.append(traj)

            batch[z] = trajectories

        return batch

    def check_double_experience_collected(self):
        """
        checks if the for all (s,a) pairs, the counts in count_P >= 2 * prev_count_P

        :return:
        """
        return (self.count_P >= 2 * self.prev_count_P).any()

    def save_current_counts(self):
        """
        wrapper to update the previous old count matrix with the updated counts

        :return:
        """
        # memory efficient way
        # np.copyto(self.prev_count_P, self.count_P)

        # # less memory efficient method
        self.prev_count_P = np.copy(self.count_P)

