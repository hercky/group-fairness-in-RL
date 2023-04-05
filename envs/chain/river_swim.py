import numpy as np

from envs.common import Env
from envs.exceptions import InvalidActionError, EpisodeDoneError

class RiverSwim(Env):

    def __init__(self, P_a, P_b, R, mu_a, mu_b, ep_len, seed=0):
        super().__init__(seed)

        self.P_a = P_a
        self.P_b = P_b
        self.R = R
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.ep_len = ep_len

        # extract info from init variables
        self.mu_z = [self.mu_a, self.mu_b]
        self.num_subgroups = 2
        self.state_space = P_a.shape[0]
        self.action_space = R.shape[1]

        # init state parameters
        self.time_step = 0
        self.current_state = 0
        self.current_subgroup = 0
        self.done = False

    def reset(self, subgroup:int = None):
        '''Reset the environment'''
        # sample z
        if subgroup is None:
            self.current_subgroup = np.random.randint(self.num_subgroups)
        else:
            self.current_subgroup = subgroup
            assert 0 <= subgroup <= self.num_subgroups, 'subgroup out of range'

        # sample s_0
        self.current_state = np.random.choice(np.arange(self.state_space), p=self.mu_z[self.current_subgroup])
        self.done = False
        self.time_step = 0

        return self.current_state

    def step(self, action):

        if self.done:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')

        if action >= self.action_space or not isinstance(action, int):
            raise InvalidActionError(
                'Invalid action {}. It must be an integer between 0 and {}'.format(action, self.action_space - 1))

        # get the current state
        current_state_idx = self.current_state

        if self.current_subgroup == 0:
            next_state_probs = self.P_a[current_state_idx, action]
        elif self.current_subgroup == 1:
            next_state_probs = self.P_b[current_state_idx, action]
        else:
            raise Exception("Unknown subgroup")

        # sample the next state P(s'|s,a)
        sampled_next_state = self.rng.choice(np.arange(self.state_space), p=next_state_probs)
        # observe the reward R(s,a)
        reward = self.R[current_state_idx, action]

        # update the state
        self.current_state = sampled_next_state

        # update time-step
        self.time_step += 1

        # if we have reached the maximum length of an episode we terminate it
        if self.time_step >= self.ep_len:
            self.done = True

        info = {'subgroup': self.current_subgroup,
                'time-step': self.time_step,
                }

        return self.current_state, reward, self.done, info

    def get_mdp_paramters(self):
        """
        Return a copy of the matrices corresponding to the different MDP parameters
        :return:
        """
        return self.P_a.copy(), self.P_b.copy(), self.R.copy(), self.mu_a.copy(), self.mu_b.copy()