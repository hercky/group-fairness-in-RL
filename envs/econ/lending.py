import copy

import numpy as np

from envs.common import Env
from envs.exceptions import InvalidActionError, EpisodeDoneError

ACCEPT_LOAN = 0
REJECT_LOAN = 1

class Lending(Env):

    def __init__(self, P_a, P_b, R, BankR, mu_a, mu_b, ep_len, seed=0):
        super().__init__(seed)

        self.P_a = P_a
        self.P_b = P_b
        self.R = R
        self.BankR = BankR
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.ep_len = ep_len

        # extract info from init variables
        self.mu_z = [self.mu_a, self.mu_b]
        self.num_subgroups = 2
        self.state_space = R.shape[0]
        self.action_space = R.shape[1]

        # init state parameters
        self.time_step = 0
        self.current_state = 0
        self.current_subgroup = 0
        self.done = False

    def reset(self, subgroup):
        '''Reset the environment'''
        # sample z

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

        # cost function for the bank
        bank_reward = self.BankR[current_state_idx, action]

        # update the state
        self.current_state = sampled_next_state

        # update time-step
        self.time_step += 1

        # if we have reached the maximum length of an episode we terminate it
        if self.time_step >= self.ep_len:
            self.done = True

        info = {'subgroup': self.current_subgroup,
                'time-step': self.time_step,
                'bank_reward': bank_reward,
                }

        return self.current_state, reward, self.done, info

    def get_mdp_paramters(self):
        """
        Return a copy of the matrices corresponding to the different MDP parameters
        :return:
        """
        return self.P_a.copy(), self.P_b.copy(), self.R.copy(), self.BankR.copy(), self.mu_a.copy(), self.mu_b.copy()



def create_transition_matrix(repayment_prob,
                             marginalized=False,
                             handicap=0.7,
                             num_states=7,
                             num_actions=2):
    """
    """
    P = np.zeros((num_states, num_actions, num_states))

    for s in range(num_states):

        # if loan accpeted then
        # calculate prob of replayment
        prob_repay = repayment_prob[s]
        prob_default = 1. - prob_repay

        P[s, ACCEPT_LOAN, min(num_states-1, s+1)] = prob_repay
        P[s, ACCEPT_LOAN, max(0, s-1)] = prob_default

        # rejected loan
        if marginalized:
            P[s, REJECT_LOAN, min(0,s-1)] = handicap
            P[s, REJECT_LOAN, s] = 1. - handicap
        else:
            P[s, REJECT_LOAN, s] = 1.0

    return copy.deepcopy(P)


def make_lending_env(interest=4.,
                     principal=10.,
                     handicap=0.7,
                     ep_len: int = 5,
                     num_states: int = 7,
                     seed: int = 0):
    """
    :return:
    """
    num_actions = 2

    # -------------------------------------------------
    #       Define the mu_z
    # -------------------------------------------------
    mu_a = np.array([0.1, 0.0, 0.1, 0.1, 0.3, 0.3, 0.1])
    mu_b = np.array([0.1, 0.0, 0.3, 0.3, 0.1, 0.1, 0.1])

    # -------------------------------------------------
    #       Define the R
    # -------------------------------------------------
    R = np.zeros((num_states, num_actions))

    # reward for being granted a loan
    #   -> fairness implies equal number of loans are granted to both sides
    for s in range(num_states):
        R[s, ACCEPT_LOAN] = 1.0

    # -------------------------------------------------
    #       Define the dynamics (P)
    # -------------------------------------------------
    # define the repayment prob
    repayment_prob = np.array([0.1, 0.5, 0.7, 0.8, 0.9, 0.9, 1.0])

    P_a = create_transition_matrix(repayment_prob=repayment_prob, marginalized=False)
    P_b = create_transition_matrix(repayment_prob=repayment_prob, marginalized=True, handicap=handicap)

    # -------------------------------------------------
    #       Define the Bank Reward
    # -------------------------------------------------
    C = np.zeros((num_states, num_actions))

    for s in range(num_states):

        # bank gets reward if
        C[s, ACCEPT_LOAN] = repayment_prob[s] * interest - (1. - repayment_prob[s]) * principal

        # bank loses nothing for not granting any loans
        C[s, REJECT_LOAN] = 0.0

    # -------------------------------------------------
    #       Create the environment and return it
    # -------------------------------------------------


    return Lending(P_a=P_a, P_b=P_b,
                   BankR=C, R=R,
                   mu_a=mu_a,
                   mu_b=mu_b,
                   ep_len=ep_len,
                   seed=seed)
