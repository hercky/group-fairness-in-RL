import numpy as np
import gym
import random

from gym import Env, spaces
from envs.econ.solow import Solow_delong
from envs.exceptions import InvalidActionError, EpisodeDoneError


class Economy(Env):
    """
    Implements the Solow growth model calculation of the
    capital-output ratio κ and other model variables
    using the update rule:

    κ_{t+1} = κ_t + ( 1 - α) ( s - (n+g+δ)κ_t )

    Built upon and modified from Stachurski-Sargeant
    <https://quantecon.org> class **Solow**
    <https://lectures.quantecon.org/py/python_oop.html>
    """

    def __init__(self,
                 n=0.01,  # population growth rate
                 s=0.20,  # savings rate
                 delta=0.03,  # depreciation rate
                 alpha=1 / 3,  # share of capital
                 g=0.01,  # productivity
                 k=0.2 / (.01 + .01 + .03),  # current capital-labor ratio
                 E=1.0,  # current efficiency of labor
                 L=1.0,  # current labor force
                 low_dg=0.001,  # change in investment
                 high_dg=0.001,
                 limit_g_low=0.0,
                 limit_g_high=1.0,
                 ):

        super().__init__()
        # copy variables
        self.n, self.s, self.delta, self.alpha, self.g = n, s, delta, alpha, g
        self.k, self.E, self.L = k, E, L

        # Define an action space ranging from 0 to 2
        #    [No-change, INCREASE, DECREASE ]
        # self.action_space = spaces.Discrete(3,)
        # self.action_space = spaces.Box(low=-dg, high=dg, shape=(1,))
        self.action_space = spaces.Box(low=low_dg, high=high_dg, shape=(1,))
        self.limit_g_low = limit_g_low
        self.limit_g_high = limit_g_high

        # Define a 1-D observation space containing all parameters
        #   [Y, K, k, alpha1]
        self.observation_shape = (4,)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.inf * np.ones(self.observation_shape))

        # initialize the environment specific variables variables
        self.model = None
        self.done = False

        self.reset()

    def step(self, action):
        """

        :param action:
        :return:
        """
        if self.done:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')

        # Assert that it is a valid action
        # assert self.action_space.contains(action), "Invalid Action"

        # call the model here
        self.model.update_investment(action)
        self.model.update()

        reward = self.model.get_reward()

        return self.model.get_observation(), reward, self.done, {}

    def reset(self):
        """
        Reset the solow model here with the input parameters
        :return:
        """
        # reinit the Solow Model
        self.model = Solow_delong(n=self.n,
                                  s=self.s,
                                  delta=self.delta,
                                  alpha=self.alpha,
                                  g=self.g,
                                  k=self.k,
                                  E=self.E,
                                  L=self.L,
                                  limit_g_low=self.limit_g_low,
                                  limit_g_high=self.limit_g_high,
                                  )

        # init state parameters
        self.time_step = 0
        self.done = False

        return self.model.get_observation()
