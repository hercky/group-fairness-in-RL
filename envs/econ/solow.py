"""
Code based on: https://github.com/braddelong/LS2019/blob/master/Basic-Solow-Model-delong.ipynb

"""

import numpy as np

class Solow_delong:
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
                 limit_g_low=0.0,
                 limit_g_high=1.0,
                 ):

        self.n, self.s, self.delta, self.alpha, self.g = n, s, delta, alpha, g
        self.k, self.E, self.L = k, E, L
        self.Y = self.k ** (self.alpha / (1 - self.alpha)) * self.E * self.L
        self.K = self.k * self.Y
        self.y = self.Y / self.L
        self.alpha1 = 1 - ((1 - np.exp((self.alpha - 1) * (self.n + self.g + self.delta))) / (self.n + self.g + self.delta))
        self.initdata = vars(self).copy()
        self.limit_g_low = limit_g_low
        self.limit_g_high = limit_g_high

    def calc_next_period_kappa(self):
        "Calculate the next period capital-output ratio."
        # Unpack parameters (get rid of self to simplify notation)
        n, s, delta, alpha1, g, k = self.n, self.s, self.delta, self.alpha1, self.g, self.k
        # Apply the update rule
        return (k + (1 - alpha1) * (s - (n + g + delta) * k))

    def calc_next_period_E(self):
        "Calculate the next period efficiency of labor."
        # Unpack parameters (get rid of self to simplify notation)
        E, g = self.E, self.g
        # Apply the update rule
        return (E * np.exp(g))

    def calc_next_period_L(self):
        "Calculate the next period labor force."
        # Unpack parameters (get rid of self to simplify notation)
        n, L = self.n, self.L
        # Apply the update rule
        return (L * np.exp(n))

    def update(self):
        "Update the current state."
        self.k = self.calc_next_period_kappa()
        self.E = self.calc_next_period_E()
        self.L = self.calc_next_period_L()
        self.Y = self.k ** (self.alpha / (1 - self.alpha)) * self.E * self.L
        self.K = self.k * self.Y
        self.y = self.Y / self.L

    def steady_state(self):
        "Compute the steady state value of the capital-output ratio."
        # Unpack parameters (get rid of self to simplify notation)
        n, s, delta, g = self.n, self.s, self.delta, self.g
        # Compute and return steady state
        return (s / (n + g + delta))

    def generate_sequence(self, t, var='k', init=True):
        "Generate and return time series of selected variable. Variable is κ by default. Start from t=0 by default."
        path = []

        # initialize data
        if init == True:
            for para in self.initdata:
                setattr(self, para, self.initdata[para])

        for i in range(t):
            path.append(vars(self)[var])
            self.update()

        return path

    def update_investment(self, dg):
        """
        acti
        :param action:
        :return:
        """
        self.g = self.g + dg[0]

        # clip the g in range [0,1]
        self.g = min(max(self.g, self.limit_g_low), self.limit_g_high)

    def get_observation(self):
        return np.array([self.E, self.L, self.Y, self.K])


    def get_reward(self):
        return self.k
