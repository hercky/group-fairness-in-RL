import numpy as np

class Env(object):
    """
    Abstract Environment wrapper.
    """
    def __init__(self, seed):
        """
        :param seed: A seed for the random number generator.
        """
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)
